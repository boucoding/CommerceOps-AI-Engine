"""
Plan-and-execute email agent.

    screen ─► plan ─► execute (tool loop) ─► validated reply
                          │  ▲
                 tool error  │ revised plan
                          ▼  │
                       replan

1. Screen: prompt-injection signatures escalate the email without any model call;
   personal data is masked before anything reaches the model.
2. Plan: one structured-output call produces the intent, the customer's name and
   language, and an ordered list of tool steps with success criteria.
3. Execute: a strict function-calling loop works through the plan, one tool per
   turn. Tools enforce permissions (reserve_stock only for orders, only for products
   looked up in this run) and the run ends only when submit_reply passes validation.
4. Replan: when a tool fails, a replanner sees the goal, plan and history and
   decides to continue, revise the remaining steps, or escalate to a person.

Hard budgets (turns, replans) bound cost; every step is recorded in the trace.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from src.agent.prompts import executor_prompt, planner_prompt, replanner_prompt
from src.agent.schemas import Plan, PlanStep, ReplanDecision
from src.agent.tools import RunContext, Toolbox
from src.config import Settings
from src.llm.base import LLMClient, LLMError
from src.models.domain import ORDER_REQUEST, CustomerEmail, EmailOutcome, EmailStatus, UsageStats
from src.safety.guardrails import ScreenedEmail, SecurityGuardrail
from src.storage.postgres import DuplicateEmailError, PostgresStore
from src.utils.logger import correlation, get_logger

logger = get_logger("agent")

_ARGS_PREVIEW = 300


def _name_in_email(name: Optional[str], text: str) -> Optional[str]:
    """Keep the planner's customer name only if it literally appears in the email (guards against hallucination)."""
    if not name:
        return None
    name = name.strip()
    return name if re.search(rf"\b{re.escape(name)}\b", text) else None


class EmailAgent:
    def __init__(self, llm: LLMClient, store: PostgresStore, toolbox: Toolbox, settings: Settings):
        self.llm = llm
        self.store = store
        self.toolbox = toolbox
        self.settings = settings
        self.guardrail = SecurityGuardrail(allowed_email_domains=[settings.SUPPORT_EMAIL.rsplit("@", 1)[-1]])

    # ------------------------------------------------------------------ public
    def handle(self, email: CustomerEmail) -> EmailOutcome:
        with correlation(email.email_id):
            started = time.perf_counter()
            existing = self.store.get_outcome(email.email_id)
            if existing is not None and existing.status.is_final:
                existing.duplicate = True
                return existing
            try:
                self.store.claim_email(email.email_id)
            except DuplicateEmailError:
                stored = self.store.get_outcome(email.email_id)
                if stored is None:
                    raise
                stored.duplicate = True
                return stored

            outcome = self._run(email)
            outcome.lines = self.store.order_lines(email.email_id)
            outcome.latency_ms = int((time.perf_counter() - started) * 1000)
            self.store.save_outcome(outcome)
            logger.info(
                "Finished: %s (%d tool calls, %d replans, %d ms)",
                outcome.status.value,
                len(outcome.tool_calls),
                outcome.replans,
                outcome.latency_ms,
            )
            return outcome

    # ------------------------------------------------------------------- stages
    def _run(self, email: CustomerEmail) -> EmailOutcome:
        screened = self.guardrail.screen(email.email_id, email.subject, email.body)
        outcome = EmailOutcome(
            email_id=email.email_id, status=EmailStatus.PROCESSED, guardrail_flags=list(screened.flags)
        )
        if not screened.is_safe:
            outcome.status = EmailStatus.ESCALATED
            outcome.trace.append({"type": "guardrail", "flags": screened.flags})
            return outcome

        usage = outcome.usage
        try:
            plan = self._plan(screened, usage)
        except LLMError as exc:
            return self._failed(outcome, f"planning failed: {exc}")
        outcome.plan = plan.model_dump()
        outcome.category = plan.intent
        outcome.language = plan.language
        outcome.customer_name = _name_in_email(plan.customer_name, screened.masked_body)
        outcome.trace.append({"type": "plan", "steps": [s.tool for s in plan.steps]})

        ctx = RunContext(email.email_id, plan.intent, screened.token_map)
        try:
            self._execute(screened, plan, ctx, outcome, usage)
        except LLMError as exc:
            return self._failed(outcome, f"execution failed: {exc}")
        finally:
            outcome.context_product_ids = list(ctx.observed_products)

        if ctx.reply is not None:
            outcome.response = ctx.reply
            if plan.intent == ORDER_REQUEST and not ctx.lines:
                outcome.status = EmailStatus.NEEDS_CLARIFICATION
        elif outcome.status == EmailStatus.PROCESSED:
            outcome.status = EmailStatus.NEEDS_REVIEW
            outcome.error = outcome.error or "step budget exhausted before a valid reply was submitted"
        return outcome

    def _plan(self, screened: ScreenedEmail, usage: UsageStats) -> Plan:
        result = self.llm.parse(
            task="plan",
            system=planner_prompt(self.settings),
            user=f"Email ID: {screened.email_id}\n{screened.prompt_block()}",
            schema=Plan,
        )
        usage.record_call(result.usage.prompt_tokens, result.usage.completion_tokens, result.replayed)
        plan = result.value
        if not plan.steps or plan.steps[-1].tool != "submit_reply":
            plan.steps.append(PlanStep(tool="submit_reply", purpose="Send the reply."))
        return plan

    def _execute(
        self, screened: ScreenedEmail, plan: Plan, ctx: RunContext, outcome: EmailOutcome, usage: UsageStats
    ) -> None:
        tools = self.toolbox.schemas(plan.intent)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": executor_prompt(self.settings)},
            {
                "role": "user",
                "content": (
                    f"Email ID: {screened.email_id}\n{screened.prompt_block()}\n\n"
                    f"Plan:\n{plan.model_dump_json(indent=1)}"
                ),
            },
        ]
        replans = 0
        for turn in range(1, self.settings.AGENT_MAX_STEPS + 1):
            result = self.llm.chat(task="execute", messages=messages, tools=tools)
            usage.record_call(result.usage.prompt_tokens, result.usage.completion_tokens, result.replayed)
            assistant = result.value
            messages.append(assistant.to_message())
            if not assistant.tool_calls:
                messages.append({"role": "user", "content": "Call a tool. Finish with submit_reply."})
                outcome.trace.append({"type": "no_tool_call", "turn": turn})
                continue

            for call in assistant.tool_calls:
                started = time.perf_counter()
                tool_result = self.toolbox.run(call.name, call.arguments, ctx)
                messages.append(tool_result.to_message(call.id))
                outcome.trace.append(
                    {
                        "type": "tool",
                        "turn": turn,
                        "tool": call.name,
                        "ok": tool_result.ok,
                        "arguments": call.arguments[:_ARGS_PREVIEW],
                        "error": tool_result.content.get("error"),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                if call.name == "submit_reply" and tool_result.ok:
                    return
                if not tool_result.ok:
                    if replans >= self.settings.AGENT_MAX_REPLANS:
                        continue  # the executor still sees the error and can self-correct
                    replans += 1
                    decision = self._replan(screened, plan, outcome.trace, usage)
                    outcome.trace.append(
                        {
                            "type": "replan",
                            "action": decision.action,
                            "reason": decision.reason,
                            "steps": [s.tool for s in decision.steps],
                        }
                    )
                    if decision.action == "escalate":
                        outcome.status = EmailStatus.NEEDS_REVIEW
                        outcome.error = f"escalated by replanner: {decision.reason}"
                        return
                    if decision.action == "revise" and decision.steps:
                        revised = json.dumps([s.model_dump() for s in decision.steps], indent=1)
                        messages.append({"role": "user", "content": f"Revised remaining plan:\n{revised}"})

    def _replan(
        self, screened: ScreenedEmail, plan: Plan, trace: List[Dict[str, Any]], usage: UsageStats
    ) -> ReplanDecision:
        history = json.dumps([e for e in trace if e["type"] in ("tool", "replan")], indent=1, ensure_ascii=False)
        result = self.llm.parse(
            task="replan",
            system=replanner_prompt(self.settings),
            user=(
                f"Email ID: {screened.email_id}\n{screened.prompt_block()}\n\n"
                f"Current plan:\n{plan.model_dump_json(indent=1)}\n\nHistory:\n{history}"
            ),
            schema=ReplanDecision,
        )
        usage.record_call(result.usage.prompt_tokens, result.usage.completion_tokens, result.replayed)
        return result.value

    @staticmethod
    def _failed(outcome: EmailOutcome, error: str) -> EmailOutcome:
        logger.error("%s", error)
        outcome.status = EmailStatus.FAILED
        outcome.error = error
        outcome.trace.append({"type": "error", "error": error})
        return outcome
