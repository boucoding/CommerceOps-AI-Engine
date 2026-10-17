"""
Gold-label evaluation of the agent on the bundled sample emails.

Runs against a freshly reset database so stock starts from the catalog file. In
`live` mode it measures the configured model; in `replay` mode it re-scores a
recorded run deterministically (used by CI).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from src.config import Settings
from src.engine import build_engine
from src.loaders import load_emails
from src.models.domain import ORDER_REQUEST, EmailOutcome
from src.safety.reply_validation import cited_product_ids


@dataclass
class EvalResult:
    mode: str
    model: str
    metrics: Dict[str, float]
    counts: Dict[str, Any]
    failures: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        rows = "\n".join(f"| {k} | {v:.1%} |" for k, v in self.metrics.items())
        counts = "\n".join(f"| {k} | {v} |" for k, v in self.counts.items())
        failures = "\n".join(f"- {f}" for f in self.failures) or "- none"
        return (
            f"## Evaluation: {self.model} ({self.mode})\n\n| Metric | Score |\n|---|---|\n{rows}\n\n"
            f"| Count | Value |\n|---|---|\n{counts}\n\n### Mismatches\n{failures}\n"
        )


def _ratio(num: float, den: float) -> float:
    return num / den if den else 1.0


def plan_adherence(outcome: EmailOutcome) -> float:
    """Share of planned steps whose tool was actually executed (multiset match, order-insensitive)."""
    planned = [s["tool"] for s in (outcome.plan or {}).get("steps", [])]
    executed = list(outcome.tool_calls)
    hits = 0
    for tool in planned:
        if tool in executed:
            executed.remove(tool)
            hits += 1
    return _ratio(hits, len(planned))


def evaluate(outcomes: Sequence[EmailOutcome], gold: Dict[str, Dict[str, Any]], mode: str, model: str) -> EvalResult:
    by_id = {o.email_id: o for o in outcomes}
    failures: List[str] = []
    n = len(gold)
    intent_ok = name_ok = status_ok = lang_n = lang_ok = 0
    tp = fp = fn = exact = orders = ctx_hit = ctx_total = grounded = 0
    adherence: List[float] = []

    for email_id, want in gold.items():
        o = by_id.get(email_id)
        if o is None:
            failures.append(f"{email_id}: missing outcome")
            continue
        adherence.append(plan_adherence(o))

        def check(ok: bool, label: str, got: Any, expected: Any, eid: str = email_id) -> int:
            if not ok:
                failures.append(f"{eid}: {label} {got!r} != {expected!r}")
            return int(ok)

        intent_ok += check(o.category == want["category"], "category", o.category, want["category"])
        name_ok += check(
            o.customer_name == want.get("customer_name"), "customer_name", o.customer_name, want.get("customer_name")
        )
        expected_status = want.get("status", "processed")
        status_ok += check(o.status.value == expected_status, "status", o.status.value, expected_status)
        if "language" in want:
            lang_n += 1
            lang_ok += check(o.language == want["language"], "language", o.language, want["language"])

        if want["category"] == ORDER_REQUEST:
            orders += 1
            expected_lines = {tuple(line) for line in want.get("lines", [])}
            got_lines = {(ln.product_id, ln.requested_quantity, ln.status) for ln in o.lines}
            tp += len(expected_lines & got_lines)
            fp += len(got_lines - expected_lines)
            fn += len(expected_lines - got_lines)
            exact += check(expected_lines == got_lines, "lines", sorted(got_lines), sorted(expected_lines))

        for pid in want.get("context", []):
            ctx_total += 1
            ctx_hit += check(pid in o.context_product_ids, "context is missing", pid, "found by a tool")

        if o.response is not None:
            grounded += int(set(cited_product_ids(o.response)) <= set(o.context_product_ids))

    answered = sum(o.response is not None for o in outcomes)
    metrics = {
        "intent accuracy": _ratio(intent_ok, n),
        "order line precision": _ratio(tp, tp + fp),
        "order line recall": _ratio(tp, tp + fn),
        "orders fully correct": _ratio(exact, orders),
        "email status accuracy": _ratio(status_ok, n),
        "customer name accuracy": _ratio(name_ok, n),
        "reply language accuracy": _ratio(lang_ok, lang_n),
        "context recall": _ratio(ctx_hit, ctx_total),
        "replies fully grounded": _ratio(grounded, answered),
        "plan adherence": _ratio(sum(adherence), len(adherence)),
    }
    counts = {
        "emails": n,
        "tool calls": sum(len(o.tool_calls) for o in outcomes),
        "replans": sum(o.replans for o in outcomes),
        "SQL queries": sum(o.tool_calls.count("query_database") for o in outcomes),
        "needs review / failed": sum(o.status.value in ("needs_review", "failed") for o in outcomes),
        "LLM calls": sum(o.usage.llm_calls for o in outcomes),
        "prompt tokens": sum(o.usage.prompt_tokens for o in outcomes),
        "completion tokens": sum(o.usage.completion_tokens for o in outcomes),
    }
    return EvalResult(mode, model, metrics, counts, failures)


def run_evaluation(settings: Settings, gold_path: str = "evals/gold_labels.json") -> EvalResult:
    with open(gold_path, encoding="utf-8") as fh:
        gold = json.load(fh)["emails"]
    engine = build_engine(settings.model_copy(update={"BATCH_WORKERS": 1}), reset=True)
    try:
        outcomes = engine.process(load_emails(settings.INBOUND_EMAILS_CSV_PATH))
    finally:
        engine.close()
    return evaluate(outcomes, gold, settings.LLM_MODE.value, engine.llm.model)
