"""
Gold-label evaluation of the full pipeline on the bundled sample emails.

Always runs against a fresh in-memory store so stock starts from the catalog file
and results are reproducible. The LLM mode decides what is being measured:
``offline`` measures the deterministic rules, ``replay`` measures the pipeline on
the fixture answers, ``live`` measures the configured model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import LLMMode, Settings, StorageBackend
from src.engine import build_engine
from src.loaders import load_emails
from src.models.domain import EmailOutcome

_PLACEHOLDER = re.compile(r"\[(?:[A-Z][A-Za-z'’]*)(?:[ /'’-][A-Za-z'’]+){0,5}\]")
Line = Tuple[str, int, str]


@dataclass
class EvalResult:
    mode: str
    replay_source: Optional[str]
    metrics: Dict[str, float]
    counts: Dict[str, Any]
    failures: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        label = self.mode + (f" ({self.replay_source} fixtures)" if self.replay_source else "")
        rows = "\n".join(f"| {name} | {value:.1%} |" for name, value in self.metrics.items())
        counts = "\n".join(f"| {name} | {value} |" for name, value in self.counts.items())
        failures = "\n".join(f"- {f}" for f in self.failures) or "- none"
        return (
            f"## Evaluation: {label}\n\n| Metric | Score |\n|---|---|\n{rows}\n\n"
            f"| Count | Value |\n|---|---|\n{counts}\n\n### Mismatches\n{failures}\n"
        )


def _ratio(num: int, den: int) -> float:
    return num / den if den else 1.0


def evaluate(
    outcomes: Sequence[EmailOutcome],
    gold: Dict[str, Dict[str, Any]],
    mode: str,
    replay_source: Optional[str],
    signature: str,
) -> EvalResult:
    by_id = {o.email_id: o for o in outcomes}
    failures: List[str] = []
    intent_ok = name_ok = status_ok = lang_n = lang_ok = 0
    exact_orders = order_emails = 0
    tp = fp = fn = 0
    ctx_hit = ctx_total = 0
    leaks = unsigned = 0

    for email_id, expected in gold.items():
        o = by_id.get(email_id)
        if o is None:
            failures.append(f"{email_id}: missing outcome")
            continue
        if o.category == expected["category"]:
            intent_ok += 1
        else:
            failures.append(f"{email_id}: category {o.category!r} != {expected['category']!r}")
        if (o.customer_name or None) == expected.get("customer_name"):
            name_ok += 1
        else:
            failures.append(f"{email_id}: customer_name {o.customer_name!r} != {expected.get('customer_name')!r}")
        expected_status = expected.get("status", "processed")
        if o.status.value == expected_status:
            status_ok += 1
        else:
            failures.append(f"{email_id}: status {o.status.value!r} != {expected_status!r}")
        if "language" in expected:
            lang_n += 1
            lang_ok += o.language == expected["language"]
            if o.language != expected["language"]:
                failures.append(f"{email_id}: language {o.language!r} != {expected['language']!r}")

        if expected["category"] == "order request":
            order_emails += 1
            want = {tuple(line) for line in expected.get("lines", [])}
            got = {(ln.product_id, ln.requested_quantity, ln.status) for ln in o.lines}
            tp += len(want & got)
            fp += len(got - want)
            fn += len(want - got)
            if want == got:
                exact_orders += 1
            else:
                failures.append(f"{email_id}: lines {sorted(got)} != {sorted(want)}")

        for pid in expected.get("context", []):
            ctx_total += 1
            if pid in o.context_product_ids:
                ctx_hit += 1
            else:
                failures.append(f"{email_id}: expected {pid} in reply context")

        if _PLACEHOLDER.search(o.response):
            leaks += 1
            failures.append(f"{email_id}: placeholder leak in reply")
        if signature not in o.response:
            unsigned += 1
            failures.append(f"{email_id}: reply is missing the brand signature")

    n = len(gold)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    metrics = {
        "intent accuracy": _ratio(intent_ok, n),
        "order line precision": precision,
        "order line recall": recall,
        "orders fully correct": _ratio(exact_orders, order_emails),
        "email status accuracy": _ratio(status_ok, n),
        "customer name accuracy": _ratio(name_ok, n),
        "reply language accuracy": _ratio(lang_ok, lang_n),
        "inquiry context recall": _ratio(ctx_hit, ctx_total),
        "replies without placeholder leaks": _ratio(n - leaks, n),
        "replies with brand signature": _ratio(n - unsigned, n),
    }
    counts = {
        "emails": n,
        "degraded (fell back to rules/templates)": sum(o.degraded for o in outcomes),
        "LLM calls": sum(o.usage.llm_calls for o in outcomes),
        "prompt tokens": sum(o.usage.prompt_tokens for o in outcomes),
        "completion tokens": sum(o.usage.completion_tokens for o in outcomes),
    }
    return EvalResult(mode, replay_source, metrics, counts, failures)


def run_evaluation(settings: Settings, mode: LLMMode, gold_path: str = "evals/gold_labels.json") -> EvalResult:
    with open(gold_path, encoding="utf-8") as fh:
        gold = json.load(fh)["emails"]
    eval_settings = settings.model_copy(update={"STORAGE_BACKEND": StorageBackend.MEMORY})
    engine = build_engine(eval_settings, mode=mode)
    try:
        outcomes = engine.orchestrator.process_batch(load_emails(eval_settings.INBOUND_EMAILS_CSV_PATH))
    finally:
        engine.close()
    source = getattr(engine.llm, "source", None) if mode == LLMMode.REPLAY else None
    return evaluate(outcomes, gold, mode.value, source, eval_settings.signature_block)
