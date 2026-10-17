from src.evaluation import evaluate, plan_adherence
from src.models.domain import STATUS_CREATED, EmailOutcome, EmailStatus, OrderLine


def outcome(
    email_id,
    category,
    lines=(),
    status=EmailStatus.PROCESSED,
    tools=(),
    planned=(),
    name=None,
    response="Hello",
    context=(),
):
    return EmailOutcome(
        email_id,
        status,
        category=category,
        response=response,
        language="en",
        customer_name=name,
        lines=list(lines),
        plan={"steps": [{"tool": t} for t in planned]},
        trace=[{"type": "tool", "tool": t} for t in tools],
        context_product_ids=list(context),
    )


def test_plan_adherence():
    o = outcome(
        "E1",
        "order request",
        tools=["search_catalog", "submit_reply"],
        planned=["search_catalog", "reserve_stock", "submit_reply"],
    )
    assert plan_adherence(o) == 2 / 3


def test_evaluate_scores_against_gold():
    gold = {
        "E1": {"category": "order request", "customer_name": "Liz", "lines": [["SWL2345", 2, "created"]]},
        "E2": {"category": "product inquiry", "customer_name": None, "context": ["LTH1098"]},
    }
    outcomes = [
        outcome(
            "E1",
            "order request",
            [OrderLine("SWL2345", "Sleek Wallet", 2, 2, STATUS_CREATED, 3)],
            name="Liz",
            tools=["submit_reply"],
            planned=["submit_reply"],
            response="Your SWL2345",
            context=["SWL2345"],
        ),
        outcome("E2", "order request", response="Try LTH9999", context=["LTH1098"]),
    ]
    result = evaluate(outcomes, gold, "replay", "m")
    assert result.metrics["intent accuracy"] == 0.5
    assert result.metrics["order line recall"] == 1.0
    assert result.metrics["replies fully grounded"] == 0.5
    assert any("E2: category" in f for f in result.failures)
    assert "## Evaluation: m (replay)" in result.to_markdown()
