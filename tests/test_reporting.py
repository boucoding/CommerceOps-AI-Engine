import pandas as pd
import pytest

from src.loaders import load_emails, load_products
from src.models.domain import STATUS_CREATED, EmailOutcome, EmailStatus, OrderLine
from src.utils.reporter import build_report, export_report, summarize


def test_sample_data_loads_cleanly():
    products = load_products("data/catalog_products.csv")
    emails = load_emails("data/inbound_customer_emails.csv")
    assert len(products) == 99 and len({p.product_id for p in products}) == 99
    assert len(emails) == 23 and all(e.subject != "nan" for e in emails)


def test_loader_rejects_bad_catalogs(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("product_id,name,category,description,stock,seasons,price\nA1,x,y,z,-1,s,1\n")
    with pytest.raises(ValueError, match="negative"):
        load_products(str(bad))


def test_report_keeps_required_sheets(tmp_path):
    outcomes = [
        EmailOutcome(
            "E1",
            EmailStatus.PROCESSED,
            category="order request",
            response="ok",
            lines=[OrderLine("SWL2345", "Sleek Wallet", 1, 1, STATUS_CREATED, 4)],
            plan={"steps": [{"tool": "reserve_stock"}]},
            trace=[{"type": "tool", "tool": "reserve_stock"}],
        ),
        EmailOutcome("E2", EmailStatus.PROCESSED, category="product inquiry", response="hi"),
        EmailOutcome("E3", EmailStatus.ESCALATED, guardrail_flags=["instruction_override"]),
    ]
    paths = export_report(build_report(outcomes), str(tmp_path))
    sheets = pd.read_excel(paths["workbook"], sheet_name=None)
    assert list(sheets) == [
        "email-classification",
        "order-status",
        "order-response",
        "inquiry-response",
        "processing-audit",
    ]
    assert sheets["order-status"].iloc[0].tolist() == ["E1", "SWL2345", 1, "created"]
    assert sheets["email-classification"]["category"].tolist() == ["order request", "product inquiry", "unclassified"]
    assert sheets["processing-audit"]["tool calls"].tolist()[0] == "reserve_stock"
    summary = summarize(outcomes, 1.0, 2.0)
    assert summary["escalated"] == 1 and summary["order_lines"] == 1 and "estimated_cost_usd" in summary
