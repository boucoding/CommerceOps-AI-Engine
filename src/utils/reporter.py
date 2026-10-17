"""
Batch reporting: the four operational tables plus a processing audit, exported to
one Excel workbook and individual CSV files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.models.domain import ORDER_REQUEST, EmailOutcome
from src.utils.logger import get_logger

logger = get_logger("reporter")

WORKBOOK_NAME = "commerceops_operations_report.xlsx"
CSV_NAMES = {
    "email-classification": "email_intent_classification.csv",
    "order-status": "order_fulfillment_status.csv",
    "order-response": "order_dispatch_responses.csv",
    "inquiry-response": "customer_inquiry_responses.csv",
    "processing-audit": "processing_audit.csv",
}


@dataclass
class Report:
    classification: pd.DataFrame
    order_status: pd.DataFrame
    order_response: pd.DataFrame
    inquiry_response: pd.DataFrame
    audit: pd.DataFrame

    def sheets(self) -> Dict[str, pd.DataFrame]:
        return {
            "email-classification": self.classification,
            "order-status": self.order_status,
            "order-response": self.order_response,
            "inquiry-response": self.inquiry_response,
            "processing-audit": self.audit,
        }


def build_report(outcomes: Sequence[EmailOutcome]) -> Report:
    classification: List[dict] = []
    status: List[dict] = []
    order_resp: List[dict] = []
    inquiry_resp: List[dict] = []
    audit: List[dict] = []
    for o in outcomes:
        classification.append({"email ID": o.email_id, "category": o.category})
        for line in o.lines:
            status.append(
                {
                    "email ID": o.email_id,
                    "product ID": line.product_id,
                    "quantity": line.requested_quantity,
                    "status": line.status,
                }
            )
        target = order_resp if o.category == ORDER_REQUEST else inquiry_resp
        target.append({"email ID": o.email_id, "response": o.response})
        audit.append(
            {
                "email ID": o.email_id,
                "category": o.category,
                "status": o.status.value,
                "confidence": o.confidence,
                "language": o.language,
                "customer name": o.customer_name or "",
                "order lines": len(o.lines),
                "unresolved mentions": "; ".join(o.unresolved_mentions),
                "context products": ", ".join(o.context_product_ids),
                "guardrail flags": ", ".join(o.guardrail_flags),
                "degraded": o.degraded,
                "degraded reasons": " | ".join(o.degraded_reasons),
                "llm calls": o.usage.llm_calls,
                "replayed calls": o.usage.replayed_calls,
                "prompt tokens": o.usage.prompt_tokens,
                "completion tokens": o.usage.completion_tokens,
                "latency ms": o.latency_ms,
                "duplicate": o.duplicate,
            }
        )
    return Report(
        classification=pd.DataFrame(classification, columns=["email ID", "category"]),
        order_status=pd.DataFrame(status, columns=["email ID", "product ID", "quantity", "status"]),
        order_response=pd.DataFrame(order_resp, columns=["email ID", "response"]),
        inquiry_response=pd.DataFrame(inquiry_resp, columns=["email ID", "response"]),
        audit=pd.DataFrame(audit),
    )


def export_report(report: Report, export_dir: str, include_audit: bool = True) -> Dict[str, str]:
    """Writes the workbook and CSVs. Returns {sheet: csv_path} plus the workbook path under 'workbook'."""
    csv_dir = os.path.join(export_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    sheets = report.sheets()
    if not include_audit:
        sheets.pop("processing-audit")

    workbook = os.path.join(export_dir, WORKBOOK_NAME)
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            column_widths = {"response": 100, "degraded reasons": 60, "context products": 40}
            sheet = writer.sheets[name]
            for idx, column in enumerate(df.columns):
                letter = sheet.cell(row=1, column=idx + 1).column_letter
                sheet.column_dimensions[letter].width = column_widths.get(column, max(12, len(column) + 2))

    paths: Dict[str, str] = {"workbook": workbook}
    for name, df in sheets.items():
        path = os.path.join(csv_dir, CSV_NAMES[name])
        df.to_csv(path, index=False)
        paths[name] = path
    logger.info("Report written to %s (%d sheets)", export_dir, len(sheets))
    return paths


def summarize(
    outcomes: Sequence[EmailOutcome], price_in: Optional[float], price_out: Optional[float]
) -> Dict[str, object]:
    prompt = sum(o.usage.prompt_tokens for o in outcomes)
    completion = sum(o.usage.completion_tokens for o in outcomes)
    summary: Dict[str, object] = {
        "emails": len(outcomes),
        "order_requests": sum(o.category == ORDER_REQUEST for o in outcomes),
        "product_inquiries": sum(o.category != ORDER_REQUEST for o in outcomes),
        "order_lines": sum(len(o.lines) for o in outcomes),
        "needs_clarification": sum(o.status.value == "needs_clarification" for o in outcomes),
        "escalated": sum(o.status.value == "escalated" for o in outcomes),
        "degraded": sum(o.degraded for o in outcomes),
        "duplicates": sum(o.duplicate for o in outcomes),
        "llm_calls": sum(o.usage.llm_calls for o in outcomes),
        "replayed_calls": sum(o.usage.replayed_calls for o in outcomes),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }
    if price_in is not None and price_out is not None:
        summary["estimated_cost_usd"] = round(prompt / 1e6 * price_in + completion / 1e6 * price_out, 4)
    return summary
