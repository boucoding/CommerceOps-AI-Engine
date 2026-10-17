"""
CSV loaders for the product catalog and the inbound email queue.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.models.domain import CustomerEmail, Product

_REQUIRED_PRODUCT_COLUMNS = {"product_id", "name", "category", "description", "stock", "seasons", "price"}


def _clean(value: object) -> str:
    """Converts a CSV cell to text; missing cells become '' (never the string 'nan')."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def load_products(path: str) -> List[Product]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Product catalog not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = _REQUIRED_PRODUCT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    products: List[Product] = []
    seen = set()
    for row_no, row in enumerate(df.to_dict("records"), start=2):
        product_id = _clean(row["product_id"]).upper()
        if not product_id:
            raise ValueError(f"{path}:{row_no}: empty product_id")
        if product_id in seen:
            raise ValueError(f"{path}:{row_no}: duplicate product_id {product_id}")
        seen.add(product_id)
        try:
            stock = int(float(_clean(row["stock"]) or 0))
            price = float(_clean(row["price"]) or 0)
        except ValueError as exc:
            raise ValueError(f"{path}:{row_no}: invalid stock/price for {product_id}") from exc
        if stock < 0 or price < 0:
            raise ValueError(f"{path}:{row_no}: negative stock/price for {product_id}")
        products.append(
            Product(
                product_id=product_id,
                name=_clean(row["name"]),
                category=_clean(row["category"]),
                description=_clean(row["description"]),
                stock=stock,
                seasons=_clean(row["seasons"]),
                price=price,
            )
        )
    return products


def load_emails(path: str) -> List[CustomerEmail]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Email queue not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    body_column = "body" if "body" in df.columns else "message"
    if "email_id" not in df.columns or body_column not in df.columns:
        raise ValueError(f"{path} needs 'email_id' and 'message' (or 'body') columns")
    return [
        CustomerEmail(
            email_id=_clean(row["email_id"]),
            subject=_clean(row.get("subject")),
            body=_clean(row[body_column]),
        )
        for row in df.to_dict("records")
    ]
