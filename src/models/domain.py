"""
Core Domain Models for CommerceOps AI Engine.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Product:
    product_id: str
    name: str
    category: str
    description: str
    stock: int
    seasons: str
    price: float
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "stock": self.stock,
            "seasons": self.seasons,
            "price": self.price
        }

    def to_rag_document(self) -> str:
        return (
            f"Product ID: {self.product_id} | Name: {self.name} | "
            f"Category: {self.category} | Seasons: {self.seasons} | Price: ${self.price:.2f} | Stock: {self.stock}\n"
            f"Description: {self.description}"
        )

@dataclass
class CustomerEmail:
    email_id: str
    subject: str
    body: str
    sender_name: Optional[str] = None
    language: str = "en"

@dataclass
class ReservationResult:
    product_id: str
    product_name: str
    requested_quantity: int
    fulfilled_quantity: int
    status: str  # 'created' | 'out of stock'
    remaining_stock: int

@dataclass
class OrderProcessingResult:
    email_id: str
    lines: List[ReservationResult]
    response_text: str
    all_fulfilled: bool
    has_out_of_stock: bool

@dataclass
class InquiryProcessingResult:
    email_id: str
    retrieved_products: List[Product]
    response_text: str
