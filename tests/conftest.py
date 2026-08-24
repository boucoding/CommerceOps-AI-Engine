"""
Pytest Fixtures and Mock Services for CommerceOps AI Engine Test Suite.
"""

import pytest
from typing import List
from openai import OpenAI
from src.models.domain import Product, CustomerEmail
from src.inventory.service import InventoryService
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.hybrid_vector_store import HybridVectorStore

@pytest.fixture
def sample_products() -> List[Product]:
    return [
        Product(
            product_id="RSG8901",
            name="Retro Sunglasses",
            category="Accessories",
            description="Vintage-inspired shades with nostalgic vibe for summer.",
            stock=1,
            seasons="Spring, Summer",
            price=26.99
        ),
        Product(
            product_id="SWL2345",
            name="Sleek Wallet",
            category="Accessories",
            description="Functional and fashionable minimalist leather wallet.",
            stock=5,
            seasons="All seasons",
            price=30.00
        ),
        Product(
            product_id="CHN0987",
            name="Chunky Knit Beanie",
            category="Accessories",
            description="Thick cozy yarn beanie with oversized fit for winter warmth.",
            stock=2,
            seasons="Fall, Winter",
            price=22.00
        ),
        Product(
            product_id="CBT8901",
            name="Chelsea Boots",
            category="Men's Shoes",
            description="Stylish leather ankle boots with lug sole.",
            stock=2,
            seasons="Fall, Winter",
            price=65.00
        ),
        Product(
            product_id="CLF2109",
            name="Cable Knit Beanie",
            category="Accessories",
            description="Premium wool classic cable knit beanie.",
            stock=2,
            seasons="Winter",
            price=16.00
        ),
        Product(
            product_id="FZZ1098",
            name="Fuzzy Slippers",
            category="Accessories",
            description="Ultra-soft plush fuzzy slippers for lounging.",
            stock=2,
            seasons="Fall, Winter",
            price=29.00
        ),
        Product(
            product_id="LTH1098",
            name="Leather Backpack",
            category="Bags",
            description="Premium leather backpack with padded laptop sleeve and compartments.",
            stock=7,
            seasons="All seasons",
            price=43.99
        ),
        Product(
            product_id="LTH5432",
            name="Leather Tote",
            category="Bags",
            description="Spacious full-grain leather tote with interior pockets for work.",
            stock=6,
            seasons="All seasons",
            price=28.00
        ),
        Product(
            product_id="SLD7654",
            name="Slide Sandals",
            category="Men's Shoes",
            description="Lightweight slide sandals with cushioned footbed.",
            stock=3,
            seasons="Spring, Summer",
            price=22.00
        ),
        Product(
            product_id="CGN2345",
            name="Cargo Pants",
            category="Men's Clothing",
            description="Durable multi-pocket tactical cargo pants.",
            stock=0,
            seasons="All seasons",
            price=45.00
        )
    ]

@pytest.fixture
def mock_embedding_service(sample_products) -> EmbeddingService:
    service = EmbeddingService(client=None)
    return service

@pytest.fixture
def hybrid_vector_store(sample_products, mock_embedding_service) -> HybridVectorStore:
    return HybridVectorStore(sample_products, mock_embedding_service)

@pytest.fixture
def inventory_service(sample_products) -> InventoryService:
    return InventoryService(sample_products)
