"""
Unit Tests for ACID Transactional Inventory Management & Stock Reservations.
"""

from src.inventory.service import InventoryService
from src.models.domain import Product

def test_successful_stock_reservation(inventory_service):
    """Tests reservation when stock >= requested quantity."""
    # SWL2345 has 5 in stock
    result = inventory_service.reserve_stock("SWL2345", 2)
    
    assert result.status == "created"
    assert result.fulfilled_quantity == 2
    assert result.remaining_stock == 3
    
    # Check that subsequent lookup reflects decremented stock
    prod = inventory_service.get_product("SWL2345")
    assert prod.stock == 3

def test_out_of_stock_insufficient_quantity(inventory_service):
    """Tests that requesting more stock than available marks out of stock and keeps stock unchanged."""
    # RSG8901 has 1 in stock
    result = inventory_service.reserve_stock("RSG8901", 3)
    
    assert result.status == "out of stock"
    assert result.fulfilled_quantity == 0
    assert result.remaining_stock == 1
    
    # Verify stock was not decremented
    prod = inventory_service.get_product("RSG8901")
    assert prod.stock == 1

def test_sequential_stock_depletion(inventory_service):
    """Tests that first order takes the stock, and second order gets out of stock."""
    # RSG8901 has stock = 1
    order1 = inventory_service.reserve_stock("RSG8901", 1)
    assert order1.status == "created"
    assert order1.remaining_stock == 0
    
    order2 = inventory_service.reserve_stock("RSG8901", 1)
    assert order2.status == "out of stock"
    assert order2.remaining_stock == 0

def test_unknown_product_reservation(inventory_service):
    """Tests reservation of a product not present in catalog."""
    result = inventory_service.reserve_stock("NONEXISTENT_SKU", 1)
    
    assert result.status == "out of stock"
    assert result.fulfilled_quantity == 0
    assert result.remaining_stock == 0

def test_zero_stock_item_reservation(inventory_service):
    """Tests reservation on an item starting with stock = 0 (e.g. CGN2345)."""
    result = inventory_service.reserve_stock("CGN2345", 5)
    
    assert result.status == "out of stock"
    assert result.fulfilled_quantity == 0
    assert result.remaining_stock == 0
