"""
ACID Transactional Inventory Service for CommerceOps AI Engine.
Supports row-level locking (SELECT FOR UPDATE) and atomic stock reservations.
"""

from typing import Dict, List, Optional, Tuple, Any
import threading
from src.database.connection import db_manager
from src.models.domain import Product, ReservationResult
from src.utils.logger import logger

class InventoryService:
    """
    Stateful & Transactional Inventory Service.
    Enforces atomic stock validation rules:
    - If stock >= requested_quantity: Status 'created', stock decremented.
    - If stock < requested_quantity: Status 'out of stock', stock remains unchanged.
    """
    def __init__(self, initial_products: Optional[List[Product]] = None):
        self._memory_lock = threading.Lock()
        self._memory_store: Dict[str, Product] = {}
        if initial_products:
            for p in initial_products:
                self._memory_store[p.product_id.upper()] = p

    def get_product(self, product_id: str) -> Optional[Product]:
        """Fetches product details from DB or memory fallback."""
        pid = product_id.strip().upper()
        with db_manager.get_connection() as conn:
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT product_id, name, category, description, stock, seasons, price FROM products WHERE product_id = %s;",
                            (pid,)
                        )
                        row = cur.fetchone()
                        if row:
                            return Product(
                                product_id=row[0],
                                name=row[1],
                                category=row[2],
                                description=row[3],
                                stock=int(row[4]),
                                seasons=row[5],
                                price=float(row[6])
                            )
                except Exception as e:
                    logger.error(f"Error fetching product {pid} from DB: {e}")

        # In-memory fallback
        with self._memory_lock:
            return self._memory_store.get(pid)

    def reserve_stock(self, product_id: str, requested_quantity: int) -> ReservationResult:
        """
        Executes atomic reservation with ACID row-locking.
        """
        pid = product_id.strip().upper()

        # 1. Try PostgreSQL ACID Transaction with Row-Level Lock
        with db_manager.get_connection() as conn:
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT product_id, name, stock FROM products WHERE product_id = %s FOR UPDATE;",
                            (pid,)
                        )
                        row = cur.fetchone()
                        if not row:
                            return ReservationResult(
                                product_id=pid,
                                product_name="Unknown Product",
                                requested_quantity=requested_quantity,
                                fulfilled_quantity=0,
                                status="out of stock",
                                remaining_stock=0
                            )

                        db_pid, name, current_stock = row[0], row[1], int(row[2])
                        if current_stock >= requested_quantity and requested_quantity > 0:
                            new_stock = current_stock - requested_quantity
                            cur.execute(
                                "UPDATE products SET stock = %s WHERE product_id = %s;",
                                (new_stock, pid)
                            )
                            conn.commit()
                            logger.info(f"[ACID Stock Reserved] {pid} ('{name}') x{requested_quantity}. Remaining: {new_stock}")
                            return ReservationResult(
                                product_id=pid,
                                product_name=name,
                                requested_quantity=requested_quantity,
                                fulfilled_quantity=requested_quantity,
                                status="created",
                                remaining_stock=new_stock
                            )
                        else:
                            conn.commit()  # Release lock
                            logger.warning(f"[ACID Out of Stock] {pid} ('{name}') requested x{requested_quantity}, available: {current_stock}")
                            return ReservationResult(
                                product_id=pid,
                                product_name=name,
                                requested_quantity=requested_quantity,
                                fulfilled_quantity=0,
                                status="out of stock",
                                remaining_stock=current_stock
                            )
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Database reservation error for {pid}: {e}. Falling back to in-memory store.")

        # 2. Thread-Safe In-Memory Fallback
        with self._memory_lock:
            if pid not in self._memory_store:
                return ReservationResult(
                    product_id=pid,
                    product_name="Unknown Product",
                    requested_quantity=requested_quantity,
                    fulfilled_quantity=0,
                    status="out of stock",
                    remaining_stock=0
                )

            prod = self._memory_store[pid]
            if prod.stock >= requested_quantity and requested_quantity > 0:
                prod.stock -= requested_quantity
                logger.info(f"[Memory Stock Reserved] {pid} ('{prod.name}') x{requested_quantity}. Remaining: {prod.stock}")
                return ReservationResult(
                    product_id=pid,
                    product_name=prod.name,
                    requested_quantity=requested_quantity,
                    fulfilled_quantity=requested_quantity,
                    status="created",
                    remaining_stock=prod.stock
                )
            else:
                logger.warning(f"[Memory Out of Stock] {pid} ('{prod.name}') requested x{requested_quantity}, available: {prod.stock}")
                return ReservationResult(
                    product_id=pid,
                    product_name=prod.name,
                    requested_quantity=requested_quantity,
                    fulfilled_quantity=0,
                    status="out of stock",
                    remaining_stock=prod.stock
                )

    def get_all_products(self) -> List[Product]:
        """Returns all products in catalog."""
        with db_manager.get_connection() as conn:
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT product_id, name, category, description, stock, seasons, price FROM products;")
                        rows = cur.fetchall()
                        if rows:
                            return [
                                Product(
                                    product_id=r[0],
                                    name=r[1],
                                    category=r[2],
                                    description=r[3],
                                    stock=int(r[4]),
                                    seasons=r[5],
                                    price=float(r[6])
                                )
                                for r in rows
                            ]
                except Exception as e:
                    logger.error(f"Error fetching all products from DB: {e}")

        with self._memory_lock:
            return list(self._memory_store.values())
