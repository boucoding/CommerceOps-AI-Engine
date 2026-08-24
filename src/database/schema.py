"""
Database Schema Bootstrap & Product Seeder.
"""

import os
from typing import List
import pandas as pd
from psycopg2.extras import execute_values
from src.database.connection import db_manager
from src.models.domain import Product
from src.utils.logger import logger

def initialize_database(schema_file: str = "init_db.sql") -> bool:
    """Executes the DDL script to create extensions, tables, and indexes."""
    if not os.path.exists(schema_file):
        logger.error(f"Schema DDL file not found at {schema_file}")
        return False

    with open(schema_file, "r", encoding="utf-8") as f:
        ddl = f.read()

    with db_manager.get_connection() as conn:
        if conn is None:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
            logger.info("Database schema, pgvector, and pg_trgm indexes initialized successfully.")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize database schema: {e}")
            return False

def sync_products_to_database(products: List[Product]) -> bool:
    """Upserts product records and embeddings into PostgreSQL."""
    with db_manager.get_connection() as conn:
        if conn is None:
            return False

        try:
            with conn.cursor() as cur:
                records = []
                for p in products:
                    emb_str = f"[{','.join(map(str, p.embedding))}]" if p.embedding else None
                    records.append((
                        p.product_id,
                        p.name,
                        p.category,
                        p.description,
                        p.stock,
                        p.seasons,
                        p.price,
                        emb_str
                    ))

                query = """
                INSERT INTO products (product_id, name, category, description, stock, seasons, price, embedding)
                VALUES %s
                ON CONFLICT (product_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    category = EXCLUDED.category,
                    description = EXCLUDED.description,
                    stock = EXCLUDED.stock,
                    seasons = EXCLUDED.seasons,
                    price = EXCLUDED.price,
                    embedding = COALESCE(EXCLUDED.embedding, products.embedding),
                    updated_at = CURRENT_TIMESTAMP;
                """
                execute_values(cur, query, records)
            conn.commit()
            logger.info(f"Synchronized {len(products)} products to PostgreSQL.")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error syncing products to database: {e}")
            return False
