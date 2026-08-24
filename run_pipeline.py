"""
Main Execution Script for CommerceOps AI Engine.
Executes the intelligent omnichannel order fulfillment and customer operations workflow.
"""

import os
import sys
from typing import List
import pandas as pd
from openai import OpenAI

from src.config import settings
from src.utils.logger import logger
from src.models.domain import Product, CustomerEmail
from src.database.connection import db_manager
from src.database.schema import initialize_database, sync_products_to_database
from src.pipeline.orchestrator import CommerceOpsOrchestrator
from src.utils.reporter import OperationsReporter

def load_products_from_csv(csv_path: str = settings.CATALOG_CSV_PATH) -> List[Product]:
    """Loads and parses product catalog from CSV."""
    target_path = csv_path
    if not os.path.exists(target_path):
        # Fallback to legacy filename if needed
        target_path = "Solving Business Problems with AI - products.csv"
        if not os.path.exists(target_path):
            logger.error(f"Products CSV not found at {csv_path} or fallback.")
            sys.exit(1)

    df = pd.read_csv(target_path)
    df.columns = [c.strip().lower() for c in df.columns]

    products: List[Product] = []
    for _, row in df.iterrows():
        products.append(Product(
            product_id=str(row['product_id']).strip().upper(),
            name=str(row.get('name', '')).strip(),
            category=str(row.get('category', '')).strip(),
            description=str(row.get('description', '')).strip(),
            stock=int(row.get('stock', 0)),
            seasons=str(row.get('seasons', '')).strip(),
            price=float(row.get('price', 0.0))
        ))
    logger.info(f"Loaded {len(products)} products from catalog ({target_path}).")
    return products

def load_emails_from_csv(csv_path: str = settings.INBOUND_EMAILS_CSV_PATH) -> List[CustomerEmail]:
    """Loads and parses customer email queue from CSV."""
    target_path = csv_path
    if not os.path.exists(target_path):
        target_path = "data.csv"
        if not os.path.exists(target_path):
            logger.error(f"Emails CSV not found at {csv_path} or fallback.")
            sys.exit(1)

    df = pd.read_csv(target_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if 'message' in df.columns and 'body' not in df.columns:
        df['body'] = df['message']

    emails: List[CustomerEmail] = []
    for _, row in df.iterrows():
        emails.append(CustomerEmail(
            email_id=str(row['email_id']).strip(),
            subject=str(row.get('subject', '')).strip(),
            body=str(row.get('body', '')).strip()
        ))
    logger.info(f"Loaded {len(emails)} customer emails for processing ({target_path}).")
    return emails

def main():
    logger.info("==========================================================")
    logger.info(f"Starting {settings.APP_NAME} (Environment: {settings.ENV})")
    logger.info("==========================================================")

    # 1. Load Data
    products = load_products_from_csv()
    emails = load_emails_from_csv()

    # 2. Initialize Database & PgVector if Available
    if db_manager.initialize_pool():
        initialize_database()
        sync_products_to_database(products)

    # 3. Initialize OpenAI Client (Loaded from Environment Variables)
    client_kwargs = {"api_key": settings.OPENAI_API_KEY or "dummy-key"}
    if settings.OPENAI_BASE_URL:
        client_kwargs["base_url"] = settings.OPENAI_BASE_URL
    client = OpenAI(**client_kwargs)

    # 4. Initialize & Run Orchestrator
    orchestrator = CommerceOpsOrchestrator(products, client)
    df_classification, df_order_status, df_order_response, df_inquiry_response = orchestrator.process_batch(emails)

    # 5. Export Deliverables
    OperationsReporter.export_all(
        df_classification,
        df_order_status,
        df_order_response,
        df_inquiry_response
    )

    logger.info("==========================================================")
    logger.info("CommerceOps Pipeline Execution Completed Successfully!")
    logger.info(f"- Total Emails Classified: {len(df_classification)}")
    logger.info(f"- Total Order Status Lines: {len(df_order_status)}")
    logger.info(f"- Total Order Responses Generated: {len(df_order_response)}")
    logger.info(f"- Total Inquiry Responses Generated: {len(df_inquiry_response)}")
    logger.info(f"- Master Excel Workbook: {settings.EXCEL_OUTPUT_PATH}")
    logger.info("==========================================================")

if __name__ == "__main__":
    main()
