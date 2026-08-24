"""
End-to-End Orchestrator for CommerceOps AI Engine.
Coordinates intent classification, Two-Stage order extraction, RAG inquiries, and reporting.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
from openai import OpenAI
from src.config import settings
from src.models.domain import CustomerEmail, Product, OrderProcessingResult, InquiryProcessingResult
from src.inventory.service import InventoryService
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.hybrid_vector_store import HybridVectorStore
from src.pipeline.intent_classifier import IntentClassifier
from src.pipeline.order_extractor import OrderExtractor
from src.pipeline.inquiry_responder import InquiryResponder
from src.pipeline.template_engine import TemplatePostProcessor
from src.utils.logger import logger

class CommerceOpsOrchestrator:
    """
    Central Pipeline Orchestrator.
    Manages workflow execution across all customer emails.
    """
    def __init__(self, products: List[Product], client: OpenAI):
        self.products = products
        self.client = client

        # Initialize Subsystems
        self.inventory_service = InventoryService(products)
        self.embedding_service = EmbeddingService(client)
        self.vector_store = HybridVectorStore(products, self.embedding_service)

        # Initialize Pipeline Stages
        self.classifier = IntentClassifier(client)
        self.order_extractor = OrderExtractor(client, self.vector_store, self.inventory_service)
        self.inquiry_responder = InquiryResponder(client, self.vector_store)

    def process_batch(self, emails: List[CustomerEmail]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes end-to-end processing across sequential customer emails.
        Returns the 4 standard output DataFrames:
        1. email_classification_df ('email ID', 'category')
        2. order_status_df ('email ID', 'product ID', 'quantity', 'status')
        3. order_response_df ('email ID', 'response')
        4. inquiry_response_df ('email ID', 'response')
        """
        classification_records = []
        order_status_records = []
        order_response_records = []
        inquiry_response_records = []

        total = len(emails)
        logger.info(f"Starting batch processing of {total} customer communications...")

        for idx, email in enumerate(emails, 1):
            logger.info(f"[{idx}/{total}] Processing Email {email.email_id} | Subject: '{email.subject}'")

            # 1. Intent Classification
            intent = self.classifier.classify(email)
            classification_records.append({
                "email ID": email.email_id,
                "category": intent.category
            })

            # 2. Routing based on Intent
            if intent.category == "order request":
                order_result = self.order_extractor.extract_and_fulfill(email)

                # Record Order Status lines
                for line in order_result.lines:
                    order_status_records.append({
                        "email ID": email.email_id,
                        "product ID": line.product_id,
                        "quantity": line.requested_quantity,
                        "status": line.status
                    })

                # Sanitize & Post-Process Response
                cleaned_response = TemplatePostProcessor.process_response(
                    order_result.response_text,
                    email,
                    customer_name=intent.customer_name
                )
                order_response_records.append({
                    "email ID": email.email_id,
                    "response": cleaned_response
                })

            else:
                inquiry_result = self.inquiry_responder.respond(email)

                # Sanitize & Post-Process Response
                cleaned_response = TemplatePostProcessor.process_response(
                    inquiry_result.response_text,
                    email,
                    customer_name=intent.customer_name
                )
                inquiry_response_records.append({
                    "email ID": email.email_id,
                    "response": cleaned_response
                })

        # Build Output DataFrames
        df_classification = pd.DataFrame(classification_records, columns=["email ID", "category"])
        df_order_status = pd.DataFrame(order_status_records, columns=["email ID", "product ID", "quantity", "status"])
        df_order_response = pd.DataFrame(order_response_records, columns=["email ID", "response"])
        df_inquiry_response = pd.DataFrame(inquiry_response_records, columns=["email ID", "response"])

        logger.info("Batch processing completed successfully.")
        return df_classification, df_order_status, df_order_response, df_inquiry_response
