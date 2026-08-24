"""
Enterprise Reporter & Multi-Sheet Spreadsheet Exporter.
"""

import os
from typing import Optional
import pandas as pd
from src.config import settings
from src.utils.logger import logger

class OperationsReporter:
    """
    Exports processing results to Excel and CSV formats with clean enterprise naming.
    """

    @classmethod
    def export_all(
        cls,
        df_classification: pd.DataFrame,
        df_order_status: pd.DataFrame,
        df_order_response: pd.DataFrame,
        df_inquiry_response: pd.DataFrame,
        excel_path: Optional[str] = None,
        csv_dir: Optional[str] = None
    ):
        excel_file = excel_path or settings.EXCEL_OUTPUT_PATH
        target_csv_dir = csv_dir or settings.CSV_OUTPUT_DIR

        # Ensure directories exist
        os.makedirs(os.path.dirname(excel_file) or ".", exist_ok=True)
        os.makedirs(target_csv_dir, exist_ok=True)

        # 1. Export Clean Primary Excel Workbook
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df_classification.to_excel(writer, sheet_name='email-classification', index=False)
                df_order_status.to_excel(writer, sheet_name='order-status', index=False)
                df_order_response.to_excel(writer, sheet_name='order-response', index=False)
                df_inquiry_response.to_excel(writer, sheet_name='inquiry-response', index=False)
            logger.info(f"Generated enterprise Excel report: {excel_file}")
            
            # Sync to legacy file for evaluation compatibility
            with pd.ExcelWriter(settings.LEGACY_EXCEL_PATH, engine='openpyxl') as writer:
                df_classification.to_excel(writer, sheet_name='email-classification', index=False)
                df_order_status.to_excel(writer, sheet_name='order-status', index=False)
                df_order_response.to_excel(writer, sheet_name='order-response', index=False)
                df_inquiry_response.to_excel(writer, sheet_name='inquiry-response', index=False)
        except Exception as e:
            logger.error(f"Failed to export Excel workbook: {e}")

        # 2. Export Clean Primary CSV files
        try:
            df_classification.to_csv(os.path.join(target_csv_dir, "email_intent_classification.csv"), index=False)
            df_order_status.to_csv(os.path.join(target_csv_dir, "order_fulfillment_status.csv"), index=False)
            df_order_response.to_csv(os.path.join(target_csv_dir, "order_dispatch_responses.csv"), index=False)
            df_inquiry_response.to_csv(os.path.join(target_csv_dir, "customer_inquiry_responses.csv"), index=False)
            logger.info(f"Exported clean CSV reports to: {target_csv_dir}")

            # Also sync to legacy folder for evaluation compatibility
            os.makedirs(settings.LEGACY_CSV_DIR, exist_ok=True)
            df_classification.to_csv(os.path.join(settings.LEGACY_CSV_DIR, "Solving Business Problems with AI - Output - email-classification.csv"), index=False)
            df_order_status.to_csv(os.path.join(settings.LEGACY_CSV_DIR, "Solving Business Problems with AI - Output - order-status.csv"), index=False)
            df_order_response.to_csv(os.path.join(settings.LEGACY_CSV_DIR, "Solving Business Problems with AI - Output - order-response.csv"), index=False)
            df_inquiry_response.to_csv(os.path.join(settings.LEGACY_CSV_DIR, "Solving Business Problems with AI - Output - inquiry-response.csv"), index=False)
        except Exception as e:
            logger.error(f"Failed to export CSV sheets: {e}")
