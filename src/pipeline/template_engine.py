"""
Deterministic Template Engine, Brand Signature & PII Unmasking Post-Processor.
Eliminates all unpopulated LLM placeholder tags ([Your Name], [Brand Name]) and restores masked PII in outbound communications.
"""

import re
from typing import Optional, Dict
from src.config import settings
from src.models.domain import CustomerEmail
from src.pipeline.guardrails import PIIScrubber

class TemplatePostProcessor:
    """
    Sanitizes, customizes, and unmasks LLM-generated email text:
    1. Replaces generic placeholder brackets ([Your Name], [Brand Name], etc.) with verified brand metadata.
    2. Reversibly unmasks sensitive PII tokens (e.g. {{EMAIL_1}} -> original email) for outbound delivery.
    3. Dynamically injects the customer's extracted name into salutations.
    4. Normalizes sign-offs and signatures.
    """

    @classmethod
    def process_response(
        cls,
        raw_response: str,
        email: CustomerEmail,
        customer_name: Optional[str] = None,
        token_map: Optional[Dict[str, str]] = None
    ) -> str:
        """Applies deterministic replacement, cleans square-bracket tags, and unmasks PII tokens."""
        text = raw_response

        # 1. Identify customer name
        name = customer_name or email.sender_name or cls._extract_name_from_text(email.body) or "Valued Customer"

        # 2. Map of common LLM placeholder variations to production values
        replacements = {
            r'\[Customer(\'?s)? Name\]': name,
            r'\[Your (Full )?Name\]': "Aura Concierge",
            r'\[Your Title(\/Position)?\]': "Senior Customer Concierge",
            r'\[Your Position\]': "Senior Customer Concierge",
            r'\[Brand Name\]': settings.BRAND_NAME,
            r'\[Company Name\]': settings.BRAND_NAME,
            r'\[Store Name\]': settings.BRAND_NAME,
            r'\[Contact Information\]': settings.SUPPORT_EMAIL,
            r'\[Customer Service Team\]': settings.BRAND_SIGNATURE,
            r'\[Support Email\]': settings.SUPPORT_EMAIL,
            r'\[Phone Number\]': settings.PHONE_NUMBER,
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 3. Clean any remaining rogue brackets that look like placeholders [XYZ]
        text = re.sub(r'\[[A-Za-z\s\/\'\-]+\]', '', text)

        # 4. Unmask any PII tokens (GDPR/CCPA reversible restoration)
        if token_map:
            text = PIIScrubber.unmask_pii(text, token_map)

        # 5. Clean extra trailing blank lines or spaces
        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        return text

    @staticmethod
    def _extract_name_from_text(body: str) -> Optional[str]:
        """Extracts customer name from common conversational phrases."""
        patterns = [
            r"my name is ([A-Z][a-z]+)",
            r"this is ([A-Z][a-z]+)",
            r"thanks,?\s+([A-Z][a-z]+)",
            r"thank you,?\s+([A-Z][a-z]+)",
            r"regards,?\s+([A-Z][a-z]+)",
            r"from ([A-Z][a-z]+)"
        ]
        for pat in patterns:
            m = re.search(pat, body, re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
        return None
