"""
Fuzzy SKU Matcher and String Normalization Engine.
Handles typo recovery (e.g., DHN0987 -> CHN0987), formatted bracketed SKUs, and product alias resolution.
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from src.models.domain import Product
from src.utils.logger import logger

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

class FuzzyProductMatcher:
    """
    High-accuracy fuzzy string matcher with SKU normalization and Levenshtein fallback.
    """
    def __init__(self, products: List[Product]):
        self.products = products
        self.sku_to_product: Dict[str, Product] = {p.product_id.upper(): p for p in products}
        self.name_to_product: Dict[str, Product] = {p.name.lower(): p for p in products}

    @staticmethod
    def normalize_sku_in_text(text: str) -> str:
        """
        Extracts and normalizes spaced/bracketed SKU patterns:
        e.g. '[CBT 89 01]' -> 'CBT8901', 'RSG 89 01' -> 'RSG8901', 'CBT-8901' -> 'CBT8901'
        """
        # Match pattern: 3 uppercase letters followed by 4 digits with optional spaces/dashes/brackets
        def repl(match):
            raw = match.group(0)
            cleaned = re.sub(r'[^A-Za-z0-9]', '', raw).upper()
            return cleaned

        pattern = r'\[?[A-Za-z]{3}[\s\-_]*\d{2}[\s\-_]*\d{2}\]?'
        return re.sub(pattern, repl, text)

    def extract_potential_sku_tokens(self, text: str) -> List[str]:
        """Finds all alphanumeric tokens that look like SKU codes (e.g. CBT8901, DHN0987, FZZ1098)."""
        normalized = self.normalize_sku_in_text(text)
        tokens = re.findall(r'\b[A-Za-z]{3}\d{4}\b', normalized)
        return [t.upper() for t in tokens]

    def match_sku_with_typo_tolerance(self, token: str, score_threshold: float = 75.0) -> Optional[Tuple[Product, float]]:
        """
        Matches a potential SKU token against catalog product IDs.
        Tolerates 1-2 character typos (e.g., DHN0987 -> CHN0987 with score ~85.7).
        """
        token = token.upper()
        # 1. Exact match
        if token in self.sku_to_product:
            return self.sku_to_product[token], 100.0

        # 2. Fuzzy match against catalog SKUs
        best_product = None
        best_score = 0.0

        for sku, prod in self.sku_to_product.items():
            score = self._compute_similarity(token, sku)
            if score > best_score:
                best_score = score
                best_product = prod

        if best_score >= score_threshold and best_product:
            logger.info(f"[Fuzzy SKU Match] Corrected typo '{token}' -> '{best_product.product_id}' ({best_product.name}) with score {best_score:.1f}")
            return best_product, best_score

        return None

    def match_product_name(self, text: str, score_threshold: float = 70.0) -> List[Tuple[Product, float]]:
        """
        Fuzzy matches product names mentioned in text.
        e.g. 'Sleek Wallet' -> SWL2345, 'Chelsea Boots' -> CBT8901.
        """
        normalized_text = text.lower()
        matches = []

        for name, prod in self.name_to_product.items():
            if name in normalized_text:
                matches.append((prod, 100.0))
            else:
                score = self._compute_partial_similarity(name, normalized_text)
                if score >= score_threshold:
                    matches.append((prod, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _compute_similarity(self, s1: str, s2: str) -> float:
        if HAS_RAPIDFUZZ:
            return fuzz.ratio(s1, s2)
        return self._levenshtein_similarity(s1, s2)

    def _compute_partial_similarity(self, s1: str, s2: str) -> float:
        if HAS_RAPIDFUZZ:
            return fuzz.partial_ratio(s1, s2)
        if s1 in s2:
            return 100.0
        return self._levenshtein_similarity(s1, s2)

    @staticmethod
    def _levenshtein_similarity(s1: str, s2: str) -> float:
        """Pure Python Levenshtein distance ratio."""
        if s1 == s2:
            return 100.0
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost # substitution
                )

        distance = dp[len1][len2]
        max_len = max(len1, len2)
        return max(0.0, (1.0 - (distance / max_len)) * 100.0)
