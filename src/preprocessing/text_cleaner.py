"""
Text cleaning, normalization, and obfuscation probing module.
"""

import re
import unicodedata
import base64
from typing import List, Dict, Any


ZERO_WIDTH_CHARS = re.compile(r"[\u200B\u200C\u200D\uFEFF\u200E\u200F\u202A-\u202E]")
BASE64_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")
HEX_PATTERN = re.compile(r"(?:(?:0x|\\x)?[0-9a-fA-F]{2}){8,}")


class TextCleaner:
    """Normalizes and cleans raw text prompts."""

    def __init__(self, strip_zero_width: bool = True, normalize_unicode: bool = True):
        self.strip_zero_width = strip_zero_width
        self.normalize_unicode = normalize_unicode

    def clean(self, text: str) -> str:
        """Applies normalization and sanitization to a single prompt string."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        if self.strip_zero_width:
            text = ZERO_WIDTH_CHARS.sub("", text)

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Cleans a batch of prompt strings."""
        return [self.clean(t) for t in texts]


class ObfuscationDetector:
    """Heuristic detector for common obfuscation methods (Base64, Hex, Leetspeak)."""

    @staticmethod
    def detect_base64(text: str, min_len: int = 16) -> bool:
        """Checks for valid Base64 payloads embedded within prompt text."""
        matches = BASE64_PATTERN.findall(text)
        for match in matches:
            if len(match) >= min_len:
                try:
                    decoded = base64.b64decode(match, validate=True)
                    # Check if decoded payload resembles printable text or instructions
                    if len(decoded) >= 8 and any(c.isascii() and c.isprintable() for c in decoded.decode("utf-8", errors="ignore")):
                        return True
                except Exception:
                    continue
        return False

    @staticmethod
    def detect_hex(text: str) -> bool:
        """Checks for contiguous hex byte strings."""
        return bool(HEX_PATTERN.search(text))

    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        """Returns obfuscation analysis summary."""
        return {
            "has_base64": cls.detect_base64(text),
            "has_hex": cls.detect_hex(text),
        }
