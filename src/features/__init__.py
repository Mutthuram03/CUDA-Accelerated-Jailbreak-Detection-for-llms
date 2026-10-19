"""
Feature extraction modules for jailbreak prompt representation.
"""

from .extractors import TfidfFeatureExtractor, PromptStructuralFeatureExtractor, HybridFeatureExtractor

__all__ = ["TfidfFeatureExtractor", "PromptStructuralFeatureExtractor", "HybridFeatureExtractor"]

