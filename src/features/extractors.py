"""
Feature extraction implementations: TF-IDF and structural / entropy markers.
"""

import math
import re
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer



INSTRUCTION_CONFLICT_KEYWORDS = [
    r"\bignore\b.*\b(previous|prior|above|system)\b",
    r"\bdisregard\b.*\b(rules|guidelines|instructions)\b",
    r"\byou are now\b",
    r"\bdeveloper mode\b",
    r"\bDAN\b",
    r"\bjailbreak\b",
    r"\bdo anything now\b",
    r"\bpretend to be\b",
    r"\bunrestricted\b",
    r"\bno safety\b"
]


def calculate_shannon_entropy(text: str) -> float:
    """Calculates byte-level or char-level Shannon entropy of the string."""
    if not text:
        return 0.0
    char_counts: Dict[str, int] = {}
    for c in text:
        char_counts[c] = char_counts.get(c, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


class PromptStructuralFeatureExtractor:
    """
    Extracts scalar structural, linguistic, and conflict markers from text.
    Implements character-diversity ratio R_div (Eq. 8), prompt length,
    non-alphanumeric ratio, entropy, and instruction-conflict counts.
    """

    def __init__(self):
        self.conflict_regexes = [re.compile(p, re.IGNORECASE) for p in INSTRUCTION_CONFLICT_KEYWORDS]

    def extract_single(self, text: str) -> np.ndarray:
        """
        Extracts fixed-length feature vector for a single prompt.
        Feature indices:
          0: Length L_i
          1: Character-diversity ratio R_div(x_i) = |{c in x_i}| / L_i (Eq. 8)
          2: Shannon entropy
          3: Uppercase character ratio
          4: Digit character ratio
          5: Non-alphanumeric character ratio (punctuation / symbols)
          6: Instruction conflict pattern match count
        """
        length = len(text)
        if length == 0:
            return np.zeros(7, dtype=np.float32)

        # Character diversity ratio: R_div(x_i) = |{c : c in x_i}| / L_i (Eq. 8)
        char_diversity_ratio = float(len(set(text)) / length)
        entropy = calculate_shannon_entropy(text)
        upper_ratio = sum(1 for c in text if c.isupper()) / length
        digit_ratio = sum(1 for c in text if c.isdigit()) / length
        punct_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / length
        conflict_matches = sum(1 for r in self.conflict_regexes if r.search(text))

        return np.array([
            float(length),
            char_diversity_ratio,
            entropy,
            upper_ratio,
            digit_ratio,
            punct_ratio,
            float(conflict_matches)
        ], dtype=np.float32)

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """Extracts feature matrix for a batch of prompts."""
        return np.stack([self.extract_single(t) for t in texts], axis=0)


class TfidfFeatureExtractor:
    """Configurable TF-IDF feature extractor wrapping scikit-learn."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 20000,
        sublinear_tf: bool = True,
        min_df: int = 2,
        analyzer: str = "word"
    ):
        self.analyzer = analyzer
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            lowercase=True,
            analyzer=analyzer
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "TfidfFeatureExtractor":
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]):
        if not self.is_fitted:
            raise RuntimeError("TfidfFeatureExtractor must be fitted before transforming.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        self.is_fitted = True
        return self.vectorizer.fit_transform(texts)


class HybridFeatureExtractor:
    """
    Combined feature extractor implementing Equation (6) from research paper:
    z_i = [ z_i^{(w)} || z_i^{(c)} || z_i^{(s)} ]
    where:
      z_i^{(w)} : word-level token patterns (word TF-IDF, Eq. 7)
      z_i^{(c)} : character-level component (char n-gram TF-IDF)
      z_i^{(s)} : structural component (length, diversity ratio Eq. 8, entropy, conflict markers)
    """

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 2),
        word_max_features: int = 20000,
        char_ngram_range: Tuple[int, int] = (2, 5),
        char_max_features: int = 5000,
        min_df: int = 2,
        sublinear_tf: bool = True,
        include_char: bool = True,
        include_structural: bool = True
    ):
        self.include_char = include_char
        self.include_structural = include_structural


        # Word-level component z^{(w)}
        self.word_extractor = TfidfFeatureExtractor(
            ngram_range=word_ngram_range,
            max_features=word_max_features,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            analyzer="word"
        )

        # Character-level component z^{(c)}
        if self.include_char:
            self.char_extractor = TfidfFeatureExtractor(
                ngram_range=char_ngram_range,
                max_features=char_max_features,
                sublinear_tf=sublinear_tf,
                min_df=min_df,
                analyzer="char"
            )
        else:
            self.char_extractor = None

        # Structural component z^{(s)}
        if self.include_structural:
            self.structural_extractor = PromptStructuralFeatureExtractor()
        else:
            self.structural_extractor = None

        self.is_fitted = False

    def fit(self, texts: List[str]) -> "HybridFeatureExtractor":
        self.word_extractor.fit(texts)
        if self.include_char and self.char_extractor:
            self.char_extractor.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]):
        if not self.is_fitted:
            raise RuntimeError("HybridFeatureExtractor must be fitted before transforming.")
        import scipy.sparse as sp

        # 1. Word TF-IDF: z^{(w)}
        X_word = self.word_extractor.transform(texts)
        parts = [X_word]

        # 2. Character TF-IDF: z^{(c)}
        if self.include_char and self.char_extractor:
            X_char = self.char_extractor.transform(texts)
            parts.append(X_char)

        # 3. Structural features: z^{(s)}
        if self.include_structural and self.structural_extractor:
            struct_feats = self.structural_extractor.extract_batch(texts)
            X_struct = sp.csr_matrix(struct_feats)
            parts.append(X_struct)

        # Concatenate: z_i = [z_i^{(w)} || z_i^{(c)} || z_i^{(s)}] (Eq. 6)
        if len(parts) == 1:
            return parts[0]
        return sp.hstack(parts, format="csr")

    def fit_transform(self, texts: List[str]):
        self.fit(texts)
        return self.transform(texts)
