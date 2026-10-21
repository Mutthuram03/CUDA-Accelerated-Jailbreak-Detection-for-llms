"""
Baseline TF-IDF + Logistic Regression jailbreak classifier.
"""

from typing import List, Optional, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from ..preprocessing.text_cleaner import TextCleaner
from ..features.extractors import TfidfFeatureExtractor, HybridFeatureExtractor
from ..evaluation.metrics import classify_three_way_decision


class BaselineJailbreakClassifier:
    """
    Transparent, highly reproducible CPU baseline classifier using
    text cleaning, TF-IDF / hybrid n-grams (Eq. 6-8), and L2-regularized Logistic Regression (Eq. 10-13).
    Supports two-way decision threshold (tau) and three-way decision policy (tau_low, tau_high) (Eq. 14-15).
    """

    def __init__(
        self,
        c_param: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
        ngram_range: tuple = (1, 2),
        max_features: int = 20000,
        min_df: int = 2,
        sublinear_tf: bool = True,
        decision_threshold: float = 0.5,
        tau_low: float = 0.30,
        tau_high: float = 0.70,
        feature_mode: str = "word",
        char_ngram_range: tuple = (2, 5),
        char_max_features: int = 8000
    ):
        self.decision_threshold = decision_threshold
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.feature_mode = feature_mode
        self.cleaner = TextCleaner()

        if feature_mode == "hybrid":
            # Implements Equation (6): z_i = [ z_i^{(w)} || z_i^{(c)} || z_i^{(s)} ]
            self.feature_extractor = HybridFeatureExtractor(
                word_ngram_range=ngram_range,
                word_max_features=max_features,
                char_ngram_range=char_ngram_range,
                char_max_features=char_max_features,
                min_df=min_df,
                sublinear_tf=sublinear_tf,
                include_char=True,
                include_structural=True
            )
        else:
            # Word-level TF-IDF baseline (Eq. 7)
            self.feature_extractor = TfidfFeatureExtractor(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                sublinear_tf=sublinear_tf,
                analyzer="word"
            )

        self.classifier = LogisticRegression(
            C=c_param,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
            solver="lbfgs"
        )
        self.is_trained = False

    def fit(self, prompts: List[str], labels: Union[List[int], np.ndarray]) -> "BaselineJailbreakClassifier":
        """Fits vectorizer and logistic regression classifier on training prompts (Eq. 10-13)."""
        cleaned_prompts = self.cleaner.clean_batch(prompts)
        X = self.feature_extractor.fit_transform(cleaned_prompts)
        y = np.asarray(labels, dtype=np.int32)
        self.classifier.fit(X, y)
        self.is_trained = True
        return self

    def predict(self, prompts: List[str]) -> np.ndarray:
        """Returns binary prediction using configured decision_threshold."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict.")
        if self.decision_threshold == 0.5:
            cleaned_prompts = self.cleaner.clean_batch(prompts)
            X = self.feature_extractor.transform(cleaned_prompts)
            return self.classifier.predict(X)
        return self.predict_with_threshold(prompts, self.decision_threshold)

    def predict_proba(self, prompts: List[str]) -> np.ndarray:
        """Returns probabilities for [P(benign), P(jailbreak)] via logistic sigmoid (Eq. 10, 11)."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict_proba.")
        cleaned_prompts = self.cleaner.clean_batch(prompts)
        X = self.feature_extractor.transform(cleaned_prompts)
        return self.classifier.predict_proba(X)

    def predict_with_threshold(self, prompts: List[str], threshold: float) -> np.ndarray:
        """Predicts binary labels by applying custom probability threshold for class 1 (jailbreak)."""
        proba = self.predict_proba(prompts)
        return (proba[:, 1] >= threshold).astype(np.int32)

    def predict_three_way(
        self,
        prompts: List[str],
        tau_low: Optional[float] = None,
        tau_high: Optional[float] = None
    ) -> np.ndarray:
        """
        Executes three-way decision policy (Eq. 14, 15):
        Returns array of 'allow', 'flag', or 'block'.
        """
        tl = tau_low if tau_low is not None else self.tau_low
        th = tau_high if tau_high is not None else self.tau_high
        probs = self.predict_proba(prompts)[:, 1]
        return classify_three_way_decision(probs, tau_low=tl, tau_high=th)


    def save(self, output_path: Union[str, "Path"]) -> None:
        """Persists fitted model artifact to disk via joblib."""
        import joblib
        from pathlib import Path
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, model_path: Union[str, "Path"]) -> "BaselineJailbreakClassifier":
        """Loads serialized model artifact from disk."""
        import joblib
        return joblib.load(model_path)

