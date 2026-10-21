"""
Real-time inference pipeline wrapping model, preprocessing, and timing.
Implements Three-Way Decision Policy (Eq. 14, 15) matching research paper Section III-E:
- allow:  p_i < tau_low
- flag:   tau_low <= p_i < tau_high
- block:  p_i >= tau_high
"""

import time
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel
from ..preprocessing.text_cleaner import TextCleaner, ObfuscationDetector


class InferenceResult(BaseModel):
    """Container for prompt prediction result with three-way risk decision."""
    prompt: str
    risk_score: float
    decision: str  # "allow", "flag", "block"
    is_jailbreak: bool
    confidence: float
    obfuscation_detected: bool
    tau_low: float
    tau_high: float
    latency_ms: float


class JailbreakDetectionPipeline:
    """
    End-to-end inference engine with integrated timing and
    three-way decision policy (Eq. 14-15).
    """

    def __init__(
        self,
        model: Any,
        threshold: float = 0.5,
        tau_low: Optional[float] = None,
        tau_high: Optional[float] = None
    ):
        self.model = model
        self.threshold = threshold
        self.tau_low = tau_low if tau_low is not None else getattr(model, "tau_low", 0.30)
        self.tau_high = tau_high if tau_high is not None else getattr(model, "tau_high", 0.70)
        self.cleaner = TextCleaner()
        self.obfuscation_detector = ObfuscationDetector()

    def predict_single(self, prompt: str) -> InferenceResult:
        """
        Evaluates a single prompt, measures execution time, and assigns three-way decision:
        d(p_i) in {allow, flag, block} (Eq. 14, 15).
        """
        start_time = time.perf_counter()

        cleaned = self.cleaner.clean(prompt)
        obfuscation_info = self.obfuscation_detector.analyze(prompt)
        has_obfuscation = bool(obfuscation_info["has_base64"] or obfuscation_info["has_hex"])

        proba = self.model.predict_proba([cleaned])[0]
        jailbreak_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Three-way decision policy: allow, flag, block (Eq. 14, 15)
        if jailbreak_prob >= self.tau_high:
            decision = "block"
        elif jailbreak_prob >= self.tau_low:
            decision = "flag"
        else:
            decision = "allow"

        # If severe obfuscation payload detected on borderline prompt, elevate to at least flag/block
        if has_obfuscation and decision == "allow":
            decision = "flag"

        is_jailbreak = bool(jailbreak_prob >= self.threshold or (has_obfuscation and jailbreak_prob >= self.tau_low))


        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return InferenceResult(
            prompt=prompt,
            risk_score=round(jailbreak_prob, 4),
            decision=decision,
            is_jailbreak=is_jailbreak,
            confidence=round(jailbreak_prob, 4),
            obfuscation_detected=has_obfuscation,
            tau_low=self.tau_low,
            tau_high=self.tau_high,
            latency_ms=round(elapsed_ms, 3)
        )

    def predict_batch(self, prompts: List[str]) -> List[InferenceResult]:
        """Evaluates a batch of prompts."""
        return [self.predict_single(p) for p in prompts]

