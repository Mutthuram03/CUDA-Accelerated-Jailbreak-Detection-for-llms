"""
Fine-grained error analysis, taxonomy slicing, and calibration diagnostics.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel
from .metrics import calculate_jailbreak_metrics, EvaluationReport
from ..data.validator import INSTRUCTION_CONFLICT_PATTERNS, PERSONA_PATTERNS
from ..preprocessing.text_cleaner import ObfuscationDetector, ZERO_WIDTH_CHARS
import re


class CategorySliceResult(BaseModel):
    category: str
    sample_count: int
    positive_count: int
    negative_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float


class ErrorAnalysisReport(BaseModel):
    total_evaluated: int
    false_positives_count: int
    false_negatives_count: int
    slices: List[CategorySliceResult]
    length_correlation_score: float
    expected_calibration_error: float
    brier_score: float


def assign_prompt_category(text: str, label: int) -> str:
    """Assigns fine-grained security taxonomy tag to a prompt."""
    if label == 0:
        return "benign_query"

    detector = ObfuscationDetector()
    if detector.detect_base64(text):
        return "obfuscation_base64"
    if detector.detect_hex(text):
        return "obfuscation_hex"
    if bool(ZERO_WIDTH_CHARS.search(text)):
        return "obfuscation_zero_width"

    conflict_res = [re.compile(p, re.IGNORECASE) for p in INSTRUCTION_CONFLICT_PATTERNS]
    if any(r.search(text) for r in conflict_res):
        return "instruction_conflict"

    persona_res = [re.compile(p, re.IGNORECASE) for p in PERSONA_PATTERNS]
    if any(r.search(text) for r in persona_res):
        return "persona_adoption"

    return "general_malicious"


def calculate_ece(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) across probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)
    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_probs[mask]))
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def run_error_analysis(
    prompts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> ErrorAnalysisReport:
    """
    Performs comprehensive diagnostic slicing, calibration analysis, and artifact checks.
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_pred, dtype=np.int32)
    y_pr = np.asarray(y_probs, dtype=np.float32)

    categories = [assign_prompt_category(p, int(l)) for p, l in zip(prompts, y_t)]
    df = pd.DataFrame({
        "prompt": prompts,
        "true": y_t,
        "pred": y_p,
        "prob": y_pr,
        "category": categories,
        "length": [len(p) for p in prompts]
    })

    # Slice evaluation
    slice_results: List[CategorySliceResult] = []
    for cat_name, group in df.groupby("category"):
        g_true = group["true"].to_numpy()
        g_pred = group["pred"].to_numpy()
        g_probs = group["prob"].to_numpy()

        metrics = calculate_jailbreak_metrics(g_true, g_pred, g_probs)
        slice_results.append(CategorySliceResult(
            category=str(cat_name),
            sample_count=len(group),
            positive_count=int(np.sum(g_true == 1)),
            negative_count=int(np.sum(g_true == 0)),
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            false_positive_rate=metrics.false_positive_rate
        ))

    # Length shortcut correlation
    # Tests if model predicted probability simply correlates with string length
    if len(df) > 1 and df["length"].std() > 0:
        corr = float(np.corrcoef(df["length"], df["prob"])[0, 1])
    else:
        corr = 0.0

    # Calibration
    ece = calculate_ece(y_t, y_pr)
    brier = float(round(np.mean((y_pr - y_t) ** 2), 4))

    fp_count = int(np.sum((y_t == 0) & (y_p == 1)))
    fn_count = int(np.sum((y_t == 1) & (y_p == 0)))

    return ErrorAnalysisReport(
        total_evaluated=len(df),
        false_positives_count=fp_count,
        false_negatives_count=fn_count,
        slices=slice_results,
        length_correlation_score=round(corr, 4),
        expected_calibration_error=ece,
        brier_score=brier
    )
