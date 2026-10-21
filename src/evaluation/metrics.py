"""
Evaluation metrics calculation for jailbreak detection models.
Implements Equations (14)-(16) and (20)-(23) from research paper:
- Accuracy (Eq. 20)
- Precision & Recall (Eq. 21)
- F1-score (Eq. 22)
- Positive prediction rate R_positive (Eq. 23)
- Macro-averaged F1 and class-wise metrics
- Expected Calibration Error (ECE, Eq. 16)
- Three-way risk decision policy (Eq. 14, 15): allow, flag, block
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from pydantic import BaseModel


class ClassMetric(BaseModel):
    """Class-wise performance metric container."""
    precision: float
    recall: float
    f1: float
    support: int


class ThreeWayPolicyReport(BaseModel):
    """Three-way decision policy summary: allow, flag, block (Eq. 14-15)."""
    tau_low: float
    tau_high: float
    allow_count: int
    flag_count: int
    block_count: int
    allow_ratio: float
    flag_ratio: float
    block_ratio: float


class EvaluationReport(BaseModel):
    """Structured container for jailbreak detection metrics matching research paper."""
    total_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    positive_prediction_rate: float
    macro_f1: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    class_metrics: Dict[str, ClassMetric]
    roc_auc: Optional[float] = None
    expected_calibration_error: Optional[float] = None
    brier_score: Optional[float] = None
    three_way_policy: Optional[ThreeWayPolicyReport] = None


def calculate_ece(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across probability bins (Eq. 16):
    ECE = sum_{m=1}^M (|S_m| / n) * |acc(S_m) - conf(S_m)|
    """
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


def classify_three_way_decision(
    probs: Union[np.ndarray, list],
    tau_low: float = 0.30,
    tau_high: float = 0.70
) -> np.ndarray:
    """
    Three-way decision policy (Eq. 14, 15):
    d(p_i) =
      allow,  p_i < tau_low
      flag,   tau_low <= p_i < tau_high
      block,  p_i >= tau_high
    where 0 <= tau_low < tau_high <= 1.
    """
    p = np.asarray(probs, dtype=np.float32)
    decisions = np.empty(len(p), dtype=object)
    decisions[p < tau_low] = "allow"
    decisions[(p >= tau_low) & (p < tau_high)] = "flag"
    decisions[p >= tau_high] = "block"
    return decisions


def evaluate_three_way_policy(
    probs: Union[np.ndarray, list],
    tau_low: float = 0.30,
    tau_high: float = 0.70
) -> ThreeWayPolicyReport:
    """Calculates distribution of decisions under three-way policy (Eq. 14-15)."""
    decisions = classify_three_way_decision(probs, tau_low=tau_low, tau_high=tau_high)
    total = len(decisions)
    allow_c = int(np.sum(decisions == "allow"))
    flag_c = int(np.sum(decisions == "flag"))
    block_c = int(np.sum(decisions == "block"))

    return ThreeWayPolicyReport(
        tau_low=round(tau_low, 3),
        tau_high=round(tau_high, 3),
        allow_count=allow_c,
        flag_count=flag_c,
        block_count=block_c,
        allow_ratio=round(allow_c / total, 4) if total > 0 else 0.0,
        flag_ratio=round(flag_c / total, 4) if total > 0 else 0.0,
        block_ratio=round(block_c / total, 4) if total > 0 else 0.0
    )


def calculate_jailbreak_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_scores: Optional[Union[np.ndarray, list]] = None,
    tau_low: float = 0.30,
    tau_high: float = 0.70
) -> EvaluationReport:
    """
    Computes rigorous classification metrics for LLM jailbreak detection
    matching research paper Section III-G (Equations 20-23) and Section III-E (Eq 14-16).

    Args:
        y_true: Ground truth binary labels (0 = benign, 1 = jailbreak).
        y_pred: Model binary predictions (0 = benign, 1 = jailbreak).
        y_scores: Optional predicted probabilities for positive class.
        tau_low: Lower threshold for three-way decision policy.
        tau_high: Upper threshold for three-way decision policy.

    Returns:
        EvaluationReport dataclass with accuracy, precision, recall, f1, R_positive, macro_f1,
        class-wise metrics, ECE, and three-way policy counts.
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_pred, dtype=np.int32)
    n = len(y_t)

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Accuracy: (TP + TN) / (TP + TN + FP + FN) (Eq. 20)
    acc = float(accuracy_score(y_t, y_p))

    # Precision & Recall (Eq. 21)
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))

    # F1-score (Eq. 22)
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # Positive prediction rate: R_positive = (TP + FP) / (TP + TN + FP + FN) (Eq. 23)
    r_pos = float((tp + fp) / n) if n > 0 else 0.0

    # False Positive Rate (FPR) = FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Macro-averaged F1
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    # Class-wise metrics
    cr = classification_report(y_t, y_p, labels=[0, 1], output_dict=True, zero_division=0)
    class_metrics = {
        "benign_class_0": ClassMetric(
            precision=round(float(cr["0"]["precision"]), 4),
            recall=round(float(cr["0"]["recall"]), 4),
            f1=round(float(cr["0"]["f1-score"]), 4),
            support=int(cr["0"]["support"])
        ),
        "malicious_class_1": ClassMetric(
            precision=round(float(cr["1"]["precision"]), 4),
            recall=round(float(cr["1"]["recall"]), 4),
            f1=round(float(cr["1"]["f1-score"]), 4),
            support=int(cr["1"]["support"])
        )
    }

    # Probability-dependent metrics (ROC-AUC, ECE, Brier score, 3-way policy)
    roc_auc = None
    ece = None
    brier = None
    three_way_report = None

    if y_scores is not None:
        scores = np.asarray(y_scores, dtype=np.float32)
        if len(np.unique(y_t)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_t, scores))
            except Exception:
                roc_auc = None

        ece = calculate_ece(y_t, scores)
        brier = float(round(float(np.mean((scores - y_t) ** 2)), 4))
        three_way_report = evaluate_three_way_policy(scores, tau_low=tau_low, tau_high=tau_high)

    return EvaluationReport(
        total_samples=n,
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        positive_prediction_rate=round(r_pos, 4),
        macro_f1=round(macro_f1, 4),
        false_positive_rate=round(fpr, 4),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        class_metrics=class_metrics,
        roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
        expected_calibration_error=ece,
        brier_score=brier,
        three_way_policy=three_way_report
    )

