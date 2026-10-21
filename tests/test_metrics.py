"""
Tests for evaluation metrics and FPR calculation.
Compatible with both unittest and pytest.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_jailbreak_metrics


class TestMetrics(unittest.TestCase):
    """Verifies correctness of evaluation metrics and false positive rate."""

    def test_perfect_metrics(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        report = calculate_jailbreak_metrics(y_true, y_pred)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.precision, 1.0)
        self.assertEqual(report.recall, 1.0)
        self.assertEqual(report.f1, 1.0)
        self.assertEqual(report.false_positive_rate, 0.0)
        self.assertEqual(report.true_positives, 2)
        self.assertEqual(report.true_negatives, 2)
        self.assertEqual(report.false_positives, 0)
        self.assertEqual(report.false_negatives, 0)

    def test_false_positive_rate_calculation(self):
        # 10 samples: 5 negative (0), 5 positive (1)
        # Predict 2 false positives out of 5 actual negatives -> FPR = 2/5 = 0.4
        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  # 2 FP, 1 FN
        report = calculate_jailbreak_metrics(y_true, y_pred)

        self.assertEqual(report.false_positives, 2)
        self.assertEqual(report.true_negatives, 3)
        self.assertEqual(report.false_positive_rate, 0.4)
        self.assertEqual(report.recall, 0.8)


if __name__ == "__main__":
    unittest.main()
