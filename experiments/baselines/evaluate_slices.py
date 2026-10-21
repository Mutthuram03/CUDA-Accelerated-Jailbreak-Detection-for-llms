#!/usr/bin/env python3
"""
CLI tool to evaluate a trained jailbreak detector across fine-grained attack slices.
"""

import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline_classifier import BaselineJailbreakClassifier
from src.evaluation.error_analysis import run_error_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate model across attack taxonomy slices.")
    parser.add_argument("--model-path", type=str, default="saved_models/baseline_tfidf_lr.joblib")
    parser.add_argument("--data-path", type=str, default="data/splits/test.parquet")
    parser.add_argument("--output-json", type=str, default="results/metrics/slice_evaluation.json")

    args = parser.parse_args()
    model_file = REPO_ROOT / args.model_path
    data_file = REPO_ROOT / args.data_path
    output_file = REPO_ROOT / args.output_json

    if not model_file.exists():
        print(f"Error: Model not found at {model_file}. Train model first.")
        return 1
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}.")
        return 1

    print(f"\nLoading model from {model_file}...")
    model = BaselineJailbreakClassifier.load(model_file)

    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    prompts = df["prompt"].tolist()
    y_true = df["label"].to_numpy(dtype=np.int32)

    print(f"Running inference across {len(prompts):,} samples...")
    probs = model.predict_proba(prompts)[:, 1]
    preds = (probs >= model.decision_threshold).astype(np.int32)

    print("Executing taxonomy slicing and error analysis...")
    report = run_error_analysis(prompts, y_true, preds, probs)

    print("\n" + "=" * 70)
    print(" TAXONOMY SLICE EVALUATION BREAKDOWN")
    print("=" * 70)
    print(f"{'Category':<25} {'Count':<8} {'Accuracy':<10} {'F1':<8} {'FPR':<8}")
    print("-" * 70)
    for s in report.slices:
        print(f"{s.category:<25} {s.sample_count:<8} {s.accuracy*100:<9.1f}% {s.f1:<8.3f} {s.false_positive_rate*100:<7.1f}%")

    print("-" * 70)
    print(f"False Positives: {report.false_positives_count} | False Negatives: {report.false_negatives_count}")
    print(f"Expected Calibration Error (ECE): {report.expected_calibration_error:.4f}")
    print(f"Brier Score: {report.brier_score:.4f}")
    print(f"Prompt Length Correlation: {report.length_correlation_score:.4f}")
    print("=" * 70 + "\n")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
    print(f"Report saved to: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
