#!/usr/bin/env python3
"""
Reproducible performance benchmarking harness: sweeps batch sizes,
measures throughput (prompts/sec), latency percentiles (p50, p95, p99),
and CPU vs GPU scaling.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline_classifier import BaselineJailbreakClassifier
from src.models.gpu_classifier import GPUJailbreakClassifier
from src.utils.system_info import get_system_telemetry


# Batch sizes explicitly specified in paper Section III-H
DEFAULT_BATCH_SIZES = [100, 500, 1000, 5000, 10000]


def benchmark_single_batch_size(
    baseline_model: BaselineJailbreakClassifier,
    gpu_model: GPUJailbreakClassifier,
    prompts_pool: List[str],
    batch_size: int,
    n_warmup: int = 5,
    n_trials: int = 25
) -> Dict[str, Any]:
    """
    Executes benchmark for a given batch size on both CPU and GPU backends.
    Measures Throughput Theta (Eq. 24), Speedup S(B) (Eq. 25), and Delta T (%) (Eq. 26).
    """
    # Sample or cycle prompts to exact batch size
    if len(prompts_pool) >= batch_size:
        batch_prompts = prompts_pool[:batch_size]
    else:
        multiplier = (batch_size // len(prompts_pool)) + 1
        batch_prompts = (prompts_pool * multiplier)[:batch_size]

    # Pre-extract sparse feature matrix to isolate classifier execution
    cleaner = baseline_model.cleaner
    extractor = baseline_model.feature_extractor
    cleaned = cleaner.clean_batch(batch_prompts)
    X_sparse = extractor.transform(cleaned)

    # 1. CPU Warmup & Benchmarking
    for _ in range(n_warmup):
        _ = baseline_model.classifier.predict_proba(X_sparse)

    cpu_latencies_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = baseline_model.classifier.predict_proba(X_sparse)
        cpu_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    # 2. GPU Warmup & Benchmarking
    for _ in range(n_warmup):
        _ = gpu_model.forward_features(X_sparse)

    gpu_latencies_ms = []
    gpu_decomps = []
    for _ in range(n_trials):
        _, decomp = gpu_model.forward_features_decomposed(X_sparse)
        gpu_latencies_ms.append(decomp["t_gpu_total_ms"])
        gpu_decomps.append(decomp)

    cpu_arr = np.array(cpu_latencies_ms)
    gpu_arr = np.array(gpu_latencies_ms)

    cpu_mean = float(np.mean(cpu_arr))
    gpu_mean = float(np.mean(gpu_arr))

    # Throughput Theta = B / T (Eq. 24) [prompts/sec]
    cpu_throughput = round(batch_size / (cpu_mean / 1000.0), 1)
    gpu_throughput = round(batch_size / (gpu_mean / 1000.0), 1)

    # GPU Speedup S(B) = T_CPU(B) / T_GPU(B) (Eq. 25)
    speedup = round(cpu_mean / max(gpu_mean, 1e-4), 2)

    # Percentage reduction in execution time: Delta T (%) = (T_CPU - T_GPU) / T_CPU * 100 (Eq. 26)
    delta_t_pct = round(((cpu_mean - gpu_mean) / max(cpu_mean, 1e-4)) * 100.0, 2)

    mean_h2d = float(np.mean([d["t_h2d_ms"] for d in gpu_decomps]))
    mean_kernel = float(np.mean([d["t_kernel_ms"] for d in gpu_decomps]))
    mean_d2h = float(np.mean([d["t_d2h_ms"] for d in gpu_decomps]))

    return {
        "batch_size": batch_size,
        "cpu": {
            "mean_latency_ms": round(cpu_mean, 3),
            "std_latency_ms": round(float(np.std(cpu_arr)), 3),
            "p50_latency_ms": round(float(np.percentile(cpu_arr, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(cpu_arr, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(cpu_arr, 99)), 3),
            "throughput_theta_qps": cpu_throughput
        },
        "gpu": {
            "mean_latency_ms": round(gpu_mean, 3),
            "std_latency_ms": round(float(np.std(gpu_arr)), 3),
            "p50_latency_ms": round(float(np.percentile(gpu_arr, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(gpu_arr, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(gpu_arr, 99)), 3),
            "throughput_theta_qps": gpu_throughput,
            "mean_t_h2d_ms": round(mean_h2d, 3),
            "mean_t_kernel_ms": round(mean_kernel, 3),
            "mean_t_d2h_ms": round(mean_d2h, 3)
        },
        "speedup_s_b": speedup,
        "execution_time_reduction_pct": delta_t_pct
    }


def run_full_benchmark(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    batch_sizes: List[int] = DEFAULT_BATCH_SIZES
) -> int:
    """
    Executes full multi-batch scaling benchmark across paper batch sizes [100, 500, 1000, 5000, 10000].
    """
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Train baseline first.")
        return 1

    print(f"\nLoading baseline model from {model_path}...")
    baseline_model = BaselineJailbreakClassifier.load(model_path)
    gpu_model = GPUJailbreakClassifier(baseline_model)

    backend_info = gpu_model.get_execution_backend()
    print(f"Active GPU/Backend: {backend_info}")

    print(f"Loading evaluation prompts from {data_path}...")
    df = pd.read_parquet(data_path)
    prompts_pool = df["prompt"].tolist()

    telemetry = get_system_telemetry()

    print("\n" + "=" * 90)
    print(" HARDWARE BENCHMARK: CPU vs GPU INFERENCE SCALING (PAPER EQUATIONS 24-26)")
    print("=" * 90)
    print(f"{'Batch':<8} {'CPU Lat (p50)':<15} {'GPU Lat (p50)':<15} {'CPU Throughput':<16} {'GPU Throughput':<16} {'Speedup S(B)':<13} {'Delta T (%)':<10}")
    print("-" * 90)

    benchmark_records = []
    for b in batch_sizes:
        res = benchmark_single_batch_size(
            baseline_model=baseline_model,
            gpu_model=gpu_model,
            prompts_pool=prompts_pool,
            batch_size=b
        )
        benchmark_records.append(res)
        print(f"{b:<8} {res['cpu']['p50_latency_ms']:>8.3f} ms     {res['gpu']['p50_latency_ms']:>8.3f} ms     {res['cpu']['throughput_theta_qps']:>10,.0f} qps  {res['gpu']['throughput_theta_qps']:>10,.0f} qps  {res['speedup_s_b']:>7.2f}x      {res['execution_time_reduction_pct']:>7.1f}%")

    print("=" * 90 + "\n")

    output_payload = {
        "telemetry": telemetry,
        "backend": backend_info,
        "batch_benchmarks": benchmark_records
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Benchmark results saved to: {output_path}")
    return 0



def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPU vs GPU inference benchmark.")
    parser.add_argument("--model-path", type=str, default="saved_models/baseline_tfidf_lr.joblib")
    parser.add_argument("--data-path", type=str, default="data/splits/test.parquet")
    parser.add_argument("--output-json", type=str, default="results/metrics/benchmark_results.json")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)

    args = parser.parse_args()
    return run_full_benchmark(
        model_path=REPO_ROOT / args.model_path,
        data_path=REPO_ROOT / args.data_path,
        output_path=REPO_ROOT / args.output_json,
        batch_sizes=args.batch_sizes
    )


if __name__ == "__main__":
    sys.exit(main())
