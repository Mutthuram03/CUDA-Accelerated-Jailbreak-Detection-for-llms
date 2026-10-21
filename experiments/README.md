# Reproducible Experiments Protocol

This directory contains executable experiment runners and benchmark scripts for jailbreak detection.

## Benchmark Methodology

To ensure genuine, reproducible scientific findings:
1. **CPU Baseline Protocol**:
   - Model: Scikit-Learn TF-IDF vectorization (unigrams + bigrams) + Logistic Regression / Linear SVM.
   - Evaluated on identical test splits (`data/splits/test.parquet` or `.csv`).
   - Latency evaluated with single-thread and multi-thread configurations.
2. **GPU & CUDA Pipeline Protocol**:
   - Compare PyTorch tensor-accelerated models and custom CUDA operators against the CPU baseline.
   - Controlled metrics:
     - End-to-end latency: preprocessing + transfer to device + forward pass + thresholding.
     - Device compute latency: isolated kernel/forward pass execution time on GPU.
     - Throughput: processed queries per second ($QPS = N / \Delta t$).
     - Memory footprint: Peak Host RAM (MB) via `psutil` and Peak Device VRAM (MB) via `torch.cuda.max_memory_allocated()`.
3. **Statistical Validity**:
   - Run a minimum of 10 warmup iterations.
   - Run at least 100 timed trials per batch size.
   - Record and output median, standard deviation, and percentiles (p50, p90, p95, p99).

## Directory Structure

```
experiments/
├── README.md               # Benchmark standards and procedures
├── baselines/              # Scripts to execute and evaluate CPU/reference models
│   └── .gitkeep
└── profiling/              # Profiling harnesses (cProfile, torch.profiler, Nsight)
    └── .gitkeep
```
