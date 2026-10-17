# CUDA Acceleration & GPU Engineering Module

This module houses custom GPU kernels, PyTorch C++/CUDA extensions, and hardware profiling experiments for accelerated jailbreak detection.

## Engineering Principles

1. **Profile Before Optimizing**:
   - Accelerating components that account for <5% of total runtime violates Amdahl's Law.
   - We profile the end-to-end inference pipeline (string normalization, tokenization/n-gram hashing, feature lookup, matrix multiplication, softmax) using CPU and GPU profilers before writing custom kernels.
2. **Distinguish PyTorch GPU vs. Custom CUDA Kernels**:
   - **PyTorch GPU Operations**: Standard cuBLAS/cuDNN tensor routines executing on device memory.
   - **Custom CUDA Kernels (`.cu`)**: Specialized kernels targeting unique bottlenecks (e.g., parallel instruction-conflict regex scanning, fused character/subword entropy calculation, parallel Levenshtein / obfuscation decoding, or fused sparse-dense classification).
3. **Measurement Rigor**:
   - Every GPU timing measurement must utilize `torch.cuda.Event(enable_timing=True)` with explicit `torch.cuda.synchronize()` before and after the timed region.
   - Initial iterations must be discarded as GPU warmup (kernel compilation, driver launch overhead, memory caching).
   - Report p50, p95, and p99 latency across diverse batch sizes ($B \in \{1, 8, 16, 32, 64, 128\}$).

## Directory Structure

```
cuda/
├── README.md               # GPU architectural overview and engineering roadmap
├── kernels/                # Custom C++/CUDA kernel sources (.cu, .cuh, .cpp)
│   └── .gitkeep
└── benchmarks/             # Standalone CUDA kernel micro-benchmarks
    └── .gitkeep
```

## System Hardware Specification (Development Machine)
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, Compute Capability 8.9)
- **VRAM**: 8188 MiB GDDR6
- **Driver**: 592.82 (Supports CUDA up to 13.1)
- **Host OS**: Microsoft Windows 11
