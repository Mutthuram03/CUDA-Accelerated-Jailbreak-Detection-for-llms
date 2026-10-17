# System Architecture & Component Design

## Architectural Overview

The system is architected as a modular, high-throughput pipeline designed to evaluate and demonstrate genuine acceleration in detecting instruction-conflict and obfuscation-based jailbreaks.

```
                          ┌───────────────────────────┐
                          │   Raw Input Prompt(s)     │
                          │   (Single or Batched)     │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │    src/preprocessing      │
                          │  - Unicode Normalization  │
                          │  - Obfuscation Decoders   │
                          │  - Token / Char Splitting │
                          └─────────────┬─────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌───────────────────────┐               ┌───────────────────────┐
        │     CPU Baseline      │               │     GPU Pipeline      │
        │      src/features     │               │   PyTorch + cuda/     │
        │ - Scikit-Learn TF-IDF │               │ - Batched GPU Tensors │
        │ - Logistic Regression │               │ - Custom CUDA Kernels │
        │ - Linear SVM          │               │ - Fused Matrix Ops    │
        └───────────┬───────────┘               └───────────┬───────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │      src/evaluation       │
                          │ - Binary Classification   │
                          │ - Calibrated Probability  │
                          │ - Attack Taxonomy Label   │
                          │ - Latency & Memory Trace  │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │     results/ & app/       │
                          │ - Benchmark Tables/Plots  │
                          │ - FastAPI Guardrail Serv. │
                          └───────────────────────────┘
```

---

## Component Responsibilities

### 1. `src/preprocessing`
- **Text Normalization**: Strips zero-width characters, homoglyphs, and non-printable control codes.
- **Obfuscation Probing**: Fast heuristic checks for Base64 headers, ROT13 patterns, hex encodings, and leetspeak substitutions.

### 2. `src/features`
- **Sparse Feature Extraction**: N-gram bag-of-words and sublinear TF-IDF vectorizers.
- **Structural Features**: Character entropy, instruction markers (e.g. system delimiters, XML/markdown tags, markdown quote blocks).
- **Dense Embeddings (Staged)**: Lightweight transformer embeddings if required by advanced models.

### 3. `src/models`
- Encapsulates baseline estimators and GPU-accelerated linear / neural classifiers under a unified API:
  - `fit(X, y)`
  - `predict(X) -> np.ndarray`
  - `predict_proba(X) -> np.ndarray`

### 4. `cuda/`
- Direct hardware execution units:
  - CUDA C++ kernels compiled to shared libraries (`.so` / `.pyd` / `.dll`) via PyTorch C++ extensions (`torch.utils.cpp_extension`) or JIT.
  - Isolated microbenchmarks measuring raw compute bandwidth, memory transfer overhead ($H2D$ and $D2H$), and execution scaling.

### 5. `src/evaluation`
- Rigorous scoring: Accuracy, Precision, Recall, F1, False Positive Rate (FPR), and Area Under the ROC/PR Curves.
- Latency and throughput benchmarking utilities with microsecond-level resolution.

### 6. `app/`
- Staged deployment layer (Phase 4):
  - Async FastAPI endpoints.
  - Interactive dashboard for security researchers.
