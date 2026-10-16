# CUDA-Accelerated Detection of Instruction-Conflict and Obfuscation-Based Jailbreak Prompts in Large Language Models

A research framework and evaluation harness dedicated to high-throughput, low-latency detection of instruction-conflict and obfuscation-based adversarial jailbreak prompts targeting Large Language Models (LLMs).

---

## 1. Research Motivation & Core Gap

Safety guardrails in production LLM inference systems (e.g., vLLM, TensorRT-LLM, Triton) must intercept adversarial jailbreak prompts before generation occurs. Existing defensive mechanisms exhibit critical limitations:
- **LLM-as-a-Judge guards (e.g., Llama Guard, GPT-4)** introduce substantial latency (300ms – 2,500ms) and prohibitive compute overhead, creating an inference bottleneck.
- **Static keyword filters & basic heuristics** are easily evaded by character-level ciphers (Base64, ROT13, hex, leetspeak) and instruction-conflict reformulations ("Ignore previous instructions", persona adoption).
- **CPU-based lightweight machine learning detectors** suffer throughput degradation during high-concurrency ingestion and cannot exploit shared GPU memory buffers in inference servers.

### Research Objective
This project investigates whether dedicated hardware acceleration—spanning optimized PyTorch GPU tensor pipelines and custom CUDA kernels—can deliver:
1. **Sub-5ms End-to-End Latency**: Drastic speedups compared to neural guard models.
2. **High-Throughput Batch Ingestion**: Processing thousands of prompts per second with minimal VRAM overhead (<300 MB).
3. **Rigorous Detection Accuracy**: Achieving $F_1 > 0.85$ and a False Positive Rate (FPR) $< 0.02$ across diverse instruction-conflict and obfuscation attack vectors.

---

## 2. Hardware & Environment Audit

The following hardware and environment profile was verified on the host machine:

| Component | Verified Specification |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home Single Language (Build 26200) |
| **Host Python** | Python 3.11.9 (64-bit) |
| **GPU Hardware** | NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, Compute Capability 8.9) |
| **VRAM** | 8188 MiB GDDR6 |
| **NVIDIA Driver** | 592.82 (Driver supports CUDA up to version 13.1) |
| **PyTorch Status** | Currently installed host package is `2.12.1+cpu` (CPU build). CUDA PyTorch wheel setup instructions are provided below. |

---

## 3. Repository Architecture

```
CUDA-Accelerated-Jailbreak-Detection-for-llms/
├── .gitignore                   # Rigorous ignore rules protecting datasets, venvs, weights, & logs
├── .env.example                 # Environment configuration template
├── pyproject.toml               # Modern build configuration and tool metadata
├── requirements.txt             # Core CPU baseline and scientific dependencies
├── requirements-cuda.txt        # PyTorch with CUDA wheel installation flags
├── requirements-dev.txt         # Testing, linting, and formatting tools
├── README.md                    # Project documentation, setup, and research overview
├── configs/                     # Reproducible experiment configurations
│   ├── baseline_tfidf.yaml      # Baseline vectorizer and classifier settings
│   └── experiment_defaults.yaml # System and evaluation defaults
├── cuda/                        # GPU engineering and custom kernels
│   ├── README.md                # CUDA roadmap and kernel design rules
│   ├── kernels/                 # CUDA C++ sources (.cu, .cuh)
│   └── benchmarks/              # Standalone kernel microbenchmarks
├── data/                        # Dataset governance and split management
│   ├── README.md                # Data integrity, de-duplication, and leakage prevention protocol
│   ├── raw/                     # Pristine raw datasets (ignored by git)
│   ├── processed/               # Standardized, normalized schema datasets (ignored by git)
│   └── splits/                  # Stratified train/val/test splits (ignored by git)
├── docs/                        # Research documentation
│   ├── architecture.md          # End-to-end component architecture
│   ├── methodology.md           # Mathematical formulation, evaluation protocol, and timing standards
│   ├── research_gap.md          # Comprehensive literature review and research gap analysis
│   └── dataset_provenance.md    # Candidate datasets, licenses, and provenance checklist
├── experiments/                 # Reproducible benchmark harnesses
│   ├── README.md                # Benchmarking methodology and statistical standards
│   ├── baselines/               # CPU reference execution scripts
│   └── profiling/               # Profiler harnesses (cProfile, torch.profiler, Nsight)
├── notebooks/                   # Exploratory data analysis notebooks
│   └── README.md                # Notebook guidelines (zero data leakage)
├── results/                     # Measured metrics and figures
│   ├── README.md                # Metric storage standards
│   ├── metrics/                 # JSON/CSV evaluation outputs (ignored by git)
│   └── figures/                 # ROC curves, confusion matrices, latency plots (ignored by git)
├── scripts/                     # Operational verification scripts
│   ├── verify_environment.py   # Structure and dependency validator
│   └── verify_gpu.py            # Hardware, driver, PyTorch CUDA, and tensor computation tester
├── src/                         # Core Python package
│   ├── preprocessing/           # Unicode normalization and obfuscation probing
│   ├── features/                # TF-IDF and structural entropy extraction
│   ├── models/                  # Baseline and neural classifiers
│   ├── evaluation/              # Accuracy, Precision, Recall, F1, FPR, Confusion Matrix
│   ├── inference/               # End-to-end inference pipeline with microsecond timing
│   └── utils/                   # Structured logging and system telemetry
├── tests/                       # Automated unit tests
│   ├── test_environment.py     # Environment sanity tests
│   ├── test_preprocessing.py   # Preprocessing and obfuscation tests
│   ├── test_models.py          # Baseline classifier tests
│   └── test_metrics.py         # Metrics and FPR accuracy tests
└── app/                         # Future API & Dashboard (Staged for Phase 4)
    └── README.md                # Deployment and UI plan
```

---

## 4. Setup & Verification Instructions (Windows PowerShell)

Follow these steps to set up and verify the research environment:

### Step 1: Clone & Navigate
```powershell
cd c:\Users\srmut\Desktop\CUDA_JAILBREAK\CUDA-Accelerated-Jailbreak-Detection-for-llms
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Core Dependencies
```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Step 4: Install PyTorch with CUDA Support
Because the host has an NVIDIA GeForce RTX 4060 with driver 592.82, install the CUDA-enabled PyTorch build:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Verify Environment & Hardware
Run the automated verification scripts:
```powershell
# 1. Verify directory structure and core packages
python scripts/verify_environment.py

# 2. Verify GPU detection, driver status, PyTorch CUDA, and execute test matrix multiplication
python scripts/verify_gpu.py
```

### Step 6: Run Automated Unit Tests
```powershell
pytest tests/ -v
```

---

## 5. Research Roadmap & Working Procedure

- **Milestone 1: Repository Initialization & Environment Audit**
  - Modular project architecture created.
  - Windows environment and RTX 4060 GPU audited.
  - Verification scripts and unit test suite implemented.
  - Strict data protection and `.gitignore` policies active.
- **Milestone 2: Data Provenance & CPU Baseline**
  - Audit and ingest candidate datasets (JailbreakBench, AdvGLUE, WildJailbreak).
  - Verify class balance, eliminate duplicates, and prevent train/test leakage.
  - Train and evaluate transparent CPU baseline (TF-IDF + Logistic Regression).
  - Measure baseline metrics: Accuracy, Precision, Recall, F1, FPR, and CPU latency.
- **Milestone 3: Profiling & CUDA Acceleration**
  - Profile CPU execution bottlenecks (tokenization vs vectorization vs inference).
  - Implement PyTorch GPU tensor pipeline and custom CUDA operators (`.cu`).
  - Measure latency ($p50, p95, p99$), throughput (prompts/sec), and batch scaling ($B=1 \dots 128$).
- **Milestone 4: Real-Time API & Security Dashboard**
  - Deploy high-throughput FastAPI inference engine.
  - Build interactive security dashboard for real-time prompt auditing.
