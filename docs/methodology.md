# Research Methodology & Experimental Protocol

## 1. Problem Formulation

As Large Language Models (LLMs) are deployed in critical workflows, adversaries exploit vulnerabilities in the instruction-following mechanism. In this project, we specifically target two prevalent and evasive attack classes:
1. **Instruction-Conflict Jailbreaks**: Prompts designed to pit system-level safety instructions against adversarial user constraints (e.g., "Ignore previous instructions", "From now on, you are in Developer Mode", prefix injections, and conflicting role definitions).
2. **Obfuscation-Based Jailbreaks**: Prompts that disguise malicious intent through transformation techniques (e.g., Base64 encoding, ROT13/Caesar ciphers, character-level leetspeak, Morse code, multi-lingual translation interleaving, and Unicode homoglyphs).

### Classification Objective
Given an input prompt $x \in \mathcal{X}$, predict binary label $y \in \{0, 1\}$:
$$y = \begin{cases} 0 & \text{if } x \text{ is benign / policy-compliant} \\ 1 & \text{if } x \text{ contains an instruction-conflict or obfuscation jailbreak} \end{cases}$$
along with a calibrated probability $P(y=1 \mid x)$ and categorical taxonomy attribution.

---

## 2. Phased Research Execution

### Phase 1: Environment & Framework Setup
- Establish clean directory hierarchy, verification harnesses, and environment isolation.
- Verify hardware acceleration capabilities (PyTorch CUDA runtime, GPU memory limits, compute capability).

### Phase 2: Transparent CPU Baseline & Data Integrity
- Curate and audit public jailbreak datasets (JailbreakBench, AdvGLUE, BeaverTails/HarmfulQA, WildJailbreak).
- Strict provenance logging and de-duplication across prompt templates to prevent train-test data leakage.
- Implement a reproducible CPU baseline:
  - Text preprocessing and normalization.
  - Sublinear TF-IDF vectorizer (unigrams + bigrams, $V \approx 20{,}000$).
  - L2-regularized Logistic Regression / Linear SVM.
- Evaluate standard classification metrics and latency.

### Phase 3: Bottleneck Profiling & GPU Acceleration
- Profile the CPU baseline:
  - String normalization & regex tokenization time vs. vectorization vs. linear projection.
- Distinguish GPU operational layers:
  1. **Standard PyTorch GPU Tensor Operations**: cuBLAS batch matrix-vector multiplications, embedding lookups, GPU tensor transformations.
  2. **Custom CUDA Kernels (`.cu`)**: Parallel regex / subword pattern scanning, parallel cipher / character entropy calculations, or fused sparse feature extraction.
- Measure latency ($p50, p95, p99$) and throughput (QPS) across batch sizes $B \in \{1, 8, 16, 32, 64, 128\}$.

### Phase 4: System Integration & Application
- Encapsulate detection engine into an asynchronous FastAPI service with GPU stream batching.
- Modern dashboard UI for live prompt auditing and latency benchmarking.

---

## 3. Evaluation Metrics

Because safety guardrails operate in production where >99% of user traffic is benign, **False Positive Rate (FPR)** is as critical as Recall:
- **Accuracy**: Overall fraction of correct classifications.
- **Precision**: $\frac{TP}{TP + FP}$ — avoiding false alarms on benign queries.
- **Recall (Detection Rate)**: $\frac{TP}{TP + FN}$ — catching true jailbreaks.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **FPR at 99% Recall / Fixed FPR Thresholds**: False Positive Rate when operating at strict safety thresholds.
- **Confusion Matrix**: Full breakdown of TP, FP, TN, FN across attack categories.

---

## 4. Hardware Profiling & Timing Standards

To prevent false claims of speedup:
- **Warmup**: Discard the first 10-20 iterations to account for driver initialization, CUDA context creation, and kernel JIT compilation.
- **Synchronization**: Wrap all timed GPU sections with:
  ```python
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  # ... GPU operations ...
  end_event.record()
  torch.cuda.synchronize()
  elapsed_ms = start_event.elapsed_time(end_event)
  ```
- **Repeatability**: Run at least 100 trials and report median, standard deviation, and percentiles.
- **System Telemetry**: Track host RAM (`psutil`), GPU VRAM allocated vs reserved (`torch.cuda.memory_allocated`, `torch.cuda.memory_reserved`), and compute utilization.
