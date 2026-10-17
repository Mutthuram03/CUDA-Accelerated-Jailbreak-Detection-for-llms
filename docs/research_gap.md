# Research Gap & Motivation

## 1. Context: LLM Safety Guardrails

Large Language Models (LLMs) deployed in user-facing applications require robust guardrails to intercept prompts that solicit harmful, illegal, or unethical outputs. Two primary categories of jailbreaks bypass naive safety filters:
1. **Instruction-Conflict Attacks**: Exploiting the model's instruction-following hierarchy by injecting instructions that command the model to disregard preceding system instructions, assume a malicious persona (e.g., "DAN", "AIM"), or simulate a fiction where rules do not apply.
2. **Obfuscation Attacks**: Encoding malicious requests in ciphers (Base64, ROT13, Morse, Caesar), multi-language mixtures, or character-level transformations (leetspeak, zero-width space interleaving, homoglyph replacement) that evade standard keyword filters.

---

## 2. Shortcomings of Existing Defenses

Current guardrail solutions face critical trade-offs between computational overhead, latency, and detection coverage:

| Guardrail Approach | Latency (Typical) | Compute Cost | Vulnerability / Limitation |
| :--- | :--- | :--- | :--- |
| **LLM-as-a-Judge (e.g., Llama Guard, GPT-4)** | 200ms – 2500ms | Extremely High (Billion-param model inference) | Adds severe latency to chat interfaces; economically prohibitive for large batch streams. |
| **Keyword / Heuristic Regex Matching** | < 1ms | Negligible | Brittle; easily bypassed by obfuscation, ciphers, and novel paraphrasing. |
| **Random Perturbation (e.g., SmoothLLM)** | 1000ms – 5000ms | Multi-query LLM calls | High latency; damages benign prompt semantic coherence. |
| **Standard CPU ML (TF-IDF / Linear / FastText)** | 5ms – 50ms | Low | CPU memory-bound; struggles with high throughput batch ingestion ($>10^4$ prompts/sec); lacks specialized hardware parallelization for obfuscation entropy analysis. |

---

## 3. The Research Gap

Existing literature either:
1. Employs large, slow neural models that cannot serve as low-latency front-line gatekeepers, or
2. Relies on lightweight CPU models that lack fine-grained detection of obfuscation patterns and cannot saturate modern GPU clusters deployed in inference fleets.

### The CUDA Acceleration Opportunity
In enterprise LLM serving architectures (e.g., Triton Inference Server, vLLM, TensorRT-LLM), GPUs are already active for LLM generation. By executing front-line jailbreak screening directly on the GPU with optimized CUDA operations:
- **Zero Host-Device Memory Stalls**: Prompts can be tokenized, sanitized, and classified directly on device memory before entering the LLM forward pass.
- **Microsecond Latency**: Parallelizing feature lookup, character entropy estimation, and classification kernels yields sub-5ms latency.
- **High Throughput**: Enables continuous screening of thousands of prompts per second with minimal VRAM footprint (~100–300 MB).

---

## 4. Research Questions & Hypotheses

- **RQ1 (Detection Efficacy)**: Can an optimized classifier trained on instruction-conflict and obfuscation features achieve competitive detection ($F_1 > 0.85$, $FPR < 0.02$) compared to computationally heavy guardrails?
- **RQ2 (Computational Speedup)**: What is the exact speedup ratio (latency and throughput) of GPU-accelerated / custom CUDA implementations compared to a tuned CPU baseline across batch sizes from $B=1$ to $B=128$?
- **RQ3 (Bottleneck Attribution)**: Where does the latency lie in CPU-based detection pipelines (string preprocessing vs feature extraction vs classification), and which specific operators benefit genuinely from CUDA acceleration?
