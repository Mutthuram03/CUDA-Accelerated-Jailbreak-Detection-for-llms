# Research Methodology & Mathematical Formulation

**Paper Title**: *CUDA-Accelerated Detection of Instruction-Conflict and Obfuscation-Based Jailbreak Prompts in Large Language Models*  
**Authors**: Mutthuram S R, P.T. Rangesh Pandian, Aanish Nithin A, Kavitha R  
**Affiliation**: SRM Institute of Science and Technology, Kattankulathur, Chennai, India  

---

## 1. Dataset & Problem Formulation (Section III-A)

Given a dataset of prompts $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ (Eq. 1), where:
- $x_i$ is an input text prompt,
- $y_i \in \{0, 1\}$ is its binary security label ($y_i = 0$ for benign, $y_i = 1$ for malicious jailbreak attempts).

The class proportion is given by:
$$P(c) = \frac{N_c}{N} \quad (2)$$

The dataset is partitioned into pairwise-disjoint subsets:
$$\mathcal{D} = \mathcal{D}_{train} \cup \mathcal{D}_{val} \cup \mathcal{D}_{test} \quad (3)$$
Exact and near-duplicate prompts are investigated and handled prior to splitting to prevent data leakage across subsets.

---

## 2. Text Preprocessing (Section III-B)

Each prompt is represented as a character sequence:
$$x_i = (c_1, c_2, \dots, c_{L_i}) \quad (4)$$
Preprocessing applies a sanitization and normalization operator:
$$\tilde{x}_i = \mathcal{P}(x_i) \quad (5)$$
Crucially, unusual characters and formatting (such as Base64 strings, zero-width characters, and non-alphanumeric symbols) are preserved rather than indiscriminately stripped, because they carry key adversarial obfuscation signals.

---

## 3. Combined Feature Representation (Section III-C)

To capture multiple complementary adversarial indicators, features combine word-level, character-level, and structural signals:
$$\mathbf{z}_i = \left[ \mathbf{z}_i^{(w)} \parallel \mathbf{z}_i^{(c)} \parallel \mathbf{z}_i^{(s)} \right] \quad (6)$$

Where:
1. **Word-level component $\mathbf{z}_i^{(w)}$**: Token n-gram patterns weighted via sublinear TF-IDF:
   $$\text{tfidf}(t, i) = \text{tf}(t, i) \cdot \log\left(\frac{N + 1}{\text{df}(t) + 1}\right) \quad (7)$$
2. **Character-level component $\mathbf{z}_i^{(c)}$**: Character n-grams capturing obfuscation, subwords, and leetspeak artifacts.
3. **Structural component $\mathbf{z}_i^{(s)}$**: Prompt metadata including the character-diversity ratio:
   $$R_{div}(x_i) = \frac{|\{c : c \in x_i\}|}{L_i}, \quad L_i > 0 \quad (8)$$
   alongside prompt length $L_i$, non-alphanumeric ratio, Shannon entropy, and instruction-conflict pattern matches.

---

## 4. Machine-Learning Classifier (Section III-D)

The detector estimates malicious risk probability:
$$p_i = f_\theta(\mathbf{z}_i), \quad p_i \in [0, 1] \quad (9)$$

Using an L2-regularized Logistic Regression baseline:
$$p_i = \sigma(\mathbf{w}^T \mathbf{z}_i + b) \quad (10)$$
$$\sigma(a) = \frac{1}{1 + \exp(-a)} \quad (11)$$

Optimized via binary cross-entropy with L2 penalty:
$$\mathcal{L}(\theta) = -\frac{1}{N_{train}} \sum_{i=1}^{N_{train}} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right] \quad (12)$$
$$\mathcal{L}_{reg} = \mathcal{L}(\theta) + \lambda \|\mathbf{w}\|_2^2 \quad (13)$$
where regularization strength $\lambda$ (or $C = \frac{1}{2\lambda}$) is tuned on validation data.

---

## 5. Three-Way Risk Decision Policy & Calibration (Section III-E)

To support practical deployment where uncertain queries require human review or secondary LLM gatekeeping, a three-way decision policy is defined:
$$d(p_i) = \begin{cases} \text{allow}, & p_i < \tau_{low} \\ \text{flag}, & \tau_{low} \le p_i < \tau_{high} \\ \text{block}, & p_i \ge \tau_{high} \end{cases} \quad (14)$$
subject to:
$$0 \le \tau_{low} < \tau_{high} \le 1 \quad (15)$$

Confidence calibration is evaluated via Expected Calibration Error (ECE) across $M$ bins:
$$\text{ECE} = \sum_{m=1}^M \frac{|S_m|}{n} |\text{acc}(S_m) - \text{conf}(S_m)| \quad (16)$$

---

## 6. CUDA-Accelerated Execution & Latency Decomposition (Section III-F)

For a batch of $B$ prompts $\mathbf{Z} = \mathcal{F}(\tilde{x}_1, \dots, \tilde{x}_B)$ (Eq. 17), threads map to samples using the global 1D index:
$$i = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x} \quad (18)$$

End-to-end inference latency is decomposed into:
$$T_{total} = T_{pre} + T_{H2D} + T_{kernel} + T_{D2H} + T_{post} \quad (19)$$
where:
- $T_{pre}$: Text cleaning, tokenization, and feature extraction.
- $T_{H2D}$: Host-to-Device tensor transfer across PCIe bus.
- $T_{kernel}$: CUDA kernel execution time on GPU streaming multiprocessors.
- $T_{D2H}$: Device-to-Host transfer of predicted probabilities.
- $T_{post}$: Decision policy post-processing ($allow, flag, block$).

---

## 7. Detection Metrics (Section III-G)

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \quad (20)$$
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN} \quad (21)$$
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \quad (22)$$
$$\text{Positive Prediction Rate: } R_{positive} = \frac{TP + FP}{TP + TN + FP + FN} \quad (23)$$

Macro-averaged F1 and class-wise precision/recall/F1 are reported alongside False Positive Rate (FPR) at low false-alarm operating points.

---

## 8. Computational Performance Metrics (Section III-H)

For batch size $B$ processed in time $T$:
$$\text{Throughput: } \Theta = \frac{B}{T} \quad (24) \quad [\text{prompts/second}]$$
$$\text{GPU Speedup: } S(B) = \frac{T_{CPU}(B)}{T_{GPU}(B)} \quad (25)$$
$$\text{Percentage Reduction in Execution Time: } \Delta T(\%) = \frac{T_{CPU} - T_{GPU}}{T_{CPU}} \times 100 \quad (26)$$

Evaluated across batch sizes $B \in \{100, 500, 1000, 5000, 10000\}$ using warm-up runs, CUDA synchronization, and repeated trials reporting mean and standard deviation.

