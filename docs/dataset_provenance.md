# Dataset Provenance, Licensing & Audit Protocol

## 1. Policy on External Data

To uphold scientific integrity and legal compliance:
1. **No Unverified Downloads**: Datasets are never downloaded without prior inspection of license terms, sample schemas, and provenance.
2. **Standardized Attribution**: Every dataset imported into `data/raw/` must have an entry in this document detailing its origin, publication, and license.
3. **Data Protection**: Prompt texts and annotations are stored locally in `data/` and excluded from version control via `.gitignore`.
4. **No Contamination / Leakage**: Datasets must be deduplicated across exact strings and template families prior to model splitting.

---

## 2. Investigation of Project Abstract Dataset: MPDD (39,234 Prompts)

### Verification Summary
- **Dataset Name**: Malicious Prompt Detection Dataset (MPDD)
- **Creator / Maintainer**: Mohammed Amine Jebbar (Independent AI Security Researcher)
- **Primary Source / Host**: [Kaggle: `mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd`](https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd)
- **Associated GitHub Repository**: [`aminejebbar/mpdd-benchmark`](https://github.com/aminejebbar/mpdd-benchmark)
- **Total Record Count**: Exactly 39,234 labeled prompts (50% benign, 50% malicious / adversarial)
- **License**: CC0: Public Domain (Open Access for research and commercial benchmarking)
- **Format & Size**: CSV / structured files (~134 MB raw)

### Provenance & Aggregation Origin
MPDD is an aggregated and cleaned composite dataset compiled from multiple prior open-source safety resources:
1. *Prompt Injection in the Wild* (Prediction Guard)
2. *Prompt Injection Malignant* datasets
3. *LLM 7 Prompt Training Dataset*
4. Community-collected jailbreak prompts and benign conversational corpora

### Label Schema
- **Target Column**: `label` (integer `0` or `1`, or categorical text)
  - `0`: Benign / Safe (legitimate user instructions, conversational text, general queries)
  - `1`: Malicious / Jailbreak / Prompt Injection (instruction-override attempts, persona-adoption attacks, adversarial injections)
- **Text Column**: `prompt` / `text`
- **Extended Metadata**: In subset files, attack category tags (e.g., direct injection, role-play override) are partially available, though not uniformly labeled across all 39,234 rows.

### Academic Citations & Usage
The dataset has been utilized in peer-reviewed and preprint security literature evaluating LLM guardrails:
- Benchmarked in studies exploring Bayesian-Optimized Word2Vec-BiLSTM classifiers and ensemble safety filters (ResearchGate / MDPI / arXiv preprints).
- Serves as a standard tabular benchmark for rapid binary prompt guardrail training.

### Limitations & Verified Unknowns
- **Aggregated Provenance**: Because MPDD is a composite of third-party public sets, template duplication across source corpora must be programmatically screened.
- **Access Method**: Requires Kaggle authentication (`kaggle.json` API token) for automated script download, or manual user export via web browser into `data/raw/`.
- **Granular Obfuscation Breakdown**: While it contains instruction-conflict attacks ("Ignore previous instructions", DAN variations), coverage of synthetic character ciphers (e.g., Base64 payloads, ROT13, zero-width spaces) is sporadic rather than systematic.

---

## 3. Comparison of Candidate Benchmark Datasets

| Candidate Dataset | Source & Maintainer | Total Samples | Attack Coverage (Instruction Conflict & Obfuscation) | Provenance & Licensing | Access Method | Strengths & Suitability | Key Limitations & Risks |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MPDD** *(Candidate A)* | Mohammed Amine Jebbar (Kaggle) | **39,234** | Broad instruction injection & jailbreaks; sporadic ciphers | CC0 / Public Domain (Aggregated) | Kaggle CLI / Web download | Exactly matches the abstract's 39,234-sample specification; perfectly balanced (50/50); ideal scale for fast CPU baseline and high-throughput GPU batch tests. | Composite origin; requires strict de-duplication; automated download requires Kaggle API token. |
| **WildJailbreak** *(Candidate B)* | Allen Institute for AI (AI2, Jiang et al., 2024) | **~262,000** | High coverage of real-world "in-the-wild" adversarial jailbreaks, synthetic variants, and paired benign queries | AI2 ImpACT License (Academic / Research) | Hugging Face (`allenai/wildjailbreak`) direct load | Gold-standard research provenance (AI2); contrastive benign pairs explicitly designed to measure false-positive / over-refusal rates. | Very large size (262K rows); slower downloads; requires sampling a standardized 40k subset for comparable benchmarking. |
| **JailbreakBench / JBB-Behaviors** *(Candidate C)* | Patrick Chao et al. (NeurIPS 2024 Datasets Track) | **100 behaviors** (+ generated attack artifacts) | Standardized taxonomy (10 risk domains); GCG, PAIR, TAP, AutoDAN attacks | MIT License | Hugging Face (`JailbreakBench/JBB-Behaviors`) / GitHub | Highest academic rigor and peer review; cleanly categorized behaviors; standardized benchmark standard. | Small core behavior set (100 malicious, 100 benign); full attack generation requires loading separate attack trace artifacts. |
| **In-The-Wild Jailbreak Prompts / Masterkey** *(Candidate D)* | Deng et al. (USENIX Security / IEEE S&P 2023) | **~1,400+** | Real-world persona and instruction conflict (DAN 1.0–11.0, Developer Mode, AIM) | Academic Open Access (`verazuo/jailbreak_llms`) | GitHub direct raw download | Genuine adversarial jailbreaks sourced directly from Reddit and Discord communities. | Small sample size (~1.4k prompts); insufficient by itself for high-throughput batching experiments (batch size 5,000–10,000). |
| **deepset/prompt-injections** *(Candidate E)* | Deepset AI (Haystack) | **~660** | Prompt injections (system overrides, context hijacking) | Apache 2.0 | Hugging Face (`deepset/prompt-injections`) | Highly curated, clean labels; excellent for unit and regression testing. | Too small for large-scale training or high-throughput GPU saturation benchmarking. |

---

## 4. Recommendation & Strategy

### Primary Dataset Recommendation: **MPDD (39,234 Prompts)**
**Rationale**:
1. **Exact Specification**: Matches the 39,234-sample target outlined in the project abstract.
2. **Computational Suitability**: 39,234 rows is ideal for rapid CPU baseline training (<10 seconds) and benchmarking batched inference scaling ($B = 100, 500, 1000, 5000, 10000$).
3. **Balance**: Balanced 50/50 split allows clean precision/recall and False Positive Rate (FPR) baseline evaluation without extreme class imbalance distortion.

### Secondary Evaluation Split: **JailbreakBench / WildJailbreak Contrastive Subset**
**Rationale**:
To ensure our detector does not overfit to MPDD-specific lexical artifacts (Phase 5 Detection Analysis), we recommend evaluating out-of-distribution generalization on a curated sample of **JailbreakBench** and **WildJailbreak contrastive benign prompts**. This directly tests whether the model catches real instruction-conflicts or merely memorizes keywords.

---

## 5. Data Quality, Preprocessing & Leakage Audit Protocol

Prior to fitting any baseline:
- [ ] **Provenance Recorded**: Official URL, author, and license confirmed in this document.
- [ ] **Schema Conformance**: Verified columns mapped to `prompt` (string) and `label` (int: 0/1).
- [ ] **Exact De-duplication**: SHA-256 hash matching on raw prompt text to remove exact duplicate entries.
- [ ] **Near-Duplicate / Template Clustering**: Group-based splitting to ensure templated prompts (e.g., repeated DAN prefixes) do not leak between training and test sets.
- [ ] **Class Balance Verification**: Confirm benign-to-adversarial count ($N_{\text{benign}}, N_{\text{jailbreak}}$).
- [ ] **Length Distribution Analysis**: Inspect character/token length histograms between classes to avoid spurious length correlations.
- [ ] **Preprocessing Scope**: All TF-IDF vectorizer vocabularies and scaling parameters fitted strictly on the training partition.

