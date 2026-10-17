# Dataset Provenance, Licensing & Audit Protocol

## Policy on External Data

To uphold scientific integrity and legal compliance:
1. **No Unverified Downloads**: Datasets are never downloaded without prior inspection of license terms and provenance.
2. **Standardized Attribution**: Every dataset imported into `data/raw/` must have an entry in this document detailing its origin, publication, and license.
3. **Data Protection**: Prompt texts and annotations are stored locally in `data/` and excluded from version control via `.gitignore`.

---

## Candidate Benchmark Datasets for Jailbreak Detection

| Dataset | Primary Focus | Source / Reference | Typical License | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **JailbreakBench** | Standardized safety benchmark; instruction injection & conflict prompts | Chao et al., 2024 (NeurIPS) | MIT / Apache 2.0 | Pending user review & download approval |
| **WildJailbreak / In-The-Wild** | Community-sourced jailbreak prompts (DAN, persona adoption, obfuscation) | Jiang et al., 2024 / Deng et al., 2023 | Research / Academic Use | Pending user review & download approval |
| **AdvGLUE** | Adversarial GLUE benchmark; token perturbation & character obfuscation | Wang et al., 2022 | CC BY 4.0 | Pending user review & download approval |
| **BeaverTails / HarmfulQA** | Harmful question answering & safety boundaries (benign vs malicious pairs) | Ji et al., 2023 (NeurIPS) | CC BY-NC 4.0 | Pending user review & download approval |
| **Anthropic Red-Teaming** | Open-source multi-turn and single-turn red-teaming transcripts | Ganguli et al., 2022 | MIT | Pending user review & download approval |

---

## Data Quality & Leakage Audit Checklist

Prior to training any baseline or model:
- [ ] **Provenance Verified**: Authors, URL, and license documented.
- [ ] **Deduplication**: Exact string hash matching to eliminate duplicates.
- [ ] **Template Leakage Check**: Ensure jailbreak prompt templates (e.g., standard "Do Anything Now" variations) do not appear simultaneously in both training and test sets.
- [ ] **Class Balance Inspection**: Record benign-to-adversarial ratio ($N_{\text{benign}} / N_{\text{jailbreak}}$).
- [ ] **Length Distribution**: Verify prompt length distributions between benign and adversarial samples to prevent length-based spurious correlation.
