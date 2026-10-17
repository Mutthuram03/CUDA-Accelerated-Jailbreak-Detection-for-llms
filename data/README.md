# Dataset Management and Governance

This directory manages the datasets used for training and evaluating jailbreak detection models.

## Strict Data Governance Rules

In accordance with rigorous research standards:
1. **Never commit raw or processed datasets to Git**: Datasets are excluded in `.gitignore` to prevent repository bloat and potential policy violations.
2. **Provenance & Licensing**: Every external dataset must be logged in `docs/dataset_provenance.md` with source URL, authors, paper citation, license type, and date retrieved.
3. **Label Standardization**:
   - `0`: **Benign** (standard benign queries, helpful instructions, creative writing, edge-case safety prompts that do not violate policies).
   - `1`: **Jailbreak / Adversarial** (prompts engineered to bypass safety boundaries via instruction conflict, persona hijacking, obfuscation/ciphers, prefix injection, or adversarial suffixes).
4. **Data Leakage Prevention**:
   - Train, validation, and test splits must be partitioned strictly by unique prompt families or templates to avoid syntactic overlap between splits.
   - De-duplication and n-gram overlap checks are performed before model training.
5. **Class Balance**:
   - Both jailbreak and benign distribution statistics must be verified prior to experiments.
   - Metrics must report False Positive Rate (FPR) alongside Precision/Recall/F1, because in real-world LLM guardrails, a high false positive rate degrades user experience on benign prompts.

## Directory Structure

```
data/
├── README.md               # Dataset documentation and protocol
├── raw/                    # Pristine, immutable raw datasets (ignored by git)
│   └── .gitkeep
├── processed/              # Normalized, de-duplicated datasets with standard schema
│   └── .gitkeep
└── splits/                 # Stratified train/val/test splits (seed-locked, reproducible)
    └── .gitkeep
```

## Standard Dataset Schema (processed)

Normalized datasets stored in `data/processed/` must conform to the following schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | string / int | Unique prompt identifier |
| `prompt` | string | Raw text prompt |
| `label` | int | Binary target (`0` = benign, `1` = jailbreak) |
| `category` | string | Attack taxonomy (e.g., `instruction_conflict`, `cipher_obfuscation`, `persona_adoption`, `benign`) |
| `source_dataset` | string | Provenance identifier (e.g., `jailbreakbench`, `advglue`, `wildjailbreak`) |
| `split` | string | Assigned split (`train`, `val`, `test`) |
