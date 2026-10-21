#!/usr/bin/env python3
"""
Reproducible data acquisition, audit, schema validation, and splitting script for MPDD.
"""

import sys
import json
from pathlib import Path
import yaml

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_or_fetch_mpdd
from src.data.validator import audit_dataset
from src.data.splitter import create_reproducible_splits


def main() -> int:
    print("\n" + "=" * 70)
    print(" MPDD Acquisition, Schema Validation, Audit & Splitting Pipeline")
    print("=" * 70)

    config_path = REPO_ROOT / "configs" / "experiment_defaults.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = REPO_ROOT / config["dataset"]["raw_dir"]
    processed_dir = REPO_ROOT / config["dataset"]["processed_dir"]
    splits_dir = REPO_ROOT / config["dataset"]["splits_dir"]
    metrics_dir = REPO_ROOT / "results" / "metrics"
    docs_dir = REPO_ROOT / "docs"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 1] Ingesting dataset into '{raw_dir}'...")
    df, sha256_hash = load_or_fetch_mpdd(raw_dir=raw_dir)
    print(f"  Raw CSV file present: {raw_dir / 'MPDD.csv'}")
    print(f"  SHA-256 Checksum: {sha256_hash}")
    print(f"  Total records loaded: {len(df):,}")

    print("\n[Step 2] Executing rigorous data quality and security audit...")
    audit_report = audit_dataset(df, sha256_checksum=sha256_hash)
    
    # Save machine-readable audit report
    json_path = metrics_dir / "data_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report.model_dump(), f, indent=2)
    print(f"  Audit metrics saved to: {json_path}")

    # Save human-readable markdown audit report
    md_path = docs_dir / "data_audit_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(audit_report.to_markdown())
    print(f"  Markdown audit report saved to: {md_path}")

    print("\n[Step 3] Saving normalized dataset...")
    norm_path = processed_dir / "mpdd_normalized.parquet"
    df.to_parquet(norm_path, index=False)
    print(f"  Normalized parquet saved to: {norm_path}")

    print("\n[Step 4] Creating stratified, leakage-free train/val/test splits...")
    seed = int(config["project"].get("seed", 42))
    splits = create_reproducible_splits(
        df=df,
        splits_dir=splits_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=seed
    )

    print(f"  Train split size: {len(splits['train']):,} (70%)")
    print(f"  Val split size  : {len(splits['val']):,} (15%)")
    print(f"  Test split size : {len(splits['test']):,} (15%)")
    print("  Splits saved to data/splits/ as train.parquet, val.parquet, test.parquet")

    print("\n" + "=" * 70)
    print(" Summary Audit Findings")
    print("=" * 70)
    print(f"  Class Balance      : Benign={audit_report.benign_count:,} ({audit_report.benign_ratio*100:.1f}%) | Malicious={audit_report.malicious_count:,} ({audit_report.malicious_ratio*100:.1f}%)")
    print(f"  Unique Prompts     : {audit_report.unique_prompts:,} / {audit_report.total_records:,} (Exact duplicates: {audit_report.exact_duplicates})")
    print(f"  Instruction Conflict: {audit_report.instruction_conflict_count:,} hits in malicious set ({audit_report.instruction_conflict_count/audit_report.malicious_count*100:.1f}%)")
    print(f"  Persona Adoption   : {audit_report.persona_adoption_count:,} hits in malicious set ({audit_report.persona_adoption_count/audit_report.malicious_count*100:.1f}%)")
    print(f"  Base64 Obfuscation : {audit_report.obfuscation_base64_count:,} hits")
    print(f"  Hex Obfuscation    : {audit_report.obfuscation_hex_count:,} hits")
    print("  Pipeline Audit Complete: READY FOR MODEL TRAINING")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
