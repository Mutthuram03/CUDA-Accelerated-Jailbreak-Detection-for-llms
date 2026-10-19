"""
Dataset audit, validation, and attack taxonomy characterization module.
"""

import hashlib
import re
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from pydantic import BaseModel
from ..preprocessing.text_cleaner import ZERO_WIDTH_CHARS, ObfuscationDetector


INSTRUCTION_CONFLICT_PATTERNS = [
    r"\bignore\b.*\b(previous|prior|above|system|rules|instructions)\b",
    r"\bdisregard\b.*\b(rules|guidelines|instructions|context|data)\b",
    r"\bforget\b.*\b(earlier|preceding|directive|instructions|all)\b",
    r"\bbypass\b.*\b(above|prior|conversation|instructions)\b",
    r"\bdo not follow\b",
    r"\bneglect\b.*\b(prior|communication)\b",
    r"\bomit\b.*\b(preceding|directives|commands)\b",
    r"\bskip\b.*\b(initial|above|instructions)\b",
    r"\bpay no attention\b"
]

PERSONA_PATTERNS = [
    r"\byou are now\b",
    r"\bdeveloper mode\b",
    r"\bDAN\b",
    r"\bpretend to be\b",
    r"\bact as\b",
    r"\bhypothetical (story|response)\b",
    r"\bdo anything now\b",
    r"\bunrestricted\b",
    r"\bno safety\b",
    r"\bAIM\b",
    r"\bBH 2\.0\b"
]


class DatasetAuditReport(BaseModel):
    """Container for comprehensive dataset audit findings."""
    total_records: int
    null_prompts: int
    empty_prompts: int
    invalid_labels: int
    exact_duplicates: int
    unique_prompts: int
    benign_count: int
    malicious_count: int
    benign_ratio: float
    malicious_ratio: float
    benign_len_mean: float
    benign_len_median: float
    benign_len_min: int
    benign_len_max: int
    malicious_len_mean: float
    malicious_len_median: float
    malicious_len_min: int
    malicious_len_max: int
    instruction_conflict_count: int
    persona_adoption_count: int
    obfuscation_base64_count: int
    obfuscation_hex_count: int
    zero_width_count: int
    sha256_checksum: str

    def to_markdown(self) -> str:
        """Renders audit findings as formatted GitHub markdown."""
        return f"""# Dataset Quality & Security Audit Report

## 1. Overview & Integrity
- **Total Records Analyzed**: {self.total_records:,}
- **SHA-256 Checksum**: `{self.sha256_checksum}`
- **Null / Missing Prompts**: {self.null_prompts}
- **Empty Prompts**: {self.empty_prompts}
- **Invalid Labels**: {self.invalid_labels}
- **Exact Duplicate Prompts**: {self.exact_duplicates} (Unique: {self.unique_prompts:,})

## 2. Class Distribution
| Class | Count | Percentage |
| :--- | :--- | :--- |
| **Benign (0)** | {self.benign_count:,} | {self.benign_ratio * 100:.2f}% |
| **Malicious / Adversarial (1)** | {self.malicious_count:,} | {self.malicious_ratio * 100:.2f}% |
| **Total** | {self.total_records:,} | 100.00% |

## 3. Prompt Length Distributions (Character Counts)
| Metric | Benign Prompts (0) | Malicious Prompts (1) |
| :--- | :--- | :--- |
| **Mean Length** | {self.benign_len_mean:.1f} chars | {self.malicious_len_mean:.1f} chars |
| **Median (p50)** | {self.benign_len_median:.1f} chars | {self.malicious_len_median:.1f} chars |
| **Min Length** | {self.benign_len_min} chars | {self.malicious_len_min} chars |
| **Max Length** | {self.benign_len_max} chars | {self.malicious_len_max} chars |

## 4. Adversarial Attack Taxonomy Breakdown (Malicious Partition)
| Attack Signature Category | Detected Hits | % of Malicious Partition |
| :--- | :--- | :--- |
| **Instruction-Conflict Overrides** (e.g., `ignore previous`, `bypass above`) | {self.instruction_conflict_count:,} | {self.instruction_conflict_count / self.malicious_count * 100:.2f}% |
| **Persona Adoption & Jailbreak Framing** (e.g., `DAN`, `Developer Mode`) | {self.persona_adoption_count:,} | {self.persona_adoption_count / self.malicious_count * 100:.2f}% |
| **Base64 Payload Obfuscation** | {self.obfuscation_base64_count:,} | {self.obfuscation_base64_count / self.malicious_count * 100:.2f}% |
| **Contiguous Hex Encoding** | {self.obfuscation_hex_count:,} | {self.obfuscation_hex_count / self.malicious_count * 100:.2f}% |
| **Zero-Width Character Infiltration** | {self.zero_width_count:,} | {self.zero_width_count / self.malicious_count * 100:.2f}% |
"""


def audit_dataset(df: pd.DataFrame, sha256_checksum: str = "N/A") -> DatasetAuditReport:
    """
    Performs comprehensive verification, quality, and security audit of the dataset.
    """
    total = len(df)
    null_prompts = int(df["prompt"].isnull().sum())
    
    # Clean string representation for non-nulls
    prompts = df["prompt"].fillna("").astype(str)
    empty_prompts = int((prompts.str.strip() == "").sum())
    
    invalid_labels = int((~df["label"].isin([0, 1])).sum())
    
    # Exact duplicate check via prompt string hashing
    unique_prompts = int(prompts.nunique())
    exact_duplicates = total - unique_prompts
    
    # Class balance
    benign_df = df[df["label"] == 0]
    malicious_df = df[df["label"] == 1]
    
    benign_count = len(benign_df)
    malicious_count = len(malicious_df)
    
    benign_ratio = float(benign_count / total) if total > 0 else 0.0
    malicious_ratio = float(malicious_count / total) if total > 0 else 0.0
    
    # Length statistics
    benign_lens = benign_df["prompt"].str.len()
    malicious_lens = malicious_df["prompt"].str.len()
    
    b_mean = float(benign_lens.mean()) if not benign_lens.empty else 0.0
    b_med = float(benign_lens.median()) if not benign_lens.empty else 0.0
    b_min = int(benign_lens.min()) if not benign_lens.empty else 0
    b_max = int(benign_lens.max()) if not benign_lens.empty else 0
    
    m_mean = float(malicious_lens.mean()) if not malicious_lens.empty else 0.0
    m_med = float(malicious_lens.median()) if not malicious_lens.empty else 0.0
    m_min = int(malicious_lens.min()) if not malicious_lens.empty else 0
    m_max = int(malicious_lens.max()) if not malicious_lens.empty else 0
    
    # Adversarial taxonomy analysis on malicious partition
    conflict_res = [re.compile(p, re.IGNORECASE) for p in INSTRUCTION_CONFLICT_PATTERNS]
    persona_res = [re.compile(p, re.IGNORECASE) for p in PERSONA_PATTERNS]
    
    mal_texts = malicious_df["prompt"].tolist()
    
    conflict_count = sum(1 for t in mal_texts if any(r.search(t) for r in conflict_res))
    persona_count = sum(1 for t in mal_texts if any(r.search(t) for r in persona_res))
    
    detector = ObfuscationDetector()
    b64_count = sum(1 for t in mal_texts if detector.detect_base64(t))
    hex_count = sum(1 for t in mal_texts if detector.detect_hex(t))
    zw_count = sum(1 for t in mal_texts if bool(ZERO_WIDTH_CHARS.search(t)))
    
    return DatasetAuditReport(
        total_records=total,
        null_prompts=null_prompts,
        empty_prompts=empty_prompts,
        invalid_labels=invalid_labels,
        exact_duplicates=exact_duplicates,
        unique_prompts=unique_prompts,
        benign_count=benign_count,
        malicious_count=malicious_count,
        benign_ratio=round(benign_ratio, 4),
        malicious_ratio=round(malicious_ratio, 4),
        benign_len_mean=round(b_mean, 2),
        benign_len_median=round(b_med, 2),
        benign_len_min=b_min,
        benign_len_max=b_max,
        malicious_len_mean=round(m_mean, 2),
        malicious_len_median=round(m_med, 2),
        malicious_len_min=m_min,
        malicious_len_max=m_max,
        instruction_conflict_count=conflict_count,
        persona_adoption_count=persona_count,
        obfuscation_base64_count=b64_count,
        obfuscation_hex_count=hex_count,
        zero_width_count=zw_count,
        sha256_checksum=sha256_checksum
    )
