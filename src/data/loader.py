"""
Data loader and downloader utility for MPDD (Malicious Prompt Detection Dataset).
"""

import hashlib
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

MPDD_RAW_URL = (
    "https://raw.githubusercontent.com/aminejebbar/mpdd-benchmark/main/processed_data/MPDD.csv"
)
EXPECTED_COLUMNS = ["Prompt", "isMalicious"]


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 cryptographic hash of a local file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_mpdd_raw(
    target_path: Path,
    url: str = MPDD_RAW_URL,
    timeout_sec: int = 60
) -> str:
    """
    Downloads pristine MPDD raw CSV to target directory.
    
    Args:
        target_path: Destination path for raw CSV file.
        url: Source URL for the CSV file.
        timeout_sec: Socket timeout in seconds.
        
    Returns:
        SHA-256 hash of the downloaded file.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) JailbreakDetectionResearch/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as response, open(target_path, "wb") as out_file:
        for chunk in iter(lambda: response.read(65536), b""):
            out_file.write(chunk)

    return compute_file_sha256(target_path)


def load_or_fetch_mpdd(
    raw_dir: Path,
    force_download: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Loads MPDD from data/raw/MPDD.csv, downloading it if not present.
    
    Args:
        raw_dir: Directory where raw dataset resides.
        force_download: If True, re-downloads even if local file exists.
        
    Returns:
        Tuple of (standardized DataFrame with ['prompt', 'label'], file SHA-256 checksum)
    """
    csv_path = raw_dir / "MPDD.csv"
    if not csv_path.exists() or force_download:
        sha256_hash = download_mpdd_raw(csv_path)
    else:
        sha256_hash = compute_file_sha256(csv_path)

    df = pd.read_csv(csv_path)
    
    # Validate schema
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"MPDD raw file missing required columns: {missing_cols}. Found: {df.columns.tolist()}"
        )

    # Standardize column names
    standardized_df = pd.DataFrame({
        "prompt": df["Prompt"].astype(str),
        "label": df["isMalicious"].astype(int)
    })

    return standardized_df, sha256_hash
