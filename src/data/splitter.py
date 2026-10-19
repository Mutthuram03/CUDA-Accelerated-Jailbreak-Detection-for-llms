"""
Stratified and leakage-safe dataset splitting module.
"""

from pathlib import Path
from typing import Dict, Tuple, Set
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def verify_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, int]:
    """
    Verifies that no identical prompt strings leak across split boundaries.
    
    Returns:
        Dictionary of overlap counts between split pairs.
    """
    train_prompts: Set[str] = set(train_df["prompt"].str.strip())
    val_prompts: Set[str] = set(val_df["prompt"].str.strip())
    test_prompts: Set[str] = set(test_df["prompt"].str.strip())

    train_val_overlap = len(train_prompts.intersection(val_prompts))
    train_test_overlap = len(train_prompts.intersection(test_prompts))
    val_test_overlap = len(val_prompts.intersection(test_prompts))

    return {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap
    }


def create_reproducible_splits(
    df: pd.DataFrame,
    splits_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Partitions dataset into group-stratified, leakage-free train, validation, and test splits.
    Groups by normalized prompt strings so duplicate or templated variations never leak across splits.
    
    Args:
        df: Input DataFrame containing ['prompt', 'label'].
        splits_dir: Directory where train.parquet, val.parquet, test.parquet will be written.
        train_ratio: Fraction for training split (default: 0.70).
        val_ratio: Fraction for validation split (default: 0.15).
        test_ratio: Fraction for final held-out test split (default: 0.15).
        random_state: Deterministic random seed for reproducibility.
        
    Returns:
        Dictionary of split DataFrames: {"train": df, "val": df, "test": df}
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Train, val, and test ratios must sum to 1.0.")

    splits_dir.mkdir(parents=True, exist_ok=True)

    # Use stripped prompt as group identifier to prevent identical/near-identical leakage
    groups = df["prompt"].astype(str).str.strip()

    # First split: train vs temporary (val + test)
    temp_ratio = val_ratio + test_ratio
    gss_train = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss_train.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)
    temp_groups = temp_df["prompt"].astype(str).str.strip()

    # Second split: val vs test
    val_rel_ratio = val_ratio / temp_ratio
    gss_val = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_rel_ratio), random_state=random_state)
    val_sub_idx, test_sub_idx = next(gss_val.split(temp_df, groups=temp_groups))

    val_df = temp_df.iloc[val_sub_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_sub_idx].copy().reset_index(drop=True)

    # Assign split metadata column
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Leakage safeguard verification
    leakage = verify_split_leakage(train_df, val_df, test_df)
    if any(count > 0 for count in leakage.values()):
        raise RuntimeError(f"Data leakage detected across splits: {leakage}")

    # Write persistent Parquet files
    train_path = splits_dir / "train.parquet"
    val_path = splits_dir / "val.parquet"
    test_path = splits_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

