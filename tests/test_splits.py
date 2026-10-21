"""
Unit tests for stratified splitting and cross-split leakage prevention.
Compatible with both unittest and pytest.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splitter import create_reproducible_splits, verify_split_leakage


class TestSplits(unittest.TestCase):
    """Verifies split proportions and leakage safeguards."""

    def test_split_generation_and_leakage_protection(self):
        prompts = [f"Benign prompt number {i}" for i in range(50)] + [
            f"Jailbreak attack prompt variation {i}" for i in range(50)
        ]
        labels = [0] * 50 + [1] * 50
        df = pd.DataFrame({"prompt": prompts, "label": labels})

        with tempfile.TemporaryDirectory() as tmp_dir:
            splits_dir = Path(tmp_dir)
            splits = create_reproducible_splits(
                df=df,
                splits_dir=splits_dir,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                random_state=42
            )

            train_df = splits["train"]
            val_df = splits["val"]
            test_df = splits["test"]

            # Check sizes
            self.assertEqual(len(train_df) + len(val_df) + len(test_df), 100)
            self.assertEqual(len(train_df), 70)
            self.assertEqual(len(val_df), 15)
            self.assertEqual(len(test_df), 15)

            # Check leakage
            leakage = verify_split_leakage(train_df, val_df, test_df)
            self.assertEqual(leakage["train_val_overlap"], 0)
            self.assertEqual(leakage["train_test_overlap"], 0)
            self.assertEqual(leakage["val_test_overlap"], 0)

            # Check files exist
            self.assertTrue((splits_dir / "train.parquet").exists())
            self.assertTrue((splits_dir / "val.parquet").exists())
            self.assertTrue((splits_dir / "test.parquet").exists())


if __name__ == "__main__":
    unittest.main()
