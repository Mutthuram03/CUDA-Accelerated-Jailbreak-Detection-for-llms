import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path("c:/Users/srmut/Desktop/CUDA_JAILBREAK/CUDA-Accelerated-Jailbreak-Detection-for-llms")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.extractors import HybridFeatureExtractor
from src.evaluation.metrics import calculate_jailbreak_metrics
from sklearn.linear_model import LogisticRegression

train_df = pd.read_parquet(REPO_ROOT / "data/splits/train.parquet")
val_df = pd.read_parquet(REPO_ROOT / "data/splits/val.parquet")
test_df = pd.read_parquet(REPO_ROOT / "data/splits/test.parquet")

train_texts = train_df["prompt"].tolist()
train_labels = train_df["label"].to_numpy(dtype=np.int32)
val_texts = val_df["prompt"].tolist()
val_labels = val_df["label"].to_numpy(dtype=np.int32)
test_texts = test_df["prompt"].tolist()
test_labels = test_df["label"].to_numpy(dtype=np.int32)

print(f"Loaded {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test.", flush=True)

print("Extracting features with expanded n-grams...", flush=True)
t0 = time.time()
ext = HybridFeatureExtractor(
    word_ngram_range=(1, 3),
    word_max_features=30000,
    char_ngram_range=(2, 5),
    char_max_features=8000,
    min_df=2,
    sublinear_tf=True
)
X_train = ext.fit_transform(train_texts)
X_val = ext.transform(val_texts)
X_test = ext.transform(test_texts)
print(f"Feature matrix extracted in {time.time()-t0:.1f}s. Shape: {X_train.shape}", flush=True)

for c_val in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
    clf = LogisticRegression(C=c_val, max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(X_train, train_labels)

    val_probs = clf.predict_proba(X_val)[:, 1]

    # 1. Standard 0.5 threshold
    acc_05 = float(np.mean((val_probs >= 0.5).astype(np.int32) == val_labels))

    # 2. Threshold tuned for maximal validation accuracy
    best_t = 0.5
    best_acc = 0.0
    for t in np.linspace(0.2, 0.8, 61):
        acc = float(np.mean((val_probs >= t).astype(np.int32) == val_labels))
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    # Evaluate on held-out test set
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_t).astype(np.int32)
    rep = calculate_jailbreak_metrics(test_labels, test_preds, test_probs)

    print(f"C={c_val:<4} | Best Val Thresh={best_t:.2f} (Val Acc={best_acc*100:.2f}%) | TEST Acc={rep.accuracy*100:.2f}% | F1={rep.f1*100:.2f}% | Prec={rep.precision*100:.2f}% | Rec={rep.recall*100:.2f}% | AUC={rep.roc_auc:.4f} | FP={rep.false_positives} | FN={rep.false_negatives}", flush=True)
