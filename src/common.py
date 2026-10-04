"""
Shared data loading, splitting, preprocessing and evaluation for the CIC-IDS2017
intrusion detection experiments (see ids_cnn.py and ids_rf.py).

Two evaluation protocols over the Tuesday (benign + FTP-Patator + SSH-Patator)
and Friday afternoon (benign + DDoS) capture days:

  random  Combine both days, balance the classes, then split 80/20 at random
          (stratified). Train and test share attack families. This is the usual
          binary CIC-IDS2017 evaluation.
  day     Train on Tuesday, test on Friday. No attack family appears in both
          days, so this measures generalization to unseen attack types.
"""

import os
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved, never shown

ROOT = Path(__file__).resolve().parent.parent

# Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "data" / "hf_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

REPO_ID = "c01dsnap/CIC-IDS2017"
TUESDAY_FILE = "Tuesday-WorkingHours.pcap_ISCX.csv"
FRIDAY_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# Identifiers and metadata that would let a model key on the capture
# environment instead of traffic behaviour.
DROP_PATTERNS = [
    "Flow ID", "Source IP", "Destination IP", "Source Port", "Destination Port",
    "Timestamp", "Label",
]
CLASS_NAMES = ["Benign", "Attack"]
SPLITS = ["random", "day"]


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------
def clean_columns(df):
    """Normalize whitespace in column names and de-duplicate repeated names."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]
    seen, cols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    df.columns = cols
    return df


def load_day(filename):
    """Download one capture day and return (numeric feature frame, binary labels)."""
    path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    df = clean_columns(pd.read_csv(path, low_memory=False))
    label_col = df.columns[-1]
    y = (df[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(int)

    keep = [c for c in df.columns if not any(p.lower() in c.lower() for p in DROP_PATTERNS)]
    X = df[keep].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X, y.to_numpy()


def balanced_indices(y, seed):
    """Row positions of an equal number of benign and attack flows, shuffled."""
    rng = np.random.RandomState(seed)
    n = np.bincount(y).min()
    benign = rng.choice(np.flatnonzero(y == 0), size=n, replace=False)
    attack = rng.choice(np.flatnonzero(y == 1), size=n, replace=False)
    idx = np.concatenate([benign, attack])
    rng.shuffle(idx)
    return idx


def build_splits(split, seed, results):
    """Return (X_train, X_test, y_train, y_test) as raw, uncleaned feature frames."""
    X_tue, y_tue = load_day(TUESDAY_FILE)
    X_fri, y_fri = load_day(FRIDAY_FILE)
    results["tuesday_flows"] = len(y_tue)
    results["friday_flows"] = len(y_fri)

    if split == "random":
        X = pd.concat([X_tue, X_fri], ignore_index=True)
        y = np.concatenate([y_tue, y_fri])
        results["combined_binary_counts"] = np.bincount(y).tolist()
        idx = balanced_indices(y, seed)
        X_bal, y_bal = X.iloc[idx].reset_index(drop=True), y[idx]
        return train_test_split(
            X_bal, y_bal, test_size=0.2, random_state=seed, stratify=y_bal
        )

    results["tuesday_binary_counts"] = np.bincount(y_tue).tolist()
    results["friday_binary_counts"] = np.bincount(y_fri).tolist()
    tr, te = balanced_indices(y_tue, seed), balanced_indices(y_fri, seed)
    return (
        X_tue.iloc[tr].reset_index(drop=True),
        X_fri.iloc[te].reset_index(drop=True),
        y_tue[tr],
        y_fri[te],
    )


def impute(X_train, X_test):
    """Drop all-missing columns and median-impute, fitting on the training split only."""
    all_nan = X_train.columns[X_train.isna().all()]
    X_train, X_test = X_train.drop(columns=all_nan), X_test.drop(columns=all_nan)
    medians = X_train.median()
    return X_train.fillna(medians), X_test.fillna(medians)


def record_split(results, X_train, X_test, y_train, y_test):
    results["feature_count"] = X_train.shape[1]
    results["train_shape"] = list(X_train.shape)
    results["test_shape"] = list(X_test.shape)
    results["train_classes"] = np.bincount(y_train).tolist()
    results["test_classes"] = np.bincount(y_test).tolist()
    print("Features:", results["feature_count"],
          "| train:", results["train_shape"], results["train_classes"],
          "| test:", results["test_shape"], results["test_classes"])


# ---------------------------------------------------------------------------
# Evaluation and figures
# ---------------------------------------------------------------------------
def evaluate(y_true, proba, results, threshold=0.5):
    """Score predicted attack probabilities on the test split and record them."""
    y_pred = (proba >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, proba),
    }
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print(report)

    results["metrics"] = {k: float(v) for k, v in metrics.items()}
    results["classification_report"] = report
    results["confusion_matrix"] = cm.tolist()
    return cm


def plot_class_distribution(split, results, y_train, y_test, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if split == "random":
        counts = results["combined_binary_counts"]
        axes[0].set_title("Combined CIC-IDS2017 (before balancing)")
        right_labels = ["Train Benign", "Train Attack", "Test Benign", "Test Attack"]
        right_counts = [int((y_train == 0).sum()), int((y_train == 1).sum()),
                        int((y_test == 0).sum()), int((y_test == 1).sum())]
        axes[1].set_title("Balanced stratified train/test split")
        axes[1].tick_params(axis="x", rotation=30)
    else:
        counts = results["tuesday_binary_counts"]
        axes[0].set_title("Tuesday (before balancing)")
        right_labels = CLASS_NAMES
        right_counts = results["friday_binary_counts"]
        axes[1].set_title("Friday DDoS (before balancing)")
    sns.barplot(x=CLASS_NAMES, y=counts, ax=axes[0])
    sns.barplot(x=right_labels, y=right_counts, ax=axes[1])
    for ax in axes:
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out / "fig1_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, split, out):
    title = {"random": "Confusion Matrix: Held-out Test Set",
             "day": "Confusion Matrix: Unseen Friday DDoS Day"}[split]
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out / "fig3_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
