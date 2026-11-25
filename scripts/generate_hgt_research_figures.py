"""
SIMGuard-SSK: Publication figures and result tables for HGT experiments.

Run from the project root:
    python scripts/generate_hgt_research_figures.py

The script reads the experiment JSON/CSV files already produced by training.
It does not invent metrics. Missing experiment files are skipped with a warning.
Outputs are written to results/research_figures/.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT / "results" / "research_figures"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "Original HGT": "results/HGT_full/test_metrics.json",
    "Improved HGT (seed 42, 50 ep)": "results/HGT_improved/test_metrics.json",
    "Improved HGT (seed 7, 50 ep)": "results/HGT_seed7/test_metrics.json",
    "Improved HGT (seed 7, 100 ep)": "results/HGT_seed7_100ep/test_metrics.json",
    "Improved HGT (seed 42, 100 ep)": "results/HGT_seed42_100ep/test_metrics.json",
    "MLP baseline": "results/MLP_full/test_metrics.json",
}

HISTORY_FILES = {
    "HGT seed 7": "results/HGT_seed7_100ep/training_history.csv",
    "HGT seed 42": "results/HGT_seed42_100ep/training_history.csv",
}

METRICS = {
    "Accuracy": "test_accuracy",
    "Precision": "test_precision",
    "Recall": "test_recall",
    "F1-score": "test_f1",
    "ROC-AUC": "test_roc_auc",
    "PR-AUC": "test_pr_auc",
}

def load_results():
    rows = []
    raw = {}
    for label, rel_path in EXPERIMENTS.items():
        path = ROOT / rel_path
        if not path.exists():
            warnings.warn(f"Skipping missing result file: {rel_path}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.warn(f"Could not read {rel_path}: {exc}")
            continue

        row = {"Experiment": label, "Source_file": rel_path}
        for metric, key in METRICS.items():
            if key in data:
                row[metric] = float(data[key])
        row["Best_epoch"] = data.get("best_epoch", np.nan)
        row["Seed"] = data.get("seed", np.nan)
        row["Test_samples"] = data.get("test_sample_count", np.nan)
        row["Known_fraud"] = data.get("test_fraud_count", np.nan)
        row["Confusion_matrix"] = data.get("test_confusion_matrix")
        rows.append(row)
        raw[label] = data

    if not rows:
        raise FileNotFoundError(
            "No test_metrics.json files were found. Run this script from "
            "D:\\PaperPresentation\\SimGuard-SSK after confirming results folders."
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "model_metrics_comparison.csv", index=False)
    return df, raw

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close()

def plot_core_metrics(df):
    cols = ["Accuracy", "Precision", "Recall", "F1-score"]
    available = [c for c in cols if c in df.columns and df[c].notna().any()]
    if not available:
        return
    plot_df = df.set_index("Experiment")[available] * 100
    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.set_title("SIMGuard-SSK: Test Classification Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    plt.xticks(rotation=25, ha="right")
    savefig("figure_1_classification_metrics.png")

def plot_auc(df):
    cols = ["ROC-AUC", "PR-AUC"]
    available = [c for c in cols if c in df.columns and df[c].notna().any()]
    if not available:
        return
    plot_df = df.set_index("Experiment")[available] * 100
    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("AUC (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.set_title("SIMGuard-SSK: Test ROC-AUC and PR-AUC")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    savefig("figure_2_auc_comparison.png")

def plot_confusion_matrices(raw):
    labels = [
        ("Improved HGT (seed 7, 100 ep)", "figure_3_confusion_seed7.png"),
        ("Improved HGT (seed 42, 100 ep)", "figure_4_confusion_seed42.png"),
    ]
    for key, filename in labels:
        data = raw.get(key)
        if not data or not data.get("test_confusion_matrix"):
            continue
        cm = np.asarray(data["test_confusion_matrix"], dtype=int)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(cm)
        ax.set_title(f"Confusion Matrix: {key}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1], labels=["Assumed non-fraud", "Fraud"])
        ax.set_yticks([0, 1], labels=["Assumed non-fraud", "Known fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_training_histories():
    loaded = []
    for label, rel_path in HISTORY_FILES.items():
        path = ROOT / rel_path
        if not path.exists():
            warnings.warn(f"Skipping missing training history: {rel_path}")
            continue
        hist = pd.read_csv(path)
        required = {"epoch", "train_loss", "validation_loss", "validation_pr_auc"}
        if not required.issubset(hist.columns):
            warnings.warn(f"Unexpected columns in {rel_path}: {hist.columns.tolist()}")
            continue
        loaded.append((label, hist))

    if not loaded:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, hist in loaded:
        axes[0].plot(hist["epoch"], hist["train_loss"], label=f"{label} train")
        axes[0].plot(hist["epoch"], hist["validation_loss"],
                     linestyle="--", label=f"{label} validation")
        axes[1].plot(hist["epoch"], hist["validation_pr_auc"], label=label)

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation PR-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PR-AUC")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "figure_5_training_convergence.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

def write_two_seed_summary(raw):
    keys = [
        "Improved HGT (seed 7, 100 ep)",
        "Improved HGT (seed 42, 100 ep)",
    ]
    records = []
    for key in keys:
        data = raw.get(key)
        if data:
            records.append({
                "Seed": data.get("seed"),
                **{metric: data.get(json_key) for metric, json_key in METRICS.items()},
            })
    if len(records) == 2:
        two = pd.DataFrame(records)
        summary = pd.DataFrame({
            "Metric": list(METRICS.keys()),
            "Mean": [two[m].mean() for m in METRICS],
            "Sample_SD": [two[m].std(ddof=1) for m in METRICS],
        })
        two.to_csv(OUT / "hgt_two_seed_individual_results.csv", index=False)
        summary.to_csv(OUT / "hgt_two_seed_summary.csv", index=False)

def main():
    df, raw = load_results()
    plot_core_metrics(df)
    plot_auc(df)
    plot_confusion_matrices(raw)
    plot_training_histories()
    write_two_seed_summary(raw)

    print("\nResearch figures and tables generated in:")
    print(OUT)
    print("\nFiles:")
    for path in sorted(OUT.iterdir()):
        print(f" - {path.name}")
    print("\nNote: ROC/PR curves require saved per-subscriber prediction scores.")
    print("AUC values alone are not sufficient to reconstruct those curves.")

if __name__ == "__main__":
    main()
