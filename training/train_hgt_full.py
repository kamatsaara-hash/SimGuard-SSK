from pathlib import Path
import json
import random
import time
import copy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

from models.hgt import HGTFraudDetector

# ---------------- CONFIGURATION ----------------

SEED = 42
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

HIDDEN_CHANNELS = 32
HEADS = 2
DROPOUT = 0.3

GRAPH_PATH = Path("data/processed_full_split/hetero_graph_split_v2.pt")
OUTPUT_DIR = Path("results/HGT_full")

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Limit CPU thread contention.
torch.set_num_threads(4)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("SIMGuard - FULL DATASET HGT TRAINING")
print("=" * 65)

# ---------------- LOAD GRAPH ----------------

print("\nLoading corrected partition-isolated graph...")
start_time = time.time()

data = torch.load(
    GRAPH_PATH,
    map_location=DEVICE,
    weights_only=False
)

metadata = data.metadata()

x_dict = {
    node_type: data[node_type].x.float()
    for node_type in data.node_types
}

edge_index_dict = data.edge_index_dict

labels = data["subscriber"].y.float()

train_mask = data["subscriber"].train_mask.bool()
val_mask = data["subscriber"].val_mask.bool()
test_mask = data["subscriber"].test_mask.bool()

input_dims = {
    node_type: data[node_type].x.size(1)
    for node_type in data.node_types
}

print("\n========== GRAPH SUMMARY ==========")
print("Graph:", GRAPH_PATH)
print("Node types:", data.node_types)
print("Edge types:", len(data.edge_types))
print("Input dimensions:", input_dims)
print("Subscribers:", len(labels))
print("Training:", int(train_mask.sum()))
print("Validation:", int(val_mask.sum()))
print("Testing:", int(test_mask.sum()))

# Check masks and labels.
assert not torch.any(train_mask & val_mask)
assert not torch.any(train_mask & test_mask)
assert not torch.any(val_mask & test_mask)
assert int(train_mask.sum() + val_mask.sum() + test_mask.sum()) == len(labels)

# ---------------- CLASS BALANCING ----------------

train_labels = labels[train_mask]

positive_count = int(train_labels.sum().item())
negative_count = int(len(train_labels) - positive_count)

if positive_count == 0:
    raise ValueError("No positive fraud labels in training split.")

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32
)

print("\n========== CLASS BALANCING ==========")
print("Training fraud:", positive_count)
print("Training assumed negatives:", negative_count)
print("Positive class weight:", round(pos_weight.item(), 4))

# ---------------- MODEL ----------------

model = HGTFraudDetector(
    metadata=metadata,
    input_dims=input_dims,
    hidden_channels=HIDDEN_CHANNELS,
    heads=HEADS,
    dropout=DROPOUT
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("\nModel parameters:", sum(p.numel() for p in model.parameters()))
print("Starting full-batch HGT training...")
print("Device:", DEVICE)

# ---------------- TRAINING ----------------

best_val_ap = -1.0
best_epoch = 0
best_state = None
epochs_without_improvement = 0
history = []

training_start = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    model.train()
    optimizer.zero_grad()

    logits = model(x_dict, edge_index_dict)

    train_loss = criterion(
        logits[train_mask],
        labels[train_mask]
    )

    train_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=2.0
    )

    optimizer.step()

    # Validation
    model.eval()

    with torch.no_grad():
        val_logits = model(x_dict, edge_index_dict)

        val_probs = torch.sigmoid(
            val_logits[val_mask]
        ).cpu().numpy()

        val_true = labels[val_mask].cpu().numpy()

        val_loss = criterion(
            val_logits[val_mask],
            labels[val_mask]
        ).item()

    val_ap = average_precision_score(val_true, val_probs)
    val_auc = roc_auc_score(val_true, val_probs)

    epoch_seconds = time.time() - epoch_start

    record = {
        "epoch": epoch,
        "train_loss": float(train_loss.item()),
        "validation_loss": float(val_loss),
        "validation_pr_auc": float(val_ap),
        "validation_roc_auc": float(val_auc),
        "epoch_seconds": float(epoch_seconds)
    }

    history.append(record)

    if val_ap > best_val_ap:
        best_val_ap = float(val_ap)
        best_epoch = epoch

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        epochs_without_improvement = 0

        # Save checkpoint immediately.
        torch.save(
            {
                "model_state_dict": best_state,
                "metadata": metadata,
                "input_dims": input_dims,
                "hidden_channels": HIDDEN_CHANNELS,
                "heads": HEADS,
                "dropout": DROPOUT,
                "best_epoch": best_epoch,
                "best_val_pr_auc": best_val_ap
            },
            OUTPUT_DIR / "best_hgt.pt"
        )

    else:
        epochs_without_improvement += 1

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss.item():.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val PR-AUC: {val_ap:.5f} | "
        f"Val ROC-AUC: {val_auc:.5f} | "
        f"Time: {epoch_seconds:.1f}s"
    )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

if best_state is None:
    raise RuntimeError("No valid HGT checkpoint was generated.")

print("\nTraining time:", round(time.time() - training_start, 2), "seconds")
print("Best epoch:", best_epoch)
print("Best validation PR-AUC:", round(best_val_ap, 6))

# ---------------- RESTORE BEST MODEL ----------------

checkpoint = torch.load(
    OUTPUT_DIR / "best_hgt.pt",
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ---------------- VALIDATION THRESHOLD ----------------

print("\nSelecting threshold using validation data...")

with torch.no_grad():
    all_logits = model(x_dict, edge_index_dict)

    val_probs = torch.sigmoid(
        all_logits[val_mask]
    ).cpu().numpy()

    test_probs = torch.sigmoid(
        all_logits[test_mask]
    ).cpu().numpy()

val_true = labels[val_mask].cpu().numpy().astype(int)
test_true = labels[test_mask].cpu().numpy().astype(int)

# Select the threshold maximizing validation F1.
thresholds = np.linspace(0.01, 0.99, 991)

best_threshold = 0.5
best_val_f1 = -1.0

for threshold in thresholds:
    predictions = (val_probs >= threshold).astype(int)

    score = f1_score(
        val_true,
        predictions,
        zero_division=0
    )

    if score > best_val_f1:
        best_val_f1 = float(score)
        best_threshold = float(threshold)

# ---------------- FINAL TEST EVALUATION ----------------

print("\nEvaluating untouched test partition...")

test_predictions = (
    test_probs >= best_threshold
).astype(int)

cm = confusion_matrix(
    test_true,
    test_predictions,
    labels=[0, 1]
)

test_metrics = {
    "experiment": "HGT_full_dataset_group_split",
    "graph": str(GRAPH_PATH),
    "seed": SEED,
    "best_epoch": int(best_epoch),
    "best_validation_pr_auc": float(best_val_ap),
    "validation_selected_threshold": float(best_threshold),
    "validation_f1_at_selected_threshold": float(best_val_f1),

    "test_accuracy": float(
        accuracy_score(test_true, test_predictions)
    ),
    "test_precision": float(
        precision_score(test_true, test_predictions, zero_division=0)
    ),
    "test_recall": float(
        recall_score(test_true, test_predictions, zero_division=0)
    ),
    "test_f1": float(
        f1_score(test_true, test_predictions, zero_division=0)
    ),
    "test_roc_auc": float(
        roc_auc_score(test_true, test_probs)
    ),
    "test_pr_auc": float(
        average_precision_score(test_true, test_probs)
    ),

    "test_confusion_matrix": cm.tolist(),
    "test_fraud_count": int(test_true.sum()),
    "test_sample_count": int(len(test_true)),
    "training_fraud_count": positive_count,
    "training_assumed_negative_count": negative_count
}

with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=4)

report = classification_report(
    test_true,
    test_predictions,
    labels=[0, 1],
    target_names=["Assumed_Non_Fraud", "Known_Fraud"],
    zero_division=0
)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

# ---------------- RESULTS ----------------

print("\n" + "=" * 65)
print("FINAL HGT TEST RESULTS")
print("=" * 65)

print("Best epoch:", best_epoch)
print("Validation PR-AUC:", round(best_val_ap, 6))
print("Selected threshold:", round(best_threshold, 4))

print("\nTEST METRICS")
print("Accuracy :", round(test_metrics["test_accuracy"], 6))
print("Precision:", round(test_metrics["test_precision"], 6))
print("Recall   :", round(test_metrics["test_recall"], 6))
print("F1-score :", round(test_metrics["test_f1"], 6))
print("ROC-AUC  :", round(test_metrics["test_roc_auc"], 6))
print("PR-AUC   :", round(test_metrics["test_pr_auc"], 6))

print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
print(cm)

print("\nClassification Report:")
print(report)

print("\nSaved results:")
print(OUTPUT_DIR / "best_hgt.pt")
print(OUTPUT_DIR / "training_history.csv")
print(OUTPUT_DIR / "test_metrics.json")
print(OUTPUT_DIR / "classification_report.txt")

print("\nTotal elapsed time:", round(time.time() - start_time, 2), "seconds")
print("\nHGT experiment completed.")
