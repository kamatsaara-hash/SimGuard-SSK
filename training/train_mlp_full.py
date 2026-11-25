import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

from models.mlp_full import MLPFraudDetector


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_PATH = "data/processed_full_split/hetero_graph_split_v2.pt"
OUTPUT_DIR = "results/MLP_full"

EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

device = torch.device("cpu")

print("=" * 65)
print("SIMGuard - FULL DATASET MLP TRAINING")
print("=" * 65)


# ============================================================
# LOAD GRAPH AND SPLITS
# ============================================================

print("\nLoading corrected partition-isolated graph...")

try:
    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False
    )
except TypeError:
    graph = torch.load(GRAPH_PATH, map_location="cpu")

x = graph["subscriber"].x.float()
y = graph["subscriber"].y.long()

train_mask = graph["subscriber"].train_mask.bool()
val_mask = graph["subscriber"].val_mask.bool()
test_mask = graph["subscriber"].test_mask.bool()

print("\n========== DATA SUMMARY ==========")
print("Subscriber features:", tuple(x.shape))
print("Training:", int(train_mask.sum()))
print("Validation:", int(val_mask.sum()))
print("Testing:", int(test_mask.sum()))

print("\n========== CLASS BALANCING ==========")

train_labels = y[train_mask]

positive = int((train_labels == 1).sum())
negative = int((train_labels == 0).sum())

pos_weight = negative / positive

print("Training fraud:", positive)
print("Training assumed negatives:", negative)
print("Positive class weight:", round(pos_weight, 4))


# ============================================================
# MODEL
# ============================================================

model = MLPFraudDetector(
    input_dim=x.shape[1],
    dropout=DROPOUT
).to(device)

print("\nModel parameters:", sum(
    p.numel() for p in model.parameters()
))

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight)
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING
# ============================================================

best_val_pr_auc = -1
best_epoch = 0
best_state = None
patience_counter = 0
history = []

print("\nStarting MLP training...")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):

    model.train()
    optimizer.zero_grad()

    logits = model(x)

    train_loss = criterion(
        logits[train_mask],
        y[train_mask].float()
    )

    train_loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        val_logits = model(x)

        val_loss = criterion(
            val_logits[val_mask],
            y[val_mask].float()
        )

        val_probs = torch.sigmoid(
            val_logits[val_mask]
        ).cpu().numpy()

        val_true = y[val_mask].cpu().numpy()

        val_pr_auc = average_precision_score(
            val_true,
            val_probs
        )

        val_roc_auc = roc_auc_score(
            val_true,
            val_probs
        )

    history.append({
        "epoch": epoch,
        "train_loss": float(train_loss.item()),
        "validation_loss": float(val_loss.item()),
        "validation_pr_auc": float(val_pr_auc),
        "validation_roc_auc": float(val_roc_auc)
    })

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss.item():.4f} | "
        f"Val Loss: {val_loss.item():.4f} | "
        f"Val PR-AUC: {val_pr_auc:.5f} | "
        f"Val ROC-AUC: {val_roc_auc:.5f}"
    )

    if val_pr_auc > best_val_pr_auc:
        best_val_pr_auc = val_pr_auc
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


training_time = time.time() - start_time

print("\nTraining time:", round(training_time, 2), "seconds")
print("Best epoch:", best_epoch)
print("Best validation PR-AUC:", round(best_val_pr_auc, 6))


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(best_state)
model.eval()

torch.save(
    {
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "validation_pr_auc": best_val_pr_auc,
        "input_dim": x.shape[1]
    },
    os.path.join(OUTPUT_DIR, "best_mlp.pt")
)

pd.DataFrame(history).to_csv(
    os.path.join(OUTPUT_DIR, "training_history.csv"),
    index=False
)


# ============================================================
# VALIDATION THRESHOLD SELECTION
# ============================================================

print("\nSelecting threshold using validation data...")

with torch.no_grad():
    all_probs = torch.sigmoid(model(x)).cpu().numpy()

val_true = y[val_mask].cpu().numpy()
val_probs = all_probs[val_mask.cpu().numpy()]

precision_curve, recall_curve, thresholds = precision_recall_curve(
    val_true,
    val_probs
)

f1_values = (
    2 * precision_curve[:-1] * recall_curve[:-1]
    / (precision_curve[:-1] + recall_curve[:-1] + 1e-12)
)

best_threshold_index = int(np.argmax(f1_values))
threshold = float(thresholds[best_threshold_index])

print("Selected threshold:", round(threshold, 6))
print("Validation F1:", round(float(f1_values[best_threshold_index]), 6))


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating untouched test partition...")

test_true = y[test_mask].cpu().numpy()
test_probs = all_probs[test_mask.cpu().numpy()]

test_predictions = (
    test_probs >= threshold
).astype(int)

metrics = {
    "model": "MLP",
    "best_epoch": int(best_epoch),
    "validation_pr_auc": float(best_val_pr_auc),
    "threshold": float(threshold),
    "accuracy": float(accuracy_score(test_true, test_predictions)),
    "precision": float(precision_score(test_true, test_predictions, zero_division=0)),
    "recall": float(recall_score(test_true, test_predictions, zero_division=0)),
    "f1": float(f1_score(test_true, test_predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(test_true, test_probs)),
    "pr_auc": float(average_precision_score(test_true, test_probs)),
    "confusion_matrix": confusion_matrix(
        test_true,
        test_predictions
    ).tolist(),
    "training_time_seconds": float(training_time)
}

print("\n" + "=" * 65)
print("FINAL MLP TEST RESULTS")
print("=" * 65)

for key, value in metrics.items():
    print(f"{key}: {value}")

report = classification_report(
    test_true,
    test_predictions,
    target_names=["Assumed_Non_Fraud", "Known_Fraud"],
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    os.path.join(OUTPUT_DIR, "test_metrics.json"),
    "w"
) as f:
    json.dump(metrics, f, indent=4)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w"
) as f:
    f.write(report)

print("\nSaved results:")
print(os.path.join(OUTPUT_DIR, "best_mlp.pt"))
print(os.path.join(OUTPUT_DIR, "training_history.csv"))
print(os.path.join(OUTPUT_DIR, "test_metrics.json"))
print(os.path.join(OUTPUT_DIR, "classification_report.txt"))

print("\nMLP experiment completed.")

