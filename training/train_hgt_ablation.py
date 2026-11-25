from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import copy
import random

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
    confusion_matrix
)

from models.hgt import HGTFraudDetector


# ---------------- CONFIGURATION ----------------

SEED = 42
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4
HIDDEN_CHANNELS = 32
HEADS = 2
DROPOUT = 0.3

GRAPH_PATH = Path("data/processed/hetero_graph.pt")
OUTPUT_DIR = Path("results/HGT_ablation")

DEVICE = torch.device("cpu")

# Remove outgoing_count, incoming_count, mean_size, type_DATA.
REMOVED_INDICES = [0, 1, 8, 9]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------- LOAD GRAPH ----------------

print("Loading heterogeneous graph...")

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

original_dim = x_dict["subscriber"].shape[1]

if original_dim != 19:
    raise ValueError(
        f"Expected 19 subscriber features, found {original_dim}"
    )

keep_indices = [
    i for i in range(original_dim)
    if i not in REMOVED_INDICES
]

x_dict["subscriber"] = x_dict["subscriber"][:, keep_indices]

edge_index_dict = data.edge_index_dict

labels = data["subscriber"].y.float()

train_mask = data["subscriber"].train_mask
val_mask = data["subscriber"].val_mask
test_mask = data["subscriber"].test_mask

input_dims = {
    node_type: x_dict[node_type].size(1)
    for node_type in data.node_types
}

print("\n========== ABLATION GRAPH SUMMARY ==========")
print("Node types:", data.node_types)
print("Edge types:", len(data.edge_types))
print("Original subscriber features:", original_dim)
print("Removed indices:", REMOVED_INDICES)
print("Remaining indices:", keep_indices)
print("New input dimensions:", input_dims)
print("Subscribers:", len(labels))
print("Training samples:", int(train_mask.sum()))
print("Validation samples:", int(val_mask.sum()))
print("Testing samples:", int(test_mask.sum()))


# ---------------- CLASS IMBALANCE ----------------

train_labels = labels[train_mask]

positive_count = int(train_labels.sum().item())
negative_count = int(len(train_labels) - positive_count)

if positive_count == 0:
    raise ValueError("Training split contains no fraud samples.")

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32,
    device=DEVICE
)

print("\n========== CLASS BALANCING ==========")
print("Training fraud:", positive_count)
print("Training provisional non-fraud:", negative_count)
print("Positive class weight:", round(pos_weight.item(), 3))


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
print("Starting HGT ablation training...")


# ---------------- TRAINING ----------------

best_val_ap = -1.0
best_epoch = 0
best_state = None
epochs_without_improvement = 0
history = []

for epoch in range(1, EPOCHS + 1):

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

    history.append({
        "epoch": epoch,
        "train_loss": float(train_loss.item()),
        "val_loss": float(val_loss),
        "val_pr_auc": float(val_ap),
        "val_roc_auc": float(val_auc)
    })

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PR-AUC: {val_ap:.4f} | "
            f"Val ROC-AUC: {val_auc:.4f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break


# ---------------- SAVE BEST MODEL ----------------

if best_state is None:
    raise RuntimeError("Training did not produce a valid checkpoint.")

model.load_state_dict(best_state)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": best_state,
        "metadata": metadata,
        "input_dims": input_dims,
        "hidden_channels": HIDDEN_CHANNELS,
        "heads": HEADS,
        "dropout": DROPOUT,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_ap,
        "removed_feature_indices": REMOVED_INDICES,
        "kept_feature_indices": keep_indices
    },
    OUTPUT_DIR / "best_hgt_ablation.pt"
)

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)

print("\n========== ABLATION TRAINING COMPLETE ==========")
print("Best epoch:", best_epoch)
print("Best validation PR-AUC:", round(best_val_ap, 4))


# ---------------- VALIDATION THRESHOLD ----------------

model.eval()

with torch.no_grad():
    all_logits = model(x_dict, edge_index_dict)

    val_probs = torch.sigmoid(
        all_logits[val_mask]
    ).cpu().numpy()

    test_probs = torch.sigmoid(
        all_logits[test_mask]
    ).cpu().numpy()

val_true = labels[val_mask].cpu().numpy()
test_true = labels[test_mask].cpu().numpy()

thresholds = np.unique(
    np.concatenate([val_probs, np.array([0.5])])
)

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
        best_val_f1 = score
        best_threshold = float(threshold)


# ---------------- FINAL TEST EVALUATION ----------------

test_predictions = (test_probs >= best_threshold).astype(int)

test_metrics = {
    "best_epoch": int(best_epoch),
    "best_validation_pr_auc": float(best_val_ap),
    "validation_selected_threshold": float(best_threshold),
    "validation_f1_at_selected_threshold": float(best_val_f1),
    "test_accuracy": float(accuracy_score(test_true, test_predictions)),
    "test_precision": float(precision_score(test_true, test_predictions, zero_division=0)),
    "test_recall": float(recall_score(test_true, test_predictions, zero_division=0)),
    "test_f1": float(f1_score(test_true, test_predictions, zero_division=0)),
    "test_roc_auc": float(roc_auc_score(test_true, test_probs)),
    "test_pr_auc": float(average_precision_score(test_true, test_probs)),
    "test_confusion_matrix": confusion_matrix(
        test_true,
        test_predictions,
        labels=[0, 1]
    ).tolist(),
    "test_fraud_count": int(test_true.sum()),
    "test_sample_count": int(len(test_true)),
    "removed_feature_indices": REMOVED_INDICES
}

with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=4)

print("\n========== HGT ABLATION TEST RESULTS ==========")
print("Threshold:", round(best_threshold, 4))
print("Accuracy:", round(test_metrics["test_accuracy"], 4))
print("Precision:", round(test_metrics["test_precision"], 4))
print("Recall:", round(test_metrics["test_recall"], 4))
print("F1-score:", round(test_metrics["test_f1"], 4))
print("ROC-AUC:", round(test_metrics["test_roc_auc"], 4))
print("PR-AUC:", round(test_metrics["test_pr_auc"], 4))

print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
print(np.array(test_metrics["test_confusion_matrix"]))

print("\nSaved files:")
print(OUTPUT_DIR / "best_hgt_ablation.pt")
print(OUTPUT_DIR / "training_history.csv")
print(OUTPUT_DIR / "test_metrics.json")

print("\nHGT ablation training and evaluation completed.")
