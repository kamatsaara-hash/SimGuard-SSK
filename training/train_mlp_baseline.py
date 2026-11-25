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
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

SEED = 42
EPOCHS = 100
PATIENCE = 15
LR = 0.003
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

DEVICE = torch.device("cpu")
OUTPUT_DIR = Path("results/MLP_baseline")
REMOVED_INDICES = [0, 1, 8, 9]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class SubscriberMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


print("Loading graph data for subscriber features and splits...")

data = torch.load(
    "data/processed/hetero_graph.pt",
    map_location=DEVICE,
    weights_only=False
)

x = data["subscriber"].x.float()

keep_indices = [
    i for i in range(x.shape[1])
    if i not in REMOVED_INDICES
]

x = x[:, keep_indices]

labels = data["subscriber"].y.float()
train_mask = data["subscriber"].train_mask
val_mask = data["subscriber"].val_mask
test_mask = data["subscriber"].test_mask

print("Subscriber feature shape:", tuple(x.shape))
print("Features retained:", keep_indices)
print("Train:", int(train_mask.sum()))
print("Validation:", int(val_mask.sum()))
print("Test:", int(test_mask.sum()))

train_labels = labels[train_mask]
positive_count = int(train_labels.sum().item())
negative_count = len(train_labels) - positive_count

if positive_count == 0:
    raise ValueError("No positive training samples.")

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32
)

model = SubscriberMLP(x.shape[1]).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("\nStarting feature-only MLP training...")
print("Training fraud:", positive_count)
print("Training provisional non-fraud:", negative_count)

best_val_ap = -1.0
best_epoch = 0
best_state = None
wait = 0
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(x)
    train_loss = criterion(logits[train_mask], labels[train_mask])
    train_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(x)
        val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
        val_true = labels[val_mask].cpu().numpy()
        val_loss = criterion(
            val_logits[val_mask], labels[val_mask]
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
        wait = 0
    else:
        wait += 1

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PR-AUC: {val_ap:.4f} | "
            f"Val ROC-AUC: {val_auc:.4f}"
        )

    if wait >= PATIENCE:
        print("Early stopping at epoch", epoch)
        break

if best_state is None:
    raise RuntimeError("No valid MLP checkpoint.")

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    logits = model(x)
    val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
    test_probs = torch.sigmoid(logits[test_mask]).cpu().numpy()

val_true = labels[val_mask].cpu().numpy()
test_true = labels[test_mask].cpu().numpy()

best_threshold = 0.5
best_val_f1 = -1.0

for threshold in np.unique(np.concatenate([val_probs, [0.5]])):
    pred = (val_probs >= threshold).astype(int)
    score = f1_score(val_true, pred, zero_division=0)

    if score > best_val_f1:
        best_val_f1 = score
        best_threshold = float(threshold)

test_pred = (test_probs >= best_threshold).astype(int)

metrics = {
    "best_epoch": int(best_epoch),
    "best_validation_pr_auc": float(best_val_ap),
    "validation_selected_threshold": float(best_threshold),
    "validation_f1": float(best_val_f1),
    "test_accuracy": float(accuracy_score(test_true, test_pred)),
    "test_precision": float(precision_score(test_true, test_pred, zero_division=0)),
    "test_recall": float(recall_score(test_true, test_pred, zero_division=0)),
    "test_f1": float(f1_score(test_true, test_pred, zero_division=0)),
    "test_roc_auc": float(roc_auc_score(test_true, test_probs)),
    "test_pr_auc": float(average_precision_score(test_true, test_probs)),
    "test_confusion_matrix": confusion_matrix(
        test_true, test_pred, labels=[0, 1]
    ).tolist(),
    "test_fraud_count": int(test_true.sum()),
    "test_sample_count": int(len(test_true)),
    "removed_feature_indices": REMOVED_INDICES
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": best_state,
        "input_dim": int(x.shape[1]),
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_ap,
        "removed_feature_indices": REMOVED_INDICES,
        "kept_feature_indices": keep_indices
    },
    OUTPUT_DIR / "best_mlp.pt"
)

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)

with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n========== MLP TEST RESULTS ==========")
for key in [
    "test_accuracy", "test_precision", "test_recall",
    "test_f1", "test_roc_auc", "test_pr_auc"
]:
    print(key, ":", round(metrics[key], 4))

print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
print(np.array(metrics["test_confusion_matrix"]))

print("\nSaved results to:", OUTPUT_DIR)
