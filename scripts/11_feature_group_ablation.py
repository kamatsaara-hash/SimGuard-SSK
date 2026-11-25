from pathlib import Path
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
DROPOUT = 0.3

DEVICE = torch.device("cpu")

FEATURES = {
    "A_activity": [6, 7, 17, 18],
    "B_network": [2, 3, 4, 5],
    "C_combined": [2, 3, 4, 5, 6, 7, 17, 18]
}

FEATURE_NAMES = [
    "outgoing_count", "incoming_count",
    "unique_out_contacts", "unique_in_contacts",
    "unique_devices", "unique_cells",
    "mean_duration", "std_duration", "mean_size",
    "type_DATA", "type_I-MOC", "type_I-MTC",
    "type_MOC", "type_MO-SMS", "type_MTC",
    "international_out", "international_in",
    "night_events", "active_span_hours"
]


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


def run_experiment(name, indices, data):
    print("\n" + "=" * 60)
    print("EXPERIMENT:", name)
    print("FEATURES:", [FEATURE_NAMES[i] for i in indices])
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    x = data["subscriber"].x.float()[:, indices]
    labels = data["subscriber"].y.float()

    train_mask = data["subscriber"].train_mask
    val_mask = data["subscriber"].val_mask
    test_mask = data["subscriber"].test_mask

    train_labels = labels[train_mask]
    positives = int(train_labels.sum().item())
    negatives = len(train_labels) - positives

    pos_weight = torch.tensor(
        [negatives / positives], dtype=torch.float32
    )

    model = SubscriberMLP(x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_ap = -1.0
    best_state = None
    best_epoch = 0
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=2.0
        )
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x)
            val_probs = torch.sigmoid(
                val_logits[val_mask]
            ).numpy()
            val_true = labels[val_mask].numpy()

        val_ap = average_precision_score(val_true, val_probs)

        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Validation PR-AUC {val_ap:.4f}"
            )

        if wait >= PATIENCE:
            print("Early stopping at epoch", epoch)
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_probs = torch.sigmoid(model(x)[val_mask]).numpy()
        test_probs = torch.sigmoid(model(x)[test_mask]).numpy()

    val_true = labels[val_mask].numpy()
    test_true = labels[test_mask].numpy()

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.unique(
        np.concatenate([val_probs, [0.5]])
    ):
        predictions = (val_probs >= threshold).astype(int)
        score = f1_score(
            val_true, predictions, zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    test_pred = (test_probs >= best_threshold).astype(int)

    metrics = {
        "experiment": name,
        "feature_indices": indices,
        "feature_names": [FEATURE_NAMES[i] for i in indices],
        "best_epoch": best_epoch,
        "validation_pr_auc": float(best_ap),
        "threshold": best_threshold,
        "accuracy": float(accuracy_score(test_true, test_pred)),
        "precision": float(precision_score(
            test_true, test_pred, zero_division=0
        )),
        "recall": float(recall_score(
            test_true, test_pred, zero_division=0
        )),
        "f1": float(f1_score(
            test_true, test_pred, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(test_true, test_probs)),
        "pr_auc": float(average_precision_score(
            test_true, test_probs
        )),
        "confusion_matrix": confusion_matrix(
            test_true, test_pred, labels=[0, 1]
        ).tolist(),
        "test_fraud_count": int(test_true.sum()),
        "test_sample_count": int(len(test_true))
    }

    print("\nTEST RESULTS")
    for key in [
        "accuracy", "precision", "recall",
        "f1", "roc_auc", "pr_auc"
    ]:
        print(f"{key}: {metrics[key]:.4f}")

    print("Confusion matrix:", metrics["confusion_matrix"])

    output_dir = Path("results/feature_group_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{name}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


print("Loading existing graph data and subscriber splits...")

data = torch.load(
    "data/processed/hetero_graph.pt",
    map_location=DEVICE,
    weights_only=False
)

results = []

for name, indices in FEATURES.items():
    results.append(run_experiment(name, indices, data))

comparison = pd.DataFrame([
    {
        "Experiment": m["experiment"],
        "Features": len(m["feature_indices"]),
        "Accuracy": m["accuracy"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"],
        "ROC_AUC": m["roc_auc"],
        "PR_AUC": m["pr_auc"]
    }
    for m in results
])

output_dir = Path("results/feature_group_ablation")
comparison.to_csv(output_dir / "comparison.csv", index=False)

print("\n========== FEATURE GROUP COMPARISON ==========")
print(comparison.round(4).to_string(index=False))
print("\nSaved to:", output_dir)
