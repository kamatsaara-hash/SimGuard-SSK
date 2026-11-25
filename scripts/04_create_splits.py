from pathlib import Path
import torch
import numpy as np
from sklearn.model_selection import train_test_split

PATH = Path("data/processed/hetero_graph.pt")

print("Loading heterogeneous graph...")

data = torch.load(PATH, map_location="cpu", weights_only=False)
labels = data["subscriber"].y.numpy()

indices = np.arange(len(labels))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=labels[temp_idx]
)

train_mask = torch.zeros(len(labels), dtype=torch.bool)
val_mask = torch.zeros(len(labels), dtype=torch.bool)
test_mask = torch.zeros(len(labels), dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data["subscriber"].train_mask = train_mask
data["subscriber"].val_mask = val_mask
data["subscriber"].test_mask = test_mask

torch.save(data, PATH)

print("\n========== DATA SPLIT SUMMARY ==========")

for name, idx in [
    ("TRAIN", train_idx),
    ("VALIDATION", val_idx),
    ("TEST", test_idx)
]:
    positive = int(labels[idx].sum())
    negative = len(idx) - positive

    print(f"\n{name}")
    print("Total:", len(idx))
    print("Fraud:", positive)
    print("Provisional non-fraud:", negative)

print("\nSaved split masks into hetero_graph.pt")
print("Split completed successfully.")
