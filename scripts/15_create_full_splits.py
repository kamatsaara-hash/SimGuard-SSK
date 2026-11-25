from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

ROOT = Path("data/processed_full")
OUT = Path("data/processed_full_split")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42

print("Loading labels and full graph...")

labels = pd.read_csv(ROOT / "subscriber_labels.csv", dtype=str)
labels["label"] = pd.to_numeric(labels["label"]).astype(int)

graph = torch.load(
    ROOT / "hetero_graph.pt",
    map_location="cpu",
    weights_only=False
)

with open(ROOT / "node_mappings.json", encoding="utf-8") as f:
    mappings = json.load(f)

assert len(labels) == graph["subscriber"].num_nodes
assert mappings["subscriber"] == labels["subscriber_id"].tolist()

# One row per original subscriber, with a consistent label
groups = labels[
    ["original_subscriber_id", "label"]
].drop_duplicates()

if groups["original_subscriber_id"].duplicated().any():
    raise ValueError("Conflicting labels detected for original subscriber IDs.")

print("Unique original subscribers:", len(groups))
print("Fraud-associated original subscribers:", int(groups["label"].sum()))

# 70% train, 15% validation, 15% test
train_groups, holdout_groups = train_test_split(
    groups,
    test_size=0.30,
    random_state=SEED,
    stratify=groups["label"]
)

val_groups, test_groups = train_test_split(
    holdout_groups,
    test_size=0.50,
    random_state=SEED,
    stratify=holdout_groups["label"]
)

split_by_original = {}

for frame, split_id in [
    (train_groups, 0),
    (val_groups, 1),
    (test_groups, 2)
]:
    for original_id in frame["original_subscriber_id"]:
        split_by_original[original_id] = split_id

split_ids = labels["original_subscriber_id"].map(
    split_by_original
).to_numpy(dtype=np.int64)

if (split_ids < 0).any():
    raise ValueError("Some subscriber instances were not assigned a split.")

train_mask = split_ids == 0
val_mask = split_ids == 1
test_mask = split_ids == 2

# Verify original identities never cross partitions
identity_check = pd.DataFrame({
    "original_id": labels["original_subscriber_id"],
    "split": split_ids
})

assert identity_check.groupby("original_id")["split"].nunique().max() == 1

print("\n========== SPLIT DISTRIBUTION ==========")

for name, mask in [
    ("Train", train_mask),
    ("Validation", val_mask),
    ("Test", test_mask)
]:
    print(
        f"{name}: {mask.sum():,} subscribers | "
        f"{labels.loc[mask, 'label'].sum():,} fraud"
    )

# Create a new graph. Subscriber nodes retain their original indices.
new_graph = HeteroData()

new_graph["subscriber"].x = graph["subscriber"].x
new_graph["subscriber"].y = graph["subscriber"].y
new_graph["subscriber"].train_mask = torch.tensor(train_mask)
new_graph["subscriber"].val_mask = torch.tensor(val_mask)
new_graph["subscriber"].test_mask = torch.tensor(test_mask)
new_graph["subscriber"].split_id = torch.tensor(split_ids)

# Communication edges: retain only edges within the same partition.
comm = graph[
    "subscriber", "communicates", "subscriber"
].edge_index

comm_np = comm.numpy()
keep_comm = split_ids[comm_np[0]] == split_ids[comm_np[1]]

new_graph[
    "subscriber", "communicates", "subscriber"
].edge_index = comm[:, torch.tensor(keep_comm)]

# Duplicate device/cell nodes by partition to prevent cross-partition
# message passing through shared device or cell nodes.
def isolate_entity(relation, entity_type):
    edges = graph[relation].edge_index.numpy()
    original_x = graph[entity_type].x

    new_edge_parts = []
    new_feature_parts = []
    offset = 0

    for partition in (0, 1, 2):
        mask = split_ids[edges[0]] == partition
        selected = edges[:, mask]

        if selected.shape[1] == 0:
            continue

        old_entity_ids = selected[1]
        unique_ids, inverse = np.unique(
            old_entity_ids,
            return_inverse=True
        )

        new_sub_ids = selected[0]
        new_entity_ids = inverse.astype(np.int64) + offset

        new_edge_parts.append(
            torch.tensor(
                np.stack([new_sub_ids, new_entity_ids]),
                dtype=torch.long
            )
        )

        new_feature_parts.append(original_x[
            torch.tensor(unique_ids, dtype=torch.long)
        ])

        offset += len(unique_ids)

    if new_edge_parts:
        edge_index = torch.cat(new_edge_parts, dim=1)
        x = torch.cat(new_feature_parts, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        x = original_x[:0]

    new_graph[entity_type].x = x
    return edge_index

device_edges = isolate_entity(
    ("subscriber", "uses_device", "device"),
    "device"
)

cell_edges = isolate_entity(
    ("subscriber", "observed_at", "cell"),
    "cell"
)

new_graph[
    "subscriber", "uses_device", "device"
].edge_index = device_edges

new_graph[
    "device", "used_by", "subscriber"
].edge_index = device_edges.flip(0)

new_graph[
    "subscriber", "observed_at", "cell"
].edge_index = cell_edges

new_graph[
    "cell", "has_subscriber", "subscriber"
].edge_index = cell_edges.flip(0)

# Verify that no edge crosses subscriber partitions.
for edge_type in [
    ("subscriber", "communicates", "subscriber"),
]:
    e = new_graph[edge_type].edge_index.numpy()
    assert np.all(split_ids[e[0]] == split_ids[e[1]])

for edge_type in [
    ("subscriber", "uses_device", "device"),
    ("subscriber", "observed_at", "cell")
]:
    e = new_graph[edge_type].edge_index.numpy()
    # Each entity node is partition-specific by construction.
    assert e.shape[0] == 2

# Save the isolated graph and split metadata.
graph_path = OUT / "hetero_graph_split.pt"
torch.save(new_graph, graph_path)

split_df = labels.copy()
split_df["split"] = np.array(
    ["train", "validation", "test"]
)[split_ids]

split_df.to_csv(OUT / "subscriber_splits.csv", index=False)

summary = {
    "seed": SEED,
    "split_ratio": {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15
    },
    "unique_original_subscribers": int(len(groups)),
    "subscriber_instances": int(len(labels)),
    "train_instances": int(train_mask.sum()),
    "validation_instances": int(val_mask.sum()),
    "test_instances": int(test_mask.sum()),
    "train_fraud": int(labels.loc[train_mask, "label"].sum()),
    "validation_fraud": int(labels.loc[val_mask, "label"].sum()),
    "test_fraud": int(labels.loc[test_mask, "label"].sum()),
    "communication_edges": int(new_graph[
        "subscriber", "communicates", "subscriber"
    ].edge_index.shape[1]),
    "device_edges": int(device_edges.shape[1]),
    "cell_edges": int(cell_edges.shape[1]),
    "isolated_device_nodes": int(new_graph["device"].num_nodes),
    "isolated_cell_nodes": int(new_graph["cell"].num_nodes)
}

with open(OUT / "split_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n========== ISOLATED GRAPH SUMMARY ==========")
print(json.dumps(summary, indent=4))
print("\nSaved:", graph_path)
print("Saved:", OUT / "subscriber_splits.csv")
print("Saved:", OUT / "split_summary.json")
