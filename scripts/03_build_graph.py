from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

ROOT = Path("data/processed")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading preprocessed data...")

cdr = pd.read_csv(ROOT / "clean_cdr.csv")
labels = pd.read_csv(ROOT / "subscriber_labels.csv")

cdr["timestamp"] = pd.to_datetime(cdr["timestamp"], errors="coerce")
cdr["duration"] = pd.to_numeric(cdr["duration"], errors="coerce").fillna(0)
cdr["size"] = pd.to_numeric(cdr["size"], errors="coerce").fillna(0)

for col in ["src_number", "dst_number"]:
    cdr[col] = cdr[col].astype(str)

subscriber_ids = sorted(labels["subscriber_id"].astype(str).unique())
subscriber_map = {v: i for i, v in enumerate(subscriber_ids)}

# --------------------------------------------------
# 1. DEVICE AND CELL NODES
# --------------------------------------------------

def valid_id(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value.lower() in {"", "unknown", "nan", "none", "0"}:
        return None
    return value

device_pairs = set()
cell_pairs = set()

for row in cdr.itertuples(index=False):
    src = str(row.src_number)
    dst = str(row.dst_number)

    for subscriber, imei in [
        (src, row.src_imei),
        (dst, row.dst_imei)
    ]:
        device = valid_id(imei)
        if subscriber in subscriber_map and device:
            device_pairs.add((subscriber, device))

    for subscriber, cell in [
        (src, row.initial_src_cell),
        (src, row.current_src_cell),
        (dst, row.initial_dst_cell),
        (dst, row.current_dst_cell)
    ]:
        cell = valid_id(cell)
        if subscriber in subscriber_map and cell:
            cell_pairs.add((subscriber, cell))

device_ids = sorted({d for _, d in device_pairs})
cell_ids = sorted({c for _, c in cell_pairs})

device_map = {v: i for i, v in enumerate(device_ids)}
cell_map = {v: i for i, v in enumerate(cell_ids)}

print("Subscribers:", len(subscriber_ids))
print("Devices:", len(device_ids))
print("Cells:", len(cell_ids))

# --------------------------------------------------
# 2. SUBSCRIBER FEATURES
# --------------------------------------------------

print("\nEngineering subscriber features...")

feature_df = pd.DataFrame(index=subscriber_ids)

src = cdr.groupby("src_number")
dst = cdr.groupby("dst_number")

feature_df["outgoing_count"] = src.size().reindex(subscriber_ids).fillna(0)
feature_df["incoming_count"] = dst.size().reindex(subscriber_ids).fillna(0)

out_contacts = cdr.groupby("src_number")["dst_number"].nunique()
in_contacts = cdr.groupby("dst_number")["src_number"].nunique()

feature_df["unique_out_contacts"] = out_contacts.reindex(subscriber_ids).fillna(0)
feature_df["unique_in_contacts"] = in_contacts.reindex(subscriber_ids).fillna(0)

# Device and cell association counts
device_counts = defaultdict(set)
cell_counts = defaultdict(set)

for subscriber, device in device_pairs:
    device_counts[subscriber].add(device)

for subscriber, cell in cell_pairs:
    cell_counts[subscriber].add(cell)

feature_df["unique_devices"] = [
    len(device_counts[s]) for s in subscriber_ids
]
feature_df["unique_cells"] = [
    len(cell_counts[s]) for s in subscriber_ids
]

# Duration and size statistics
mean_duration = cdr.groupby("src_number")["duration"].mean()
std_duration = cdr.groupby("src_number")["duration"].std()
mean_size = cdr.groupby("src_number")["size"].mean()

feature_df["mean_duration"] = mean_duration.reindex(subscriber_ids).fillna(0)
feature_df["std_duration"] = std_duration.reindex(subscriber_ids).fillna(0)
feature_df["mean_size"] = mean_size.reindex(subscriber_ids).fillna(0)

# CDR-type activity counts
for cdr_type in sorted(cdr["cdr_type"].dropna().unique()):
    counts = cdr[cdr["cdr_type"] == cdr_type].groupby("src_number").size()
    feature_df[f"type_{cdr_type}"] = counts.reindex(subscriber_ids).fillna(0)

# International activity
international = cdr["cdr_type"].astype(str).str.startswith("I-")
international_out = cdr[international].groupby("src_number").size()
international_in = cdr[international].groupby("dst_number").size()

feature_df["international_out"] = international_out.reindex(subscriber_ids).fillna(0)
feature_df["international_in"] = international_in.reindex(subscriber_ids).fillna(0)

# Time-based features
cdr["hour"] = cdr["timestamp"].dt.hour
cdr["night_event"] = ((cdr["hour"] < 6) | (cdr["hour"] >= 22)).astype(float)

night = cdr.groupby("src_number")["night_event"].sum()
feature_df["night_events"] = night.reindex(subscriber_ids).fillna(0)

active_span = (
    cdr.groupby("src_number")["timestamp"].agg(
        lambda x: (x.max() - x.min()).total_seconds() / 3600
        if x.notna().any() else 0
    )
)
feature_df["active_span_hours"] = active_span.reindex(subscriber_ids).fillna(0)

# All features are nonnegative; log transform reduces scale differences.
feature_df = feature_df.replace([np.inf, -np.inf], 0).fillna(0)
feature_values = np.log1p(feature_df.clip(lower=0).values).astype(np.float32)

# --------------------------------------------------
# 3. NODE FEATURES
# --------------------------------------------------

device_degree = defaultdict(int)
cell_degree = defaultdict(int)

for subscriber, device in device_pairs:
    device_degree[device] += 1

for subscriber, cell in cell_pairs:
    cell_degree[cell] += 1

device_features = np.zeros((len(device_ids), 3), dtype=np.float32)
for device, idx in device_map.items():
    device_features[idx] = [
        1.0,
        np.log1p(device_degree[device]),
        np.log1p(sum(1 for _, d in device_pairs if d == device))
    ]

cell_features = np.zeros((len(cell_ids), 3), dtype=np.float32)
for cell, idx in cell_map.items():
    cell_features[idx] = [
        1.0,
        np.log1p(cell_degree[cell]),
        np.log1p(sum(1 for _, c in cell_pairs if c == cell))
    ]

# --------------------------------------------------
# 4. EDGE INDEX CONSTRUCTION
# --------------------------------------------------

print("\nConstructing graph edges...")

communication_edges = []
for row in cdr.itertuples(index=False):
    src_id = subscriber_map.get(str(row.src_number))
    dst_id = subscriber_map.get(str(row.dst_number))

    if src_id is not None and dst_id is not None and src_id != dst_id:
        communication_edges.append((src_id, dst_id))

def edge_tensor(edges):
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

subscriber_device = [
    (subscriber_map[s], device_map[d])
    for s, d in device_pairs
]

device_subscriber = [
    (device_map[d], subscriber_map[s])
    for s, d in device_pairs
]

subscriber_cell = [
    (subscriber_map[s], cell_map[c])
    for s, c in cell_pairs
]

cell_subscriber = [
    (cell_map[c], subscriber_map[s])
    for s, c in cell_pairs
]

# --------------------------------------------------
# 5. CREATE HETERODATA
# --------------------------------------------------

data = HeteroData()

data["subscriber"].x = torch.tensor(feature_values, dtype=torch.float32)

data["device"].x = torch.tensor(device_features, dtype=torch.float32)
data["cell"].x = torch.tensor(cell_features, dtype=torch.float32)

label_map = dict(zip(labels["subscriber_id"].astype(str), labels["label"].astype(int)))
data["subscriber"].y = torch.tensor(
    [int(label_map[s]) for s in subscriber_ids],
    dtype=torch.long
)

data["subscriber", "communicates", "subscriber"].edge_index = edge_tensor(
    communication_edges
)

data["subscriber", "uses_device", "device"].edge_index = edge_tensor(
    subscriber_device
)

data["device", "used_by", "subscriber"].edge_index = edge_tensor(
    device_subscriber
)

data["subscriber", "observed_at", "cell"].edge_index = edge_tensor(
    subscriber_cell
)

data["cell", "has_subscriber", "subscriber"].edge_index = edge_tensor(
    cell_subscriber
)

# --------------------------------------------------
# 6. SAVE GRAPH AND NODE MAPPINGS
# --------------------------------------------------

torch.save(data, OUT / "hetero_graph.pt")

mappings = {
    "subscriber": subscriber_ids,
    "device": device_ids,
    "cell": cell_ids
}

with open(OUT / "node_mappings.json", "w", encoding="utf-8") as f:
    json.dump(mappings, f)

print("\n========== GRAPH SUMMARY ==========")
print(data)
print("\nNode counts:")
for node_type in data.node_types:
    print(node_type, data[node_type].num_nodes)

print("\nEdge counts:")
for edge_type in data.edge_types:
    print(edge_type, data[edge_type].edge_index.shape[1])

print("\nSubscriber feature dimensions:", data["subscriber"].x.shape)
print("Fraud labels:", int(data["subscriber"].y.sum()))
print("Non-fraud labels:", int((data["subscriber"].y == 0).sum()))

print("\nSaved:")
print(OUT / "hetero_graph.pt")
print(OUT / "node_mappings.json")

