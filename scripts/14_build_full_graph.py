from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

ROOT = Path("data/processed_full")
CHUNK = 100_000

print("Loading subscriber labels...")
labels = pd.read_csv(ROOT / "subscriber_labels.csv", dtype=str)

subscriber_ids = labels["subscriber_id"].tolist()
N = len(subscriber_ids)
subscriber_map = {s: i for i, s in enumerate(subscriber_ids)}

print("Subscriber instances:", N)

# Subscriber feature accumulators
outgoing = np.zeros(N, dtype=np.float64)
incoming = np.zeros(N, dtype=np.float64)
duration_sum = np.zeros(N, dtype=np.float64)
duration_sq = np.zeros(N, dtype=np.float64)
size_sum = np.zeros(N, dtype=np.float64)
activity_count = np.zeros(N, dtype=np.int64)
night_events = np.zeros(N, dtype=np.float64)
intl_out = np.zeros(N, dtype=np.float64)
intl_in = np.zeros(N, dtype=np.float64)

types = ["DATA", "I-MOC", "I-MTC", "MOC", "MO-SMS", "MTC"]
type_counts = np.zeros((N, len(types)), dtype=np.float64)

min_time = np.full(N, np.iinfo(np.int64).max, dtype=np.int64)
max_time = np.full(N, np.iinfo(np.int64).min, dtype=np.int64)

device_map = {}
cell_map = {}

contact_pairs = set()
device_pairs = set()
cell_pairs = set()

def entity_ids(series, mapping, exclude_zero=False):
    values = series.astype("string")
    valid = values.notna()

    if exclude_zero:
        valid &= ~values.str.endswith("::0", na=False)

    result = np.full(len(values), -1, dtype=np.int64)

    unique_values = values[valid].unique()
    for value in unique_values:
        if value not in mapping:
            mapping[value] = len(mapping)

    result[valid.to_numpy()] = np.fromiter(
        (mapping[v] for v in values[valid]),
        dtype=np.int64,
        count=int(valid.sum())
    )
    return result

def add_pairs(target, left, right):
    valid = (left >= 0) & (right >= 0) & (left != right)
    if valid.any():
        codes = (left[valid].astype(np.uint64) << 32) | right[valid].astype(np.uint64)
        target.update(codes.tolist())

def decode_pairs(pairs):
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)

    values = np.fromiter(pairs, dtype=np.uint64)
    left = (values >> 32).astype(np.int64)
    right = (values & np.uint64(0xFFFFFFFF)).astype(np.int64)

    return torch.from_numpy(np.stack([left, right]))

columns = [
    "timestamp", "cdr_type", "src_number", "src_imei",
    "initial_src_cell", "current_src_cell",
    "dst_number", "dst_imei",
    "initial_dst_cell", "current_dst_cell",
    "duration", "size"
]

print("Reading CDR records in chunks...")
start = time.time()
total_rows = 0

reader = pd.read_csv(
    ROOT / "clean_cdr.csv",
    usecols=columns,
    dtype=str,
    chunksize=CHUNK,
    low_memory=False
)

for chunk_no, df in enumerate(reader, start=1):
    total_rows += len(df)

    src = df["src_number"].map(subscriber_map).fillna(-1).to_numpy(dtype=np.int64)
    dst = df["dst_number"].map(subscriber_map).fillna(-1).to_numpy(dtype=np.int64)

    valid_src = src >= 0
    valid_dst = dst >= 0

    duration = pd.to_numeric(df["duration"], errors="coerce").fillna(0).to_numpy()
    size = pd.to_numeric(df["size"], errors="coerce").fillna(0).to_numpy()

    np.add.at(outgoing, src[valid_src], 1)
    np.add.at(incoming, dst[valid_dst], 1)

    np.add.at(duration_sum, src[valid_src], duration[valid_src])
    np.add.at(duration_sq, src[valid_src], duration[valid_src] ** 2)
    np.add.at(size_sum, src[valid_src], size[valid_src])
    np.add.at(activity_count, src[valid_src], 1)

    cdr_type = df["cdr_type"].fillna("").str.strip()

    for j, typ in enumerate(types):
        mask = cdr_type.eq(typ).to_numpy() & valid_src
        np.add.at(type_counts[:, j], src[mask], 1)

    mask_out = cdr_type.eq("I-MOC").to_numpy() & valid_src
    mask_in = cdr_type.eq("I-MTC").to_numpy() & valid_dst

    np.add.at(intl_out, src[mask_out], 1)
    np.add.at(intl_in, dst[mask_in], 1)

    # Night activity: 00:00-05:59 and 22:00-23:59
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    hours = ts.dt.hour.to_numpy()
    night = ((hours <= 5) | (hours >= 22)) & ~pd.isna(hours) & valid_src
    np.add.at(night_events, src[night], 1)

    valid_ts = ts.notna().to_numpy() & valid_src
    if valid_ts.any():
        ns = ts.astype("int64").to_numpy()
        np.minimum.at(min_time, src[valid_ts], ns[valid_ts])
        np.maximum.at(max_time, src[valid_ts], ns[valid_ts])

    add_pairs(contact_pairs, src, dst)

    # Subscriber-device associations
    src_device = entity_ids(df["src_imei"], device_map, True)
    dst_device = entity_ids(df["dst_imei"], device_map, True)

    add_pairs(device_pairs, src, src_device)
    add_pairs(device_pairs, dst, dst_device)

    # Subscriber-cell associations
    for col, sub in [
        ("initial_src_cell", src),
        ("current_src_cell", src),
        ("initial_dst_cell", dst),
        ("current_dst_cell", dst)
    ]:
        cells = entity_ids(df[col], cell_map, True)
        add_pairs(cell_pairs, sub, cells)

    if chunk_no % 5 == 0:
        print(
            f"Chunks: {chunk_no} | Rows: {total_rows:,} | "
            f"Devices: {len(device_map):,} | Cells: {len(cell_map):,} | "
            f"Elapsed: {(time.time()-start)/60:.1f} min"
        )

print("\nCDR processing complete.")
print("Total CDR records:", f"{total_rows:,}")

# Unique contact counts
unique_out = np.zeros(N, dtype=np.float64)
unique_in = np.zeros(N, dtype=np.float64)

for code in contact_pairs:
    a = code >> 32
    b = code & 0xFFFFFFFF
    unique_out[a] += 1
    unique_in[b] += 1

unique_devices = np.zeros(N, dtype=np.float64)
unique_cells = np.zeros(N, dtype=np.float64)

for code in device_pairs:
    unique_devices[code >> 32] += 1

for code in cell_pairs:
    unique_cells[code >> 32] += 1

# Subscriber features
denom = np.maximum(activity_count, 1)

mean_duration = duration_sum / denom
variance = np.maximum(
    (duration_sq - (duration_sum ** 2) / denom) /
    np.maximum(activity_count - 1, 1),
    0
)
std_duration = np.sqrt(variance)
mean_size = size_sum / denom

active_span = np.zeros(N, dtype=np.float64)
valid_activity = min_time != np.iinfo(np.int64).max
active_span[valid_activity] = (
    max_time[valid_activity] - min_time[valid_activity]
) / 3.6e12

features = np.column_stack([
    outgoing, incoming, unique_out, unique_in,
    unique_devices, unique_cells,
    mean_duration, std_duration, mean_size,
    type_counts,
    intl_out, intl_in, night_events, active_span
])

# Log transform non-negative features to reduce scale skew
features = np.log1p(np.maximum(features, 0))
features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

# Node features for devices and cells
device_degree = np.zeros(len(device_map), dtype=np.float32)
cell_degree = np.zeros(len(cell_map), dtype=np.float32)

for code in device_pairs:
    device_degree[code & 0xFFFFFFFF] += 1

for code in cell_pairs:
    cell_degree[code & 0xFFFFFFFF] += 1

device_features = np.column_stack([
    np.ones(len(device_map)),
    np.log1p(device_degree),
    np.log1p(device_degree)
]).astype(np.float32)

cell_features = np.column_stack([
    np.ones(len(cell_map)),
    np.log1p(cell_degree),
    np.log1p(cell_degree)
]).astype(np.float32)

# Build heterogeneous graph
print("\nConstructing heterogeneous graph...")

graph = HeteroData()

graph["subscriber"].x = torch.tensor(features, dtype=torch.float32)
graph["subscriber"].y = torch.tensor(
    pd.to_numeric(labels["label"]).to_numpy(),
    dtype=torch.long
)

graph["device"].x = torch.tensor(device_features, dtype=torch.float32)
graph["cell"].x = torch.tensor(cell_features, dtype=torch.float32)

graph["subscriber", "communicates", "subscriber"].edge_index = decode_pairs(contact_pairs)

device_edges = decode_pairs(device_pairs)
graph["subscriber", "uses_device", "device"].edge_index = device_edges
graph["device", "used_by", "subscriber"].edge_index = device_edges.flip(0)

cell_edges = decode_pairs(cell_pairs)
graph["subscriber", "observed_at", "cell"].edge_index = cell_edges
graph["cell", "has_subscriber", "subscriber"].edge_index = cell_edges.flip(0)

# Save outputs
output_graph = ROOT / "hetero_graph.pt"
torch.save(graph, output_graph)

mappings = {
    "subscriber": subscriber_ids,
    "device": list(device_map.keys()),
    "cell": list(cell_map.keys())
}

with open(ROOT / "node_mappings.json", "w", encoding="utf-8") as f:
    json.dump(mappings, f)

print("\n========== FULL GRAPH SUMMARY ==========")
print("Subscribers:", graph["subscriber"].num_nodes)
print("Devices:", graph["device"].num_nodes)
print("Cells:", graph["cell"].num_nodes)
print("Subscriber features:", tuple(graph["subscriber"].x.shape))
print("Device features:", tuple(graph["device"].x.shape))
print("Cell features:", tuple(graph["cell"].x.shape))
print("Communication edges:", graph["subscriber", "communicates", "subscriber"].edge_index.shape[1])
print("Device edges:", device_edges.shape[1])
print("Cell edges:", cell_edges.shape[1])
print("Fraud labels:", int(graph["subscriber"].y.sum()))
print("Graph saved:", output_graph)
print("Mappings saved:", ROOT / "node_mappings.json")
print("Total elapsed minutes:", round((time.time()-start)/60, 2))
