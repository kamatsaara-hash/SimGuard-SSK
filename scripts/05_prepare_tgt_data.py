"""Prepare SIMGuard TGT inputs from the audited FraudZen scenario graphs.

Run in the SIMGuard project root after 02_preprocess.py, 04_create_splits.py,
03_build_graph.py and audit_graphs.py:

    python scripts/05_prepare_tgt_data.py --limit 1
    python scripts/05_prepare_tgt_data.py

For each scenario creates:
    data/tgt/<group>/<scenario>/tgt_inputs.npz
    data/tgt/<group>/<scenario>/metadata.json

The compressed NumPy file contains a temporal subscriber graph with observable
features; labels and split IDs are stored separately from the features.
Node features and graph edges are chronological, binned into fixed-width windows.
This is *input preparation*, not a fitted TGT or a leakage-proof training loop.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# These features have no fraud-label, split, raw phone/IMEI/cell identifier,
# or scenario-name features. The raw identifiers are only used to connect nodes.
FEATURE_NAMES = [
    "source_events",                 # All CDR rows for which this ID is src_number
    "source_DATA",                   # Source CDR event types, as written in the data
    "source_MOC",
    "source_I-MOC",
    "source_MTC",
    "source_MO-SMS",
    "source_other_type",
    "destination_nonDATA_events",    # Times this ID is dst of a different source
    "source_nonDATA_duration_sum_s", # Excludes DATA
    "source_nonDATA_duration_mean_s",# Excludes DATA
    "source_size_sum",               # Missing size is counted as 0
    "unique_nonDATA_destinations",   # Unique destination IDs for non-DATA events
    "unique_nonDATA_sources",        # Unique incoming source IDs
    "unique_source_devices",         # Number of source IMEIs, not raw IMEI values
    "unique_source_cells",           # Initial and current source cells
    "source_device_changes",         # Across consecutive source observations
    "source_cell_changes",           # Across consecutive source observations
]
TYPE_FEATURE = {"DATA": 1, "MOC": 2, "I-MOC": 3, "MTC": 4, "MO-SMS": 5}
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
REQUIRED_EVENTS = {
    "scenario_id", "timestamp", "CDR_type", "src_number", "dst_number",
    "src_imei", "initial_src_cellId", "current_src_cellId", "duration", "size",
}
MISSING_VALUES = {"", "none", "nan", "null", "-", "unknown"}
EPOCH = datetime(1970, 1, 1)


def valid_id(value):
    value = (value or "").strip()
    return value if value.lower() not in MISSING_VALUES else None


def parse_time(value, file_path, row_number):
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{file_path}:{row_number}: invalid timestamp {value!r}") from exc


def nonnegative_number(value, file_path, row_number, column):
    raw = (value or "").strip()
    if not raw:
        return 0.0
    try:
        result = float(raw)
    except ValueError as exc:
        raise ValueError(f"{file_path}:{row_number}: invalid {column}={raw!r}") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{file_path}:{row_number}: non-finite/negative {column}={raw!r}")
    return result


def read_subscriber_nodes(nodes_path):
    ids, graph_ids = [], []
    with nodes_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"node_id", "node_type", "raw_id"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Malformed nodes.csv: {nodes_path}")
        for row in reader:
            if row["node_type"] == "subscriber":
                ids.append(row["raw_id"])
                graph_ids.append(row["node_id"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate subscriber nodes in {nodes_path}")
    if not ids:
        raise ValueError(f"No subscriber nodes in {nodes_path}")
    return ids, graph_ids, {number: i for i, number in enumerate(ids)}


def read_targets(targets_path, subscriber_to_idx, graph_ids):
    indices, labels, splits = [], [], []
    seen = set()
    with targets_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"node_id", "subscriber_id", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Malformed targets.csv: {targets_path}")
        for row in reader:
            number = row["subscriber_id"]
            if number not in subscriber_to_idx:
                raise ValueError(f"Target {number} does not exist as a subscriber node")
            index = subscriber_to_idx[number]
            if index in seen:
                raise ValueError(f"Duplicate classification target: {number}")
            seen.add(index)
            if graph_ids[index] != row["node_id"]:
                raise ValueError(f"Mismatched graph node ID for {number}")
            if row["label"] not in ("0", "1") or row["split"] not in SPLIT_CODES:
                raise ValueError(f"Invalid target label/split: {row}")
            indices.append(index)
            labels.append(int(row["label"]))
            splits.append(SPLIT_CODES[row["split"]])
    return (np.asarray(indices, dtype=np.int32),
            np.asarray(labels, dtype=np.int8),
            np.asarray(splits, dtype=np.int8))


def inspect_time_range(events_path, scenario):
    earliest, latest, previous = None, None, None
    total = 0
    with events_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not REQUIRED_EVENTS.issubset(reader.fieldnames or []):
            raise ValueError(f"{events_path}: missing columns {sorted(REQUIRED_EVENTS - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, start=2):
            if row["scenario_id"] != scenario:
                raise ValueError(f"{events_path}:{line}: scenario does not match its directory")
            t = parse_time(row["timestamp"], events_path, line)
            # The graph builder is event-order-sensitive. Reject unsorted event files
            # rather than accidentally counting later events before earlier ones.
            if previous is not None and t < previous:
                raise ValueError(f"{events_path}:{line}: CDRs not chronological; sort by timestamp before TGT preparation")
            previous = t
            if earliest is None:
                earliest = t
            latest = t
            total += 1
    if not total:
        raise ValueError(f"No events in {events_path}")
    if earliest.utcoffset() is not None:
        raise ValueError("FraudZen timestamp should be timezone-naive; avoid timezone-dependent conversions")
    return earliest, latest, total


def prepare_one(events_path, processed, graphs, output, window_seconds):
    scenario = events_path.parent.relative_to(processed).as_posix()
    graph_path = graphs / scenario
    nodes_path, targets_path, summary_path = (graph_path / "nodes.csv",
                                           graph_path / "targets.csv",
                                           graph_path / "summary.json")
    for file in (nodes_path, targets_path, summary_path):
        if not file.is_file():
            raise FileNotFoundError(f"Missing audited graph input: {file}")
    graph_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if graph_summary.get("scenario_id") != scenario:
        raise ValueError(f"Scenario ID differs between events and graph summary for {scenario}")

    subscriber_ids, graph_ids, node_index = read_subscriber_nodes(nodes_path)
    target_indices, labels, splits = read_targets(targets_path, node_index, graph_ids)
    if len(target_indices) != graph_summary.get("targets") or int(labels.sum()) != graph_summary.get("fraud_targets"):
        raise ValueError(f"Graph target summary does not match targets.csv for {scenario}")

    start, end, n_events = inspect_time_range(events_path, scenario)
    if n_events != graph_summary.get("events"):
        raise ValueError(f"Graph summary event count differs for {scenario}")
    # Floor to a clean fixed window boundary without assuming a real timezone.
    first_seconds = int((start - EPOCH).total_seconds())
    last_seconds = int((end - EPOCH).total_seconds())
    origin_sec = (first_seconds // window_seconds) * window_seconds
    num_windows = (last_seconds - origin_sec) // window_seconds + 1
    if num_windows > 10_000:
        raise ValueError(f"Too many windows ({num_windows}) for {scenario}; choose a larger window")
    n_nodes = len(subscriber_ids)
    x = np.zeros((n_nodes, num_windows, len(FEATURE_NAMES)), dtype=np.float32)
    active = np.zeros((n_nodes, num_windows), dtype=bool)

    unique_outgoing = defaultdict(set)
    unique_incoming = defaultdict(set)
    unique_devices = defaultdict(set)
    unique_cells = defaultdict(set)
    previous_device = {}
    previous_cell = {}
    # Each key is (window_index, src_index, dst_index): number of observed
    # non-DATA CDRs between them. Self-records are not considered call edges.
    temporal_edges = Counter()
    source_counts = Counter()

    with events_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for line, row in enumerate(reader, start=2):
            src, dst = row["src_number"], row["dst_number"]
            if src not in node_index or dst not in node_index:
                raise ValueError(f"{events_path}:{line}: subscriber missing from graph nodes")
            s, d = node_index[src], node_index[dst]
            when = parse_time(row["timestamp"], events_path, line)
            time_seconds = int((when - EPOCH).total_seconds())
            w = (time_seconds - origin_sec) // window_seconds
            if not 0 <= w < num_windows:
                raise ValueError(f"Window outside timeline for {scenario} at row {line}")
            activity = x[s, w]
            activity[0] += 1
            cdr_type = row["CDR_type"]
            activity[TYPE_FEATURE.get(cdr_type, 6)] += 1
            activity[10] += nonnegative_number(row["size"], events_path, line, "size")
            active[s, w] = True
            source_counts[src] += 1

            device = valid_id(row["src_imei"])
            if device:
                unique_devices[(s, w)].add(device)
                if s in previous_device and device != previous_device[s]:
                    activity[15] += 1
                previous_device[s] = device
            cell = valid_id(row["current_src_cellId"])
            initial_cell = valid_id(row["initial_src_cellId"])
            if initial_cell:
                unique_cells[(s, w)].add(initial_cell)
            if cell:
                unique_cells[(s, w)].add(cell)
                if s in previous_cell and cell != previous_cell[s]:
                    activity[16] += 1
                previous_cell[s] = cell

            if cdr_type != "DATA":
                activity[8] += nonnegative_number(row["duration"], events_path, line, "duration")
                if src != dst:
                    active[d, w] = True
                    x[d, w, 7] += 1
                    unique_outgoing[(s, w)].add(dst)
                    unique_incoming[(d, w)].add(src)
                    temporal_edges[(w, s, d)] += 1

    # Mean duration is computed over source non-DATA event counts, not all DATA.
    denominator = x[:, :, 0] - x[:, :, 1]
    np.divide(x[:, :, 8], denominator, out=x[:, :, 9], where=denominator > 0)
    for bucket, values in unique_outgoing.items():
        x[bucket[0], bucket[1], 11] = len(values)
    for bucket, values in unique_incoming.items():
        x[bucket[0], bucket[1], 12] = len(values)
    for bucket, values in unique_devices.items():
        x[bucket[0], bucket[1], 13] = len(values)
    for bucket, values in unique_cells.items():
        x[bucket[0], bucket[1], 14] = len(values)

    # Ensure the graph's supervised targets correspond to observed source IDs.
    missing_sources = [subscriber_ids[i] for i in target_indices if subscriber_ids[i] not in source_counts]
    if missing_sources:
        raise ValueError(f"{scenario}: classification target without source activity, first={missing_sources[:3]}")

    sorted_edges = sorted(temporal_edges.items())
    edge_window = np.fromiter((key[0] for key, _ in sorted_edges), dtype=np.int32, count=len(sorted_edges))
    edge_src = np.fromiter((key[1] for key, _ in sorted_edges), dtype=np.int32, count=len(sorted_edges))
    edge_dst = np.fromiter((key[2] for key, _ in sorted_edges), dtype=np.int32, count=len(sorted_edges))
    edge_count = np.fromiter((count for _, count in sorted_edges), dtype=np.int32, count=len(sorted_edges))
    window_starts = origin_sec + np.arange(num_windows, dtype=np.int64) * window_seconds

    outdir = output / scenario
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "tgt_inputs.npz"
    np.savez_compressed(
        outfile,
        node_features=x,
        active_mask=active,
        feature_names=np.asarray(FEATURE_NAMES),
        subscriber_ids=np.asarray(subscriber_ids),
        graph_subscriber_node_ids=np.asarray(graph_ids),
        target_indices=target_indices,
        target_labels=labels,
        target_splits=splits,
        split_names=np.asarray(["train", "val", "test"]),
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_window=edge_window,
        edge_count=edge_count,
        window_starts=window_starts,
        window_seconds=np.asarray(window_seconds, dtype=np.int32),
    )
    info = {
        "scenario_id": scenario,
        "source_events": n_events,
        "subscriber_nodes": n_nodes,
        "classification_targets": len(target_indices),
        "fraud_targets": int(labels.sum()),
        "target_split_counts": {name: int(np.count_nonzero(splits == code)) for name, code in SPLIT_CODES.items()},
        "time_start": start.isoformat(sep=" "),
        "time_end": end.isoformat(sep=" "),
        "window_seconds": window_seconds,
        "num_windows": num_windows,
        "temporal_interaction_edges": len(sorted_edges),
        "source_nonDATA_interaction_events": int(sum(temporal_edges.values())),
        "feature_names": FEATURE_NAMES,
        "features_shape": list(x.shape),
        "npz_file": outfile.name,
        "notes": [
            "Source event types are used as recorded; CDR_type MTC is not interpreted as a directed outgoing phone call.",
            "Only non-DATA CDRs between distinct src/dst IDs produce subscriber interaction edges.",
            "Destination-only nodes provide context but have no supervised label.",
            "Raw phone, device and cell IDs are metadata, not numeric model features.",
            "Window features summarize events available by END of each window.",
            "Window timestamps have no declared timezone in the source dataset.",
            "Training must enforce causal temporal attention, proper split isolation, and train-only scaling.",
            "Identity-disjoint targets do not imply held-out scenario independence.",
        ],
    }
    (outdir / "metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"{scenario}: events={n_events:,}, subscribers={n_nodes:,}, "
          f"windows={num_windows}, interactions={len(sorted_edges):,}, "
          f"targets={len(target_indices):,}, fraud={int(labels.sum()):,}")
    return info


def main():
    parser = argparse.ArgumentParser(description="Create chronological SIMGuard TGT graph inputs")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--graphs", type=Path, default=Path("data/graphs"))
    parser.add_argument("--output", type=Path, default=Path("data/tgt"))
    parser.add_argument("--window-minutes", type=int, default=60,
                        help="Window length in minutes; default 60")
    parser.add_argument("--limit", type=int, default=None, help="Test on first N scenarios")
    args = parser.parse_args()
    if args.window_minutes < 1:
        parser.error("--window-minutes must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    paths = sorted(args.processed.rglob("events.csv"))
    if not paths:
        parser.error(f"No events.csv found under {args.processed}")
    if args.limit is not None:
        paths = paths[:args.limit]
    reports = [prepare_one(p, args.processed, args.graphs, args.output,
                           args.window_minutes * 60) for p in paths]
    print(f"Completed {len(reports)} scenarios; "
          f"{sum(r['source_events'] for r in reports):,} events; "
          f"{sum(r['classification_targets'] for r in reports):,} supervised targets. "
          f"Output: {args.output}")


if __name__ == "__main__":
    main()
