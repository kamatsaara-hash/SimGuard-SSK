"""Train the SIMGuard graph-aware Temporal Transformer on generated NPZ files.

Run from SIMGuard project root:
  python training/train_tgt.py --limit 1 --epochs 2 --batch-size 64
  python training/train_tgt.py --epochs 12 --batch-size 64

Reads data/tgt/<strategy>/<scenario>/tgt_inputs.npz and
      data/splits/subscriber_splits.csv.
Uses train-only feature scaling, validation for selection and threshold,
and an untouched test set for final metrics. Scenarios can share background
traffic; this is SUBSCRIBER-DISJOINT, not SCENARIO-HELD-OUT evaluation.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Allow direct execution: `python training/train_tgt.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models.tgt import TemporalGraphTransformer  # noqa: E402

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Train graph-aware temporal Transformer for FraudZen")
    parser.add_argument("--inputs", type=Path, default=Path("data/tgt"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/subscriber_splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/TGT"))
    parser.add_argument("--limit", type=int, default=None, help="Smoke test on first N scenarios")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--context", choices=("split-isolated", "transductive"),
                        default="split-isolated", help="Neighbor graph access policy")
    args = parser.parse_args()
    for attr in ("epochs", "batch_size", "patience", "num_threads", "heads", "hidden_size", "layers"):
        if getattr(args, attr) < 1:
            parser.error(f"--{attr.replace('_','-')} must be >= 1")
    if args.hidden_size % args.heads:
        parser.error("--hidden-size must be divisible by --heads")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def scan_inputs(root, limit):
    paths = sorted(root.rglob("tgt_inputs.npz"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No tgt_inputs.npz files found in {root}")
    return paths


def read_split_assignments(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing shared split file: {path}")
    mapping = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"subscriber_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Split file lacks subscriber_id or split columns")
        for row in reader:
            sid, split = row["subscriber_id"], row["split"]
            if split not in SPLIT_CODE:
                raise ValueError(f"Invalid split {split!r} for {sid!r}")
            value = SPLIT_CODE[split]
            if sid in mapping and mapping[sid] != value:
                raise ValueError(f"Subscriber appears in multiple splits: {sid}")
            mapping[sid] = value
    if not mapping:
        raise ValueError("Empty split file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return mapping, digest


# Loading the compressed NPZ once per scenario is essential. Accessing
# npz["node_features"] for every training batch would decompress an entire
# scenario repeatedly and make full-dataset training unnecessarily slow.
SCENARIO_KEYS = (
    "node_features", "subscriber_ids", "target_indices", "target_labels",
    "target_splits", "edge_src", "edge_dst", "edge_window", "edge_count",
)


def load_scenario(path):
    with np.load(path, allow_pickle=False) as source:
        check_inputs(source, path)
        return {key: source[key] for key in SCENARIO_KEYS}


def check_inputs(d, path):
    required = {"node_features", "subscriber_ids", "target_indices", "target_labels",
                "target_splits", "edge_src", "edge_dst", "edge_window", "edge_count"}
    missing = required - set(d.files)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    x = d["node_features"]
    if x.ndim != 3 or x.shape[-1] != 17 or x.shape[1] < 1:
        raise ValueError(f"{path}: expected (subscribers, windows, 17) features")
    n = len(d["target_indices"])
    if not (len(d["target_labels"]) == len(d["target_splits"]) == n):
        raise ValueError(f"{path}: inconsistent target arrays")
    if np.any(x < 0) or not np.isfinite(x).all():
        raise ValueError(f"{path}: features must be finite and nonnegative")
    return x


def fit_train_scaler(paths):
    """Fit log1p feature mean/std using TRAIN targets and all their time windows."""
    sums = np.zeros(17, dtype=np.float64)
    sums_sq = np.zeros(17, dtype=np.float64)
    total_rows = 0
    class_counts = np.zeros(2, dtype=np.int64)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            x = check_inputs(data, path)
            mask = data["target_splits"] == SPLIT_CODE["train"]
            node_idx = data["target_indices"][mask]
            labels = data["target_labels"][mask]
            class_counts += np.bincount(labels.astype(np.int64), minlength=2)
            if len(node_idx) == 0:
                continue
            train_x = np.log1p(x[node_idx].astype(np.float64))
            sums += train_x.sum(axis=(0, 1))
            sums_sq += np.square(train_x).sum(axis=(0, 1))
            total_rows += train_x.shape[0] * train_x.shape[1]
    if total_rows == 0 or class_counts.min() == 0:
        raise ValueError("Training split must contain both classes")
    mean = sums / total_rows
    var = np.maximum(sums_sq / total_rows - mean * mean, 0.0)
    std = np.maximum(np.sqrt(var), 0.01)
    return mean.astype(np.float32), std.astype(np.float32), class_counts


def make_graph_context(d, global_splits, policy):
    """For each TARGET node, aggregate weighted same-hour neighbor features.

    Only target nodes get output contexts; destination-only nodes can contribute
    neighbor features but do not create additional supervised examples.
    In split-isolated mode, omit edges whose two endpoints have KNOWN DIFFERENT
    global splits. Unknown destination-only nodes may still provide context.
    This is not a guarantee of scenario/behavior independence, because each
    subscriber's own prepared features already reflect all observed CDRs.
    """
    x = d["node_features"]
    target_indices = d["target_indices"].astype(np.int64)
    n_nodes, steps, feats = x.shape
    n_targets = len(target_indices)
    node_to_target = np.full(n_nodes, -1, dtype=np.int32)
    node_to_target[target_indices] = np.arange(n_targets, dtype=np.int32)
    outgoing = np.zeros((n_targets, steps, feats), dtype=np.float32)
    incoming = np.zeros((n_targets, steps, feats), dtype=np.float32)
    outgoing_weight = np.zeros((n_targets, steps), dtype=np.float32)
    incoming_weight = np.zeros((n_targets, steps), dtype=np.float32)

    src = d["edge_src"].astype(np.int64)
    dst = d["edge_dst"].astype(np.int64)
    times = d["edge_window"].astype(np.int64)
    weights = d["edge_count"].astype(np.float32)
    if not (len(src) == len(dst) == len(times) == len(weights)):
        raise ValueError("Inconsistent edge arrays")
    if len(src) and (src.min() < 0 or dst.min() < 0 or src.max() >= n_nodes
                     or dst.max() >= n_nodes or times.min() < 0 or times.max() >= steps):
        raise ValueError("Out-of-range edge indices")
    if np.any(weights <= 0):
        raise ValueError("Nonpositive edge weight")

    eligible = np.ones(len(src), dtype=bool)
    if policy == "split-isolated" and len(src):
        subscriber_ids = d["subscriber_ids"].tolist()
        assigned = np.asarray([global_splits.get(sid, -1) for sid in subscriber_ids], dtype=np.int8)
        eligible = ((assigned[src] == assigned[dst])
                    | (assigned[src] < 0) | (assigned[dst] < 0))
        # Confirm the split map agrees with the scenario's supervised labels.
        mapped_targets = assigned[target_indices]
        if not np.array_equal(mapped_targets, d["target_splits"]):
            raise ValueError("Scenario target split conflicts with global identity assignment")

    # Process one time window at a time to avoid materializing an (edges,F) tensor
    # for every event in the full week.
    for w in range(steps):
        idx = np.flatnonzero(eligible & (times == w))
        if len(idx) == 0:
            continue
        s = src[idx]
        t = dst[idx]
        edge_weights = weights[idx]

        out_slots = node_to_target[s]
        out_valid = out_slots >= 0
        if out_valid.any():
            slot = out_slots[out_valid]
            np.add.at(outgoing[:, w, :], slot,
                      x[t[out_valid], w, :] * edge_weights[out_valid, None])
            np.add.at(outgoing_weight[:, w], slot, edge_weights[out_valid])

        in_slots = node_to_target[t]
        in_valid = in_slots >= 0
        if in_valid.any():
            slot = in_slots[in_valid]
            np.add.at(incoming[:, w, :], slot,
                      x[s[in_valid], w, :] * edge_weights[in_valid, None])
            np.add.at(incoming_weight[:, w], slot, edge_weights[in_valid])

    np.divide(outgoing, np.maximum(outgoing_weight[..., None], 1), out=outgoing)
    np.divide(incoming, np.maximum(incoming_weight[..., None], 1), out=incoming)
    return outgoing, incoming, outgoing_weight > 0, incoming_weight > 0


def prepare_batch(d, context, target_slots, mean, std, device):
    node_ids = d["target_indices"][target_slots]
    raw_self = d["node_features"][node_ids]
    out, inc, out_present, in_present = context
    raw = np.stack([raw_self, out[target_slots], inc[target_slots]], axis=2)
    norm = (np.log1p(raw) - mean[None, None, None, :]) / std[None, None, None, :]
    available = np.stack([out_present[target_slots], in_present[target_slots]], axis=-1)
    tokens = torch.from_numpy(np.ascontiguousarray(norm, dtype=np.float32)).to(device)
    masks = torch.from_numpy(np.ascontiguousarray(available)).to(device)
    labels = torch.from_numpy(d["target_labels"][target_slots].astype(np.float32)).to(device)
    return tokens, masks, labels


def train_epoch(model, paths, assignments, policy, optimizer, loss_fn, batch_size, mean, std, device, rng):
    model.train()
    order = rng.permutation(len(paths))
    running_loss, total_rows = 0.0, 0
    for path_index in order:
        path = paths[int(path_index)]
        d = load_scenario(path)
        slots = np.flatnonzero(d["target_splits"] == SPLIT_CODE["train"])
        if len(slots) == 0:
            continue
        context = make_graph_context(d, assignments, policy)
        slots = rng.permutation(slots)
        for start in range(0, len(slots), batch_size):
            batch = slots[start:start + batch_size]
            tokens, mask, y = prepare_batch(d, context, batch, mean, std, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(batch)
            total_rows += len(batch)
    return running_loss / max(1, total_rows)


@torch.no_grad()
def collect_predictions(model, paths, assignments, policy, split, batch_size, mean, std, device):
    model.eval()
    all_y, all_probs = [], []
    for path in paths:
        d = load_scenario(path)
        slots = np.flatnonzero(d["target_splits"] == SPLIT_CODE[split])
        if len(slots) == 0:
            continue
        context = make_graph_context(d, assignments, policy)
        for start in range(0, len(slots), batch_size):
            batch = slots[start:start + batch_size]
            tokens, mask, y = prepare_batch(d, context, batch, mean, std, device)
            logits = model(tokens, mask)
            prob = torch.sigmoid(logits).cpu().numpy()
            all_y.append(y.cpu().numpy().astype(np.int8))
            all_probs.append(prob)
    if not all_y:
        raise ValueError(f"No examples for {split} split")
    return np.concatenate(all_y), np.concatenate(all_probs)


def average_precision(y, scores):
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = y[order].astype(np.int64)
    tp = np.cumsum(ranked)
    precision = tp / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def best_f1_threshold(y, probs):
    """Choose threshold by maximum VALIDATION F1 (not by TEST F1)."""
    positives = int(y.sum())
    if positives == 0:
        return 0.5
    order = np.argsort(-probs, kind="stable")
    ranked = y[order].astype(np.int64)
    sorted_prob = probs[order]
    tp = np.cumsum(ranked)
    n_predicted = np.arange(1, len(ranked) + 1)
    f1 = 2.0 * tp / (n_predicted + positives)
    # Only evaluate thresholds AFTER all tied probabilities.
    ends = np.flatnonzero(np.r_[sorted_prob[:-1] != sorted_prob[1:], True])
    best = ends[np.argmax(f1[ends])]
    return float(sorted_prob[best])


def metrics(y, probs, threshold):
    pred = probs >= threshold
    y_bool = y.astype(bool)
    tp = int(np.sum(y_bool & pred))
    fp = int(np.sum(~y_bool & pred))
    fn = int(np.sum(y_bool & ~pred))
    tn = int(np.sum(~y_bool & ~pred))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "rows": len(y), "fraud_rows": int(y.sum()), "threshold": threshold,
        "precision": precision, "recall": recall, "f1": f1,
        "average_precision_PR_AUC": average_precision(y, probs),
        "false_positive_rate": fp / max(1, fp + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu")
    paths = scan_inputs(args.inputs, args.limit)
    assignments, split_hash = read_split_assignments(args.split_file)
    print(f"Device: {device}; scenarios: {len(paths)}; graph context: {args.context}", flush=True)
    print("Fitting log1p mean/std on TRAIN targets only ...", flush=True)
    mean, std, class_counts = fit_train_scaler(paths)
    positive_weight = min(float(class_counts[0] / class_counts[1]), 50.0)
    print(f"Training rows: legitimate={class_counts[0]:,}, fraud={class_counts[1]:,}; "
          f"positive loss weight={positive_weight:.2f}", flush=True)

    model = TemporalGraphTransformer(
        feature_dim=17, hidden_dim=args.hidden_size, heads=args.heads,
        layers=args.layers, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    rng = np.random.default_rng(args.seed)
    best_ap, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, paths, assignments, args.context, optimizer, loss_fn,
                           args.batch_size, mean, std, device, rng)
        y_val, p_val = collect_predictions(model, paths, assignments, args.context, "val",
                                           args.batch_size, mean, std, device)
        ap = average_precision(y_val, p_val)
        val_threshold = best_f1_threshold(y_val, p_val)
        val_score = metrics(y_val, p_val, val_threshold)
        history.append({"epoch": epoch, "training_loss": loss, "val_AP": ap,
                        "val_F1_tuned": val_score["f1"]})
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={loss:.4f} | "
              f"val_AP={ap:.4f} | val_F1={val_score['f1']:.4f}", flush=True)
        if ap is not None and ap > best_ap + 1e-6:
            best_ap, best_epoch, stale = ap, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {stale} epochs without validation AP improvement.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Could not select model: validation has no positive examples")
    model.load_state_dict(best_state)
    y_val, p_val = collect_predictions(model, paths, assignments, args.context, "val",
                                       args.batch_size, mean, std, device)
    threshold = best_f1_threshold(y_val, p_val)
    val_results = metrics(y_val, p_val, threshold)
    # The test set is not used for gradient updates, hyperparameter/epoch
    # selection, scaling, or threshold selection.
    y_test, p_test = collect_predictions(model, paths, assignments, args.context, "test",
                                         args.batch_size, mean, std, device)
    test_results = metrics(y_test, p_test, threshold)

    out = args.output / (f"smoke_{args.limit}" if args.limit else "full")
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "feature_dim": 17, "hidden_dim": args.hidden_size,
        "heads": args.heads, "layers": args.layers, "dropout": args.dropout,
    }
    torch.save({
        "model_state": model.state_dict(), "model_config": settings,
        "train_feature_mean_log1p": mean.tolist(),
        "train_feature_std_log1p": std.tolist(),
        "threshold_selected_on_val": threshold,
        "best_epoch": best_epoch,
        "split_sha256": split_hash,
        "context_mode": args.context,
    }, out / "model.pt")
    report = {
        "evaluation_type": "subscriber-identity-disjoint, repeated scenario-row targets",
        "scenario_count": len(paths), "smoke_test": args.limit is not None,
        "split_sha256": split_hash,
        "context_mode": args.context,
        "model_config": settings,
        "best_epoch": best_epoch, "training_class_counts": class_counts.tolist(),
        "loss_positive_weight": positive_weight,
        "validation": val_results, "test": test_results,
        "training_history": history,
        "cautions": [
            "Graph neighbor inputs are weighted per-hour summaries, not edge-level attention.",
            "Split-isolated edges exclude cross-split neighbors with known identities, but own "
            "subscriber features still include all observable CDRs. This is not scenario independence.",
            "Repeated subscriber-scenario rows are not independent observations.",
            "All predictions are end-of-observation, not early or streaming detection.",
            "Do not use single-scenario smoke results as research-paper accuracy.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Best epoch: {best_epoch}; threshold chosen from validation: {threshold:.4f}")
    print(f"VAL  precision={val_results['precision']:.4f} recall={val_results['recall']:.4f} "
          f"F1={val_results['f1']:.4f} AP={val_results['average_precision_PR_AUC']:.4f}")
    print(f"TEST precision={test_results['precision']:.4f} recall={test_results['recall']:.4f} "
          f"F1={test_results['f1']:.4f} AP={test_results['average_precision_PR_AUC']:.4f}")
    print(f"Saved checkpoint and metrics to: {out}")


if __name__ == "__main__":
    main()
