"""Create deterministic, identity-disjoint subscriber splits for SIMGuard.

Run from the SIMGuard project root after 02_preprocess.py:
    python scripts/04_create_splits.py

Input:  data/processed/<strategy>/<scenario>/subscribers.csv
Output: data/splits/subscriber_splits.csv
        data/splits/split_summary.json

IMPORTANT: All rows for the same subscriber_id stay in ONE split across
all scenarios. This prevents direct target-identity overlap but does not,
by itself, guarantee temporal, feature, or neighborhood leakage-free graphs.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

FIELDNAMES = ["scenario_id", "subscriber_id", "label", "split"]


def read_subscribers(processed):
    files = sorted(processed.rglob("subscribers.csv"))
    if not files:
        raise FileNotFoundError(f"No subscribers.csv files in {processed}")

    rows = []
    ever_fraud = defaultdict(int)
    scenarios = set()
    seen = set()
    for file in files:
        scenario_id = file.parent.relative_to(processed).as_posix()
        scenarios.add(scenario_id)
        with file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            expected = {"scenario_id", "subscriber_id", "label"}
            if not expected.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing expected columns in {file}")
            for index, item in enumerate(reader, start=2):
                if item["scenario_id"] != scenario_id:
                    raise ValueError(f"{file}:{index}: scenario_id disagrees with path")
                number = item["subscriber_id"].strip()
                label_raw = item["label"].strip()
                if not number or label_raw not in {"0", "1"}:
                    raise ValueError(f"{file}:{index}: invalid subscriber_id or label")
                key = (scenario_id, number)
                if key in seen:
                    raise ValueError(f"Duplicate scenario/subscriber pair: {key}")
                seen.add(key)
                label = int(label_raw)
                rows.append((scenario_id, number, label))
                ever_fraud[number] = max(ever_fraud[number], label)
    return rows, ever_fraud, scenarios


def choose_splits(ever_fraud, seed, train_ratio, val_ratio):
    """Stratify subscriber identities by whether they are ever fraud-labelled."""
    assignments = {}
    for group in (0, 1):
        numbers = [number for number, label in ever_fraud.items() if label == group]
        # Cryptographic ordering is stable across Python versions/machines.
        numbers.sort(key=lambda number: (hashlib.sha256(
            f"SIMGuard|{seed}|{group}|{number}".encode("utf-8")
        ).hexdigest(), number))
        n = len(numbers)
        cutoff_train = round(n * train_ratio)
        cutoff_val = cutoff_train + round(n * val_ratio)
        for index, number in enumerate(numbers):
            assignments[number] = (
                "train" if index < cutoff_train else
                "val" if index < cutoff_val else "test"
            )
    return assignments


def main():
    parser = argparse.ArgumentParser(description="Create shared SIMGuard subscriber splits")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/splits"))
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing split manifest")
    args = parser.parse_args()
    if not (0 < args.train < 1 and 0 < args.val < 1 and args.train + args.val < 1):
        parser.error("Ratios must satisfy 0 < train, val and train + val < 1")

    rows, ever_fraud, scenarios = read_subscribers(args.processed)
    assignment = choose_splits(ever_fraud, args.seed, args.train, args.val)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "subscriber_splits.csv"
    json_path = args.output / "split_summary.json"
    if not args.overwrite and (csv_path.exists() or json_path.exists()):
        parser.error("Split files already exist. Review them before using --overwrite.")

    counts = {split: Counter() for split in ("train", "val", "test")}
    assigned_ids = {split: set() for split in counts}
    with csv_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
        writer.writeheader()
        for scenario_id, subscriber_id, label in sorted(rows):
            split = assignment[subscriber_id]
            writer.writerow(dict(scenario_id=scenario_id,
                                 subscriber_id=subscriber_id,
                                 label=label, split=split))
            counts[split]["rows"] += 1
            counts[split]["fraud_rows"] += label
            assigned_ids[split].add(subscriber_id)

    assert not (assigned_ids["train"] & assigned_ids["val"])
    assert not (assigned_ids["train"] & assigned_ids["test"])
    assert not (assigned_ids["val"] & assigned_ids["test"])
    for split in counts:
        counts[split]["unique_subscribers"] = len(assigned_ids[split])
        counts[split]["legitimate_rows"] = counts[split]["rows"] - counts[split]["fraud_rows"]

    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    summary = {
        "seed": args.seed,
        "train_ratio": args.train,
        "val_ratio": args.val,
        "test_ratio": round(1 - args.train - args.val, 10),
        "scenario_count": len(scenarios),
        "classification_rows": len(rows),
        "unique_subscribers": len(ever_fraud),
        "unique_ever_fraud_subscribers": sum(ever_fraud.values()),
        "identity_disjoint": True,
        "splits": {split: dict(counts[split]) for split in counts},
        "sha256_subscriber_splits_csv": digest,
        "caution": "ID-disjoint target splits; graph construction must still avoid future leakage."
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Processed {len(scenarios)} scenarios; {len(rows):,} subscriber-scenario rows")
    print(f"Unique subscriber IDs: {len(ever_fraud):,}")
    for split in ("train", "val", "test"):
        c = counts[split]
        print(f"{split.upper():5} : targets={c['rows']:,}, fraud={c['fraud_rows']:,}, "
              f"legitimate={c['legitimate_rows']:,}, unique IDs={c['unique_subscribers']:,}")
    print("Shared subscriber IDs between splits: 0")
    print(f"Split checksum SHA256: {digest}")
    print(f"Saved: {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
