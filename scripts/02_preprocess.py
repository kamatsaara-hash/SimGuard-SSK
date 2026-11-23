"""Preprocess official FraudZen operator CDRs without loading all scenarios into RAM.

Usage (from SIMGuard root):
  python scripts/02_preprocess.py --dataset data/raw/Fraudzen-dataset/Fraudzen-dataset --limit 1
  python scripts/02_preprocess.py --dataset data/raw/Fraudzen-dataset/Fraudzen-dataset
"""
import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

COLUMNS = ["timestamp", "CDR_type", "src_number", "src_imei", "initial_src_cellId", "current_src_cellId", "dst_number", "dst_imei", "initial_dst_cellId", "current_dst_cellId", "duration", "size"]
EVENT_FIELDS = ["scenario_id", *COLUMNS]
SUBSCRIBER_FIELDS = ["scenario_id", "subscriber_id", "label", "outgoing_events", "incoming_events"]


def parse_record(line, line_number, file_path):
    parts = line.split()
    if len(parts) not in (12, 13):
        raise ValueError(f"{file_path}:{line_number}: expected 12 or 13 tokens, got {len(parts)}")
    # Date and time are separate whitespace tokens; size can be absent.
    values = [parts[0] + " " + parts[1], *parts[2:]]
    if len(values) == 11:
        values.append("")
    record = dict(zip(COLUMNS, values))
    try:
        timestamp = datetime.strptime(record["timestamp"], "%d-%m-%Y %H:%M:%S")
    except ValueError as exc:
        raise ValueError(f"{file_path}:{line_number}: invalid timestamp: {record['timestamp']}") from exc
    record["timestamp"] = timestamp.isoformat(sep=" ")
    return record


def preprocess_scenario(scenario_dir, cdr_root, output_root):
    scenario_id = scenario_dir.relative_to(cdr_root).as_posix()
    cdr_path = scenario_dir / "Op_1_CDRTrace.txt"
    labels_path = scenario_dir / "fraudulentUsers.txt"
    if not cdr_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"Missing required files in {scenario_dir}")

    fraud_ids = {line.strip() for line in labels_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    out_dir = output_root / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)
    event_file = out_dir / "events.csv"
    subscribers = Counter()
    incoming = Counter()
    row_count = 0
    types = Counter()
    with cdr_path.open("r", encoding="utf-8-sig") as source, event_file.open("w", newline="", encoding="utf-8") as destination:
        header = source.readline().strip().split()
        if header != COLUMNS:
            raise ValueError(f"Unexpected CDR header in {cdr_path}: {header}")
        writer = csv.DictWriter(destination, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for line_number, line in enumerate(source, start=2):
            if not line.strip():
                continue
            record = parse_record(line, line_number, cdr_path)
            writer.writerow({"scenario_id": scenario_id, **record})
            subscribers[record["src_number"]] += 1
            incoming[record["dst_number"]] += 1
            types[record["CDR_type"]] += 1
            row_count += 1

    # Only src_number subscribers are used as classification candidates initially;
    # destination-only identifiers may include external parties, not operator subscribers.
    observed = set(subscribers)
    missing_fraud = fraud_ids - observed
    with (out_dir / "subscribers.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=SUBSCRIBER_FIELDS)
        writer.writeheader()
        for subscriber_id in sorted(observed):
            writer.writerow({"scenario_id": scenario_id, "subscriber_id": subscriber_id,
                             "label": int(subscriber_id in fraud_ids),
                             "outgoing_events": subscribers[subscriber_id],
                             "incoming_events": incoming[subscriber_id]})
    print(f"{scenario_id}: events={row_count:,}, observed_src={len(observed):,}, "
          f"fraud_labels={len(fraud_ids)}, observed_fraud={len(fraud_ids & observed)}, "
          f"unobserved_fraud={len(missing_fraud)}, types={dict(types)}")
    if missing_fraud:
        print("  WARNING: Some labeled fraud IDs do not occur as src_number; investigate before training.")
    return row_count


def main():
    parser = argparse.ArgumentParser(description="Preprocess official FraudZen CDR scenarios")
    parser.add_argument("--dataset", type=Path, required=True, help="Directory containing FraudZen_Cdrs")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N scenarios for testing")
    args = parser.parse_args()
    root = args.dataset / "FraudZen_Cdrs"
    if not root.is_dir():
        parser.error(f"Cannot find {root}. Check --dataset path.")
    scenarios = sorted(p.parent for p in root.rglob("Op_1_CDRTrace.txt") if p.is_file())
    if not scenarios:
        parser.error("No operator CDR files found")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        scenarios = scenarios[:args.limit]
    total = 0
    for scenario in scenarios:
        total += preprocess_scenario(scenario, root, args.output)
    print(f"Completed {len(scenarios)} scenarios; {total:,} CDR events. Output: {args.output}")


if __name__ == "__main__":
    main()
