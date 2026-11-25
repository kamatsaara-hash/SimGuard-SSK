from pathlib import Path
import re
import pandas as pd

ROOT = Path("data/raw/Fraudzen-dataset/Fraudzen-dataset")
SCENARIO = ROOT / "FraudZen_Cdrs/all_advanced/advanced_3%_50sims"
OUTPUT = Path("data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

def normalize_number(value):
    value = str(value).strip()
    if value.lower() in {"unknown", "nan", "none", ""}:
        return None
    digits = re.sub(r"\D", "", value)
    return digits if digits else None

def parse_cdr_file(path):
    records = []
    malformed = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.strip().split()

            if not p or p[0].lower() == "timestamp":
                continue

            if len(p) < 12:
                malformed += 1
                continue

            try:
                timestamp = pd.to_datetime(
                    p[0] + " " + p[1],
                    dayfirst=True,
                    errors="coerce"
                )

                if pd.isna(timestamp):
                    malformed += 1
                    continue

                records.append({
                    "timestamp": timestamp,
                    "cdr_type": p[2],
                    "src_number": normalize_number(p[3]),
                    "src_imei": p[4] if p[4].lower() != "unknown" else None,
                    "initial_src_cell": p[5],
                    "current_src_cell": p[6],
                    "dst_number": normalize_number(p[7]),
                    "dst_imei": p[8] if p[8].lower() != "unknown" else None,
                    "initial_dst_cell": p[9],
                    "current_dst_cell": p[10],
                    "duration": pd.to_numeric(p[11], errors="coerce"),
                    "size": pd.to_numeric(p[12], errors="coerce")
                             if len(p) > 12 else 0
                })
            except (ValueError, IndexError):
                malformed += 1

    return records, malformed

print("Reading CDR files...")

all_records = []
total_malformed = 0

for filename in ["Op_1_CDRTrace.txt", "International_CDRTrace.txt"]:
    path = SCENARIO / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing CDR file: {path}")

    records, malformed = parse_cdr_file(path)
    all_records.extend(records)
    total_malformed += malformed

    print(f"{filename}: {len(records):,} records, {malformed} malformed")

cdr = pd.DataFrame(all_records)

if cdr.empty:
    raise RuntimeError("No CDR records parsed.")

# Remove records that cannot be linked to either subscriber.
cdr = cdr.dropna(subset=["src_number", "dst_number"]).copy()

# Clean cell identifiers.
cell_columns = [
    "initial_src_cell", "current_src_cell",
    "initial_dst_cell", "current_dst_cell"
]

for col in cell_columns:
    cdr[col] = cdr[col].astype(str).str.strip()

# Load known fraud numbers.
fraud_file = SCENARIO / "fraudulentUsers.txt"

if not fraud_file.exists():
    raise FileNotFoundError(f"Missing fraud labels: {fraud_file}")

fraud_text = fraud_file.read_text(encoding="utf-8", errors="replace")
fraud_ids = set(normalize_number(x) for x in re.findall(r"\+?\d+", fraud_text))
fraud_ids.discard(None)

subscribers = sorted(
    set(cdr["src_number"].dropna()) |
    set(cdr["dst_number"].dropna())
)

matched_fraud = fraud_ids.intersection(subscribers)

if len(matched_fraud) != len(fraud_ids):
    print("WARNING: Some fraud IDs did not match CDR subscribers.")

labels = pd.DataFrame({
    "subscriber_id": subscribers,
    "label": [int(s in fraud_ids) for s in subscribers],
    "label_status": [
        "known_fraud" if s in fraud_ids else "unlabelled_assumed_negative"
        for s in subscribers
    ]
})

cdr.to_csv(OUTPUT / "clean_cdr.csv", index=False)
labels.to_csv(OUTPUT / "subscriber_labels.csv", index=False)

print("\n========== PREPROCESSING SUMMARY ==========")
print("CDR records:", f"{len(cdr):,}")
print("Malformed records:", total_malformed)
print("Unique subscribers:", f"{len(subscribers):,}")
print("Known fraud IDs:", len(fraud_ids))
print("Fraud IDs matched:", len(matched_fraud))
print("Fraud subscribers:", int(labels["label"].sum()))
print("Provisional negative subscribers:", int((labels["label"] == 0).sum()))
print("\nCDR types:")
print(cdr["cdr_type"].value_counts().to_string())
print("\nSaved:")
print(OUTPUT / "clean_cdr.csv")
print(OUTPUT / "subscriber_labels.csv")
