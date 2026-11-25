
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

SCENARIO = (
    RAW / "Fraudzen-dataset" / "Fraudzen-dataset"
    / "FraudZen_Cdrs" / "all_advanced" / "advanced_3%_50sims"
)

# Read fraud labels, preserving subscriber IDs as strings.
label_file = SCENARIO / "fraudulentUsers.txt"

fraud_ids = set()

with open(label_file, "r", encoding="utf-8") as f:
    for line in f:
        number = line.strip().lstrip("+")
        if number:
            fraud_ids.add(number)

print("FRAUD LABEL COUNT:", len(fraud_ids))

# Stream CDR files without loading the entire dataset into memory.
cdr_files = [
    SCENARIO / "Op_1_CDRTrace.txt",
    SCENARIO / "International_CDRTrace.txt"
]

all_subscribers = set()
source_subscribers = set()
destination_subscribers = set()

total_records = 0
malformed_records = 0

for file in cdr_files:
    print("\nProcessing:", file.name)

    with open(file, "r", encoding="utf-8") as f:
        header = next(f, None)

        for line in f:
            parts = line.split()

            # Standard CDR layout:
            # date time type src_number ... dst_number ...
            if len(parts) < 8:
                malformed_records += 1
                continue

            src = parts[3].lstrip("+")
            dst = parts[7].lstrip("+")

            if not src.isdigit() or not dst.isdigit():
                malformed_records += 1
                continue

            source_subscribers.add(src)
            destination_subscribers.add(dst)
            all_subscribers.add(src)
            all_subscribers.add(dst)

            total_records += 1

print("\n" + "=" * 50)
print("LABEL VERIFICATION RESULTS")
print("=" * 50)

matched = fraud_ids.intersection(all_subscribers)
unmatched = fraud_ids - all_subscribers

print("Total CDR records:", total_records)
print("Malformed/skipped records:", malformed_records)
print("Unique source subscribers:", len(source_subscribers))
print("Unique destination subscribers:", len(destination_subscribers))
print("Unique subscribers overall:", len(all_subscribers))

print("\nFraudulent IDs:", len(fraud_ids))
print("Fraudulent IDs found in CDRs:", len(matched))
print("Fraudulent IDs not found in CDRs:", len(unmatched))

if fraud_ids:
    print(
        "Fraud-label matching rate:",
        f"{len(matched) / len(fraud_ids) * 100:.2f}%"
    )

print("\nSample matched IDs:")
for number in sorted(matched)[:10]:
    print(number)

print("\nSample unmatched IDs:")
for number in sorted(unmatched)[:10]:
    print(number)
