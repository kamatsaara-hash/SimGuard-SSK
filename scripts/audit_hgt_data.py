
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

SCENARIO = (
    RAW / "Fraudzen-dataset" / "Fraudzen-dataset"
    / "FraudZen_Cdrs" / "all_advanced" / "advanced_3%_50sims"
)

fraud_file = SCENARIO / "fraudulentUsers.txt"

fraud_ids = {
    line.strip().lstrip("+")
    for line in open(fraud_file, encoding="utf-8")
    if line.strip()
}

cdr_files = [
    "Op_1_CDRTrace.txt",
    "International_CDRTrace.txt"
]

all_users = set()
source_users = set()
destination_users = set()

imei_values = set()
cell_values = set()

event_types = Counter()
records_per_file = Counter()
missing_size = Counter()
malformed = Counter()

fraud_source = set()
fraud_destination = set()

for filename in cdr_files:
    file = SCENARIO / filename
    print("\nProcessing:", filename)

    with open(file, encoding="utf-8") as f:
        next(f, None)

        for line in f:
            p = line.split()

            # Required fields end at duration (index 11).
            if len(p) < 12:
                malformed[filename] += 1
                continue

            src = p[3].lstrip("+")
            src_imei = p[4]
            src_cell = p[5]

            dst = p[7].lstrip("+")
            dst_imei = p[8]
            dst_cell = p[9]

            event = p[2]

            if not src.isdigit() or not dst.isdigit():
                malformed[filename] += 1
                continue

            records_per_file[filename] += 1

            if len(p) < 13:
                missing_size[filename] += 1

            source_users.add(src)
            destination_users.add(dst)
            all_users.update([src, dst])

            event_types[event] += 1

            if src in fraud_ids:
                fraud_source.add(src)

            if dst in fraud_ids:
                fraud_destination.add(dst)

            for imei in [src_imei, dst_imei]:
                if imei.lower() != "unknown":
                    imei_values.add(imei)

            for cell in [src_cell, dst_cell]:
                if cell != "0":
                    cell_values.add(cell)

print("\n" + "=" * 55)
print("CORRECTED HGT DATA AUDIT")
print("=" * 55)

print("\nRECORDS PER FILE:")
for name, count in records_per_file.items():
    print(name, ":", count)

print("\nMISSING SIZE FIELD:")
for name, count in missing_size.items():
    print(name, ":", count)

print("\nMALFORMED RECORDS:")
for name, count in malformed.items():
    print(name, ":", count)

print("\nEVENT TYPES:")
for name, count in event_types.most_common():
    print(name, ":", count)

print("\nSUBSCRIBERS:")
print("Total unique:", len(all_users))
print("Unique source:", len(source_users))
print("Unique destination:", len(destination_users))

print("\nFRAUD LABEL COVERAGE:")
print("Fraud IDs:", len(fraud_ids))
print("Fraud IDs appearing as source:", len(fraud_source))
print("Fraud IDs appearing as destination:", len(fraud_destination))
print("Fraud IDs found in either role:",
      len(fraud_ids.intersection(all_users)))

print("\nGRAPH ENTITY COUNTS:")
print("Known IMEIs:", len(imei_values))
print("Non-zero cell IDs:", len(cell_values))

print("\nCLASS COUNTS:")
print("Fraud-labelled:", len(fraud_ids))
print("Unlabelled observed:", len(all_users - fraud_ids))
