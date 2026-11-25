
from pathlib import Path
from collections import Counter

ROOT = Path("data/raw/Fraudzen-dataset/Fraudzen-dataset")
SCENARIO = ROOT / "FraudZen_Cdrs/all_advanced/advanced_3%_50sims"

# Normalize phone numbers but preserve them as strings
def norm(value):
    return value.strip().lstrip("+")

# Load subscribers from selected CDR scenario
cdr_subscribers = set()

for filename in ["Op_1_CDRTrace.txt", "International_CDRTrace.txt"]:
    with (SCENARIO / filename).open(encoding="utf-8") as f:
        next(f, None)

        for line in f:
            p = line.split()
            if len(p) >= 12:
                cdr_subscribers.add(norm(p[3]))
                cdr_subscribers.add(norm(p[7]))

# Load fraud labels
fraud_ids = {
    norm(x)
    for x in (SCENARIO / "fraudulentUsers.txt").read_text().splitlines()
    if x.strip()
}

# Audit regular subscriber mobility
mobility_file = ROOT / "Inputs/regular_mob_traces.csv"

unique_subscribers = set()
unique_cells = set()
matched_subscribers = set()
fraud_in_mobility = set()

total_rows = 0
matched_rows = 0
fraud_rows = 0
malformed = 0

min_time_sec = None
max_time_sec = None
first_timestamp = None
last_timestamp = None

with mobility_file.open(encoding="utf-8") as f:
    next(f, None)

    for line in f:
        p = line.split()

        if len(p) < 5:
            malformed += 1
            continue

        total_rows += 1

        time_sec = int(p[0])
        subscriber = norm(p[1])
        cell = p[2]
        timestamp = p[3].strip('"') + " " + p[4].strip('"')

        unique_subscribers.add(subscriber)
        unique_cells.add(cell)

        if subscriber in cdr_subscribers:
            matched_subscribers.add(subscriber)
            matched_rows += 1

        if subscriber in fraud_ids:
            fraud_in_mobility.add(subscriber)
            fraud_rows += 1

        min_time_sec = time_sec if min_time_sec is None else min(min_time_sec, time_sec)
        max_time_sec = time_sec if max_time_sec is None else max(max_time_sec, time_sec)

        if first_timestamp is None:
            first_timestamp = timestamp
        last_timestamp = timestamp

print("\n========== MOBILITY ALIGNMENT AUDIT ==========")

print("\nRECORD COUNTS:")
print("Mobility rows:", total_rows)
print("Malformed rows:", malformed)

print("\nSUBSCRIBER OVERLAP:")
print("Unique mobility subscribers:", len(unique_subscribers))
print("Unique CDR subscribers:", len(cdr_subscribers))
print("Mobility subscribers found in CDR:", len(matched_subscribers))
print("CDR subscribers found in mobility:",
      len(cdr_subscribers & unique_subscribers))

print("\nFRAUD OVERLAP:")
print("Known fraud IDs:", len(fraud_ids))
print("Fraud IDs found in mobility:", len(fraud_in_mobility))
print("Fraud mobility records:", fraud_rows)
print("Fraud IDs missing from mobility:", len(fraud_ids - unique_subscribers))

print("\nCELL COVERAGE:")
print("Unique mobility cells:", len(unique_cells))

print("\nTIME COVERAGE:")
print("First timestamp:", first_timestamp)
print("Last timestamp:", last_timestamp)
print("Minimum time_sec:", min_time_sec)
print("Maximum time_sec:", max_time_sec)

print("\n========== AUDIT COMPLETE ==========")
