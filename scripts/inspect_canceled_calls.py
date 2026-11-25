
from pathlib import Path
from collections import Counter

ROOT = Path("data/raw/Fraudzen-dataset/Fraudzen-dataset")
SCENARIO = ROOT / "FraudZen_Cdrs/all_advanced/advanced_3%_50sims"

cancel_file = SCENARIO / "softswitch_canceled_calls.txt"
fraud_file = SCENARIO / "fraudulentUsers.txt"

# Normalize subscriber numbers while preserving them as strings
def normalize(number):
    return number.strip().lstrip("+")

# Read fraud labels
fraud_ids = {
    normalize(line)
    for line in fraud_file.read_text().splitlines()
    if line.strip()
}

# Read subscriber IDs from both CDR files
cdr_subscribers = set()

for filename in ["Op_1_CDRTrace.txt", "International_CDRTrace.txt"]:
    path = SCENARIO / filename

    with path.open("r", encoding="utf-8") as f:
        next(f, None)

        for line in f:
            p = line.split()
            if len(p) < 12:
                continue

            cdr_subscribers.add(normalize(p[3]))
            cdr_subscribers.add(normalize(p[7]))

# Read cancelled-call records
destinations = []
reasons = Counter()
durations = []
records = []

with cancel_file.open("r", encoding="utf-8") as f:
    next(f, None)

    for line in f:
        p = line.split()

        if len(p) != 5:
            continue

        destination = normalize(p[2])
        duration = int(p[3])
        reason = p[4]

        destinations.append(destination)
        reasons[reason] += 1
        durations.append(duration)
        records.append((destination, duration, reason))

destination_set = set(destinations)
fraud_matches = destination_set & fraud_ids
cdr_matches = destination_set & cdr_subscribers

print("\n========== CANCELLED CALL AUDIT ==========")

print("Total records:", len(records))
print("Unique destinations:", len(destination_set))
print("Duplicate records:", len(records) - len(set(records)))

print("\nCANCELLATION REASONS:")
for reason, count in reasons.items():
    print(reason, ":", count)

print("\nDURATION:")
if durations:
    print("Minimum:", min(durations))
    print("Maximum:", max(durations))
    print("Average:", round(sum(durations) / len(durations), 2))

print("\nSUBSCRIBER MATCHING:")
print("Destinations found in CDRs:", len(cdr_matches))
print("Destinations not found in CDRs:", len(destination_set - cdr_subscribers))

print("\nFRAUD MATCHING:")
print("Known fraud IDs:", len(fraud_ids))
print("Fraud IDs appearing in cancelled calls:", len(fraud_matches))
print("Fraud IDs appearing in cancelled calls:", sorted(fraud_matches))

print("\n========== AUDIT COMPLETE ==========")
