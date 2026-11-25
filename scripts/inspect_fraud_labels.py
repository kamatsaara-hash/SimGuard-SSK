
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

scenario = (
    RAW
    / "Fraudzen-dataset"
    / "Fraudzen-dataset"
    / "FraudZen_Cdrs"
    / "all_advanced"
    / "advanced_3%_50sims"
)

print("SCENARIO:", scenario)

if not scenario.exists():
    print("Scenario folder not found!")
    exit()

print("\nFILES IN SCENARIO:")
for f in sorted(scenario.iterdir()):
    if f.is_file():
        print(f"{f.name}  |  {f.stat().st_size:,} bytes")

label_file = scenario / "fraudulentUsers.txt"

print("\n" + "=" * 60)
print("FRAUD LABEL FILE")
print("=" * 60)

if label_file.exists():
    with open(label_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 25:
                break
            print(repr(line))
else:
    print("fraudulentUsers.txt not found!")

print("\n" + "=" * 60)
print("CDR FILE SAMPLES")
print("=" * 60)

cdr_files = list(scenario.glob("*CDRTrace*"))

for file in cdr_files[:3]:
    print("\nFILE:", file.name)

    with open(file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 6:
                break
            print(repr(line))
