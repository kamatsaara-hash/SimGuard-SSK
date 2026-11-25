
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

files = [
    f for f in RAW.rglob("*")
    if f.is_file() and not f.name.startswith("._")
]

print("TOTAL FILES:", len(files))

print("\nFOLDER STRUCTURE:")
folders = sorted({
    str(f.parent.relative_to(RAW))
    for f in files
})

for folder in folders:
    print(" -", folder)

print("\nFILE EXTENSIONS:")
print(Counter(f.suffix.lower() for f in files))

print("\nCSV FILES OUTSIDE INPUTS:")
for f in files:
    if f.suffix.lower() == ".csv" and "Inputs" not in f.parts:
        print(f.relative_to(RAW))

print("\nFILES WITH POSSIBLE LABEL / FRAUD INFORMATION:")
keywords = ["label", "fraud", "simbox", "groundtruth", "ground_truth", "target"]

for f in files:
    if any(word in f.name.lower() for word in keywords):
        print(f.relative_to(RAW))
