
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

files = [
    f for f in RAW.rglob("*.csv")
    if not f.name.startswith("._")
]

print("TOTAL CSV FILES:", len(files))

total_rows = 0
all_columns = set()
event_counts = {}
subscribers = set()
contacts = set()

for i, file in enumerate(files, 1):
    try:
        df = pd.read_csv(file, sep=r"\s+", engine="python")

        total_rows += len(df)
        all_columns.update(df.columns)

        if "evt" in df.columns:
            counts = df["evt"].value_counts()
            for event, count in counts.items():
                event_counts[event] = (
                    event_counts.get(event, 0) + count
                )

        if "callernumber" in df.columns:
            subscribers.update(df["callernumber"].dropna())

        if "callednumber" in df.columns:
            contacts.update(df["callednumber"].dropna())

        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} files")

    except Exception as e:
        print("ERROR:", file.name, e)

print("\n========== DATASET SUMMARY ==========")
print("Total files:", len(files))
print("Total records:", total_rows)
print("Columns:", sorted(all_columns))
print("Unique caller numbers:", len(subscribers))
print("Unique called numbers:", len(contacts))
print("Event distribution:", event_counts)

print("\nInspection complete!")
