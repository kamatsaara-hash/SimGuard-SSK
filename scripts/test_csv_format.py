
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

files = [
    f for f in RAW.rglob("*.csv")
    if not f.name.startswith("._")
]

print("Testing first 5 CSV files...\n")

for file in files[:5]:
    print("=" * 60)
    print("FILE:", file.name)

    try:
        df = pd.read_csv(
            file,
            sep=r"\s+",
            engine="python",
            nrows=5
        )

        print("COLUMNS:", df.columns.tolist())
        print("\nSAMPLE:")
        print(df.to_string(index=False))

    except Exception as e:
        print("ERROR:", e)
