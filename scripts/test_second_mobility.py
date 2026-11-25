
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

files = [
    f for f in RAW.rglob("*.csv")
    if not f.name.startswith("._")
]

for file in files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            header = f.readline().strip()

        if header == "time_sec id cell_id time":
            print("\nFOUND:", file)
            print("\nRAW RECORDS:")

            with open(file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == 8:
                        break
                    print(repr(line))

            df = pd.read_csv(
                file,
                sep=" ",
                quotechar='"',
                skipinitialspace=True,
                nrows=10
            )

            print("\nPARSED DATA:")
            print(df.to_string(index=False))
            break

    except (UnicodeDecodeError, OSError):
        continue
else:
    print("No matching mobility file found.")
