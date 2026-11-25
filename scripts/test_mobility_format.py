
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

file = RAW / "Fraudzen-dataset" / "Fraudzen-dataset" / "Inputs" / "regular_mob_traces.csv"

df = pd.read_csv(
    file,
    sep=" ",
    quotechar='"',
    skipinitialspace=True,
    nrows=10
)

print("COLUMNS:", df.columns.tolist())
print("\nDATA:")
print(df.to_string(index=False))

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())
