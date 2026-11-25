import pandas as pd
import numpy as np

cdr = pd.read_csv("data/processed/clean_cdr.csv")
labels = pd.read_csv("data/processed/subscriber_labels.csv")

fraud_ids = set(
    labels.loc[labels["label"] == 1, "subscriber_id"].astype(str)
)

cdr["src_number"] = cdr["src_number"].astype(str)

cdr["group"] = np.where(
    cdr["src_number"].isin(fraud_ids),
    "FRAUD",
    "PROVISIONAL_NON_FRAUD"
)

print("CDR columns:", cdr.columns.tolist())
print("\n========== RAW CDR GROUP SUMMARY ==========")

for group, df in cdr.groupby("group"):
    print(f"\n{group}")
    print("Total outgoing records:", len(df))
    print("Unique source subscribers:", df["src_number"].nunique())
    print("\nCDR types:")
    print(df["cdr_type"].value_counts().to_string())

    for col in ["size", "duration"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            print(f"\n{col} statistics:")
            print(values.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(3))

print("\n========== FRAUD SUBSCRIBER SAMPLE ==========")

fraud_summary = (
    cdr[cdr["group"] == "FRAUD"]
    .groupby("src_number")
    .agg(
        records=("src_number", "size"),
        mean_size=("size", "mean"),
        mean_duration=("duration", "mean"),
        data_records=("cdr_type", lambda x: (x == "DATA").sum())
    )
)

print(fraud_summary.head(15).round(3).to_string())

print("\n========== PROVISIONAL NON-FRAUD SAMPLE ==========")

normal_summary = (
    cdr[cdr["group"] == "PROVISIONAL_NON_FRAUD"]
    .groupby("src_number")
    .agg(
        records=("src_number", "size"),
        mean_size=("size", "mean"),
        mean_duration=("duration", "mean"),
        data_records=("cdr_type", lambda x: (x == "DATA").sum())
    )
)

print(normal_summary.head(15).round(3).to_string())
