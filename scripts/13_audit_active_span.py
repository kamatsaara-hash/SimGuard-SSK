import pandas as pd
import numpy as np

cdr = pd.read_csv("data/processed/clean_cdr.csv")
labels = pd.read_csv("data/processed/subscriber_labels.csv")

cdr["src_number"] = cdr["src_number"].astype(str)
cdr["timestamp"] = pd.to_datetime(cdr["timestamp"], errors="coerce")
cdr["duration"] = pd.to_numeric(cdr["duration"], errors="coerce").fillna(0)

fraud_ids = set(
    labels.loc[labels["label"] == 1, "subscriber_id"].astype(str)
)

per_sub = cdr.groupby("src_number").agg(
    records=("timestamp", "size"),
    mean_duration=("duration", "mean"),
    std_duration=("duration", "std"),
    first_event=("timestamp", "min"),
    last_event=("timestamp", "max")
)

per_sub["active_span_hours"] = (
    per_sub["last_event"] - per_sub["first_event"]
).dt.total_seconds() / 3600

per_sub["label"] = np.where(
    per_sub.index.isin(fraud_ids), "FRAUD", "NON_FRAUD"
)

print("\n===== FRAUD ACTIVE SPAN VALUES =====")
print(
    per_sub[per_sub["label"] == "FRAUD"][
        ["records", "active_span_hours"]
    ].to_string()
)

print("\n===== ACTIVE SPAN DISTRIBUTION =====")
print(
    per_sub.groupby("label")["active_span_hours"]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    .round(5)
)

print("\n===== EXACT FRAUD SPAN FREQUENCY =====")
print(
    per_sub[per_sub["label"] == "FRAUD"]
    ["active_span_hours"].value_counts()
)
