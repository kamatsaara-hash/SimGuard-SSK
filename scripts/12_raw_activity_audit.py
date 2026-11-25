import pandas as pd
import numpy as np

cdr = pd.read_csv("data/processed/clean_cdr.csv")
labels = pd.read_csv("data/processed/subscriber_labels.csv")

fraud_ids = set(
    labels.loc[labels["label"] == 1, "subscriber_id"].astype(str)
)

cdr["src_number"] = cdr["src_number"].astype(str)

cdr["timestamp"] = pd.to_datetime(
    cdr["timestamp"], errors="coerce"
)

cdr["duration"] = pd.to_numeric(
    cdr["duration"], errors="coerce"
).fillna(0)

cdr["hour"] = cdr["timestamp"].dt.hour

outgoing = cdr[cdr["src_number"].isin(
    labels["subscriber_id"].astype(str)
)].copy()

outgoing["group"] = np.where(
    outgoing["src_number"].isin(fraud_ids),
    "FRAUD",
    "NON_FRAUD"
)

summary = outgoing.groupby("group").agg(
    records=("src_number", "size"),
    subscribers=("src_number", "nunique"),
    mean_duration=("duration", "mean"),
    std_duration=("duration", "std"),
    night_events=("hour", lambda x: ((x < 6) | (x >= 22)).sum())
)

print("\n===== RAW ACTIVITY STATISTICS =====")
print(summary.round(4).to_string())

print("\n===== PER-SUBSCRIBER ACTIVITY SAMPLE =====")

per_subscriber = outgoing.groupby(
    ["src_number", "group"]
).agg(
    records=("duration", "size"),
    mean_duration=("duration", "mean"),
    std_duration=("duration", "std"),
    night_events=("hour", lambda x: ((x < 6) | (x >= 22)).sum()),
    active_span_hours=(
        "timestamp",
        lambda x: (x.max() - x.min()).total_seconds() / 3600
    )
).reset_index()

print(
    per_subscriber.groupby("group")[
        ["records", "mean_duration", "std_duration",
         "night_events", "active_span_hours"]
    ].describe().round(4).to_string()
)
