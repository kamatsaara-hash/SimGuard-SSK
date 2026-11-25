import pandas as pd

cdr = pd.read_csv("data/processed/clean_cdr.csv")
labels = pd.read_csv("data/processed/subscriber_labels.csv")

fraud_ids = set(
    labels.loc[labels["label"] == 1, "subscriber_id"].astype(str)
)

cdr["src_number"] = cdr["src_number"].astype(str)
fraud = cdr[cdr["src_number"].isin(fraud_ids)].copy()

summary = fraud.groupby("src_number").agg(
    total_records=("src_number", "size"),
    data_records=("cdr_type", lambda x: (x == "DATA").sum()),
    moc_records=("cdr_type", lambda x: (x == "MOC").sum()),
    mean_size=("size", "mean"),
    mean_duration=("duration", "mean")
)

print("========== FRAUD SUBSCRIBER PATTERN ==========")
print("Fraud subscribers:", len(summary))
print("\nRecords per subscriber:")
print(summary["total_records"].value_counts().sort_index())

print("\nCDR type counts per subscriber:")
print(summary[["data_records", "moc_records"]].value_counts().to_string())


print("\nFraud subscriber summary:")
print(summary.describe().round(3).to_string())

print("\nSaved audit:")
summary.to_csv("results/HGT/fraud_pattern_audit.csv")
