import torch
import numpy as np
import pandas as pd

data = torch.load(
    "data/processed/hetero_graph.pt",
    map_location="cpu",
    weights_only=False
)

x = data["subscriber"].x.numpy()
y = data["subscriber"].y.numpy()

removed = [0, 1, 8, 9]

features = [
    "outgoing_count", "incoming_count",
    "unique_out_contacts", "unique_in_contacts",
    "unique_devices", "unique_cells",
    "mean_duration", "std_duration", "mean_size",
    "type_DATA", "type_I-MOC", "type_I-MTC",
    "type_MOC", "type_MO-SMS", "type_MTC",
    "international_out", "international_in",
    "night_events", "active_span_hours"
]

rows = []

for i, name in enumerate(features):
    if i in removed:
        continue

    fraud = x[y == 1, i]
    nonfraud = x[y == 0, i]

    pooled_std = np.sqrt(
        (np.var(fraud) + np.var(nonfraud)) / 2
    )

    difference = (
        (np.mean(fraud) - np.mean(nonfraud)) / pooled_std
        if pooled_std > 0 else 0
    )

    rows.append({
        "Feature": name,
        "Fraud_Mean": np.mean(fraud),
        "NonFraud_Mean": np.mean(nonfraud),
        "Fraud_Median": np.median(fraud),
        "NonFraud_Median": np.median(nonfraud),
        "Standardized_Difference": difference
    })

df = pd.DataFrame(rows)
df["Absolute_Difference"] = df["Standardized_Difference"].abs()
df = df.sort_values("Absolute_Difference", ascending=False)

print("\n======= REMAINING MLP FEATURE ANALYSIS =======")
print(df.drop(columns="Absolute_Difference").round(4).to_string(index=False))

df.to_csv("results/MLP_baseline/remaining_feature_analysis.csv", index=False)
print("\nSaved feature analysis.")
