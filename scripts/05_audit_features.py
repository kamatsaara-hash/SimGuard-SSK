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

feature_names = [
    "outgoing_count",
    "incoming_count",
    "unique_out_contacts",
    "unique_in_contacts",
    "unique_devices",
    "unique_cells",
    "mean_duration",
    "std_duration",
    "mean_size",
    "type_DATA",
    "type_I-MOC",
    "type_I-MTC",
    "type_MOC",
    "type_MO-SMS",
    "type_MTC",
    "international_out",
    "international_in",
    "night_events",
    "active_span_hours"
]

print("Feature matrix:", x.shape)
print("Fraud subscribers:", int(y.sum()))
print("Provisional non-fraud:", int((y == 0).sum()))

results = []

for i, name in enumerate(feature_names):
    fraud = x[y == 1, i]
    nonfraud = x[y == 0, i]

    pooled_std = np.sqrt(
        (fraud.var() + nonfraud.var()) / 2
    )

    effect = (
        (fraud.mean() - nonfraud.mean()) / pooled_std
        if pooled_std > 1e-12 else 0
    )

    results.append({
        "feature": name,
        "fraud_mean": fraud.mean(),
        "nonfraud_mean": nonfraud.mean(),
        "fraud_median": np.median(fraud),
        "nonfraud_median": np.median(nonfraud),
        "standardized_difference": effect
    })

df = pd.DataFrame(results)
df["absolute_difference"] = df["standardized_difference"].abs()
df = df.sort_values("absolute_difference", ascending=False)

print("\n========== FEATURE SEPARATION AUDIT ==========")
print(df.drop(columns="absolute_difference").round(4).to_string(index=False))

df.to_csv("results/HGT/feature_separation_audit.csv", index=False)

print("\nSaved: results/HGT/feature_separation_audit.csv")
