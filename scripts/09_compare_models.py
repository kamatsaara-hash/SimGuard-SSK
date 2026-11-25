import json
import pandas as pd
from pathlib import Path

files = {
    "Original HGT": "results/HGT/test_metrics.json",
    "Ablation HGT": "results/HGT_ablation/test_metrics.json",
    "MLP": "results/MLP_baseline/test_metrics.json"
}

rows = []

for model, path in files.items():
    with open(path) as f:
        m = json.load(f)

    rows.append({
        "Model": model,
        "Accuracy": m["test_accuracy"],
        "Precision": m["test_precision"],
        "Recall": m["test_recall"],
        "F1": m["test_f1"],
        "ROC_AUC": m["test_roc_auc"],
        "PR_AUC": m["test_pr_auc"]
    })

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/model_comparison.csv", index=False)

print("\n========== MODEL COMPARISON ==========")
print(df.round(4).to_string(index=False))
print("\nSaved: results/model_comparison.csv")
