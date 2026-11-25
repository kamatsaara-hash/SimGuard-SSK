import torch

data = torch.load(
    "data/processed/hetero_graph.pt",
    map_location="cpu",
    weights_only=False
)

print("Subscriber feature shape:", data["subscriber"].x.shape)
print("Feature indices to remove: 0, 1, 8, 9")
print("Remaining feature count:", data["subscriber"].x.shape[1] - 4)

print("Train:", int(data["subscriber"].train_mask.sum()))
print("Validation:", int(data["subscriber"].val_mask.sum()))
print("Test:", int(data["subscriber"].test_mask.sum()))
