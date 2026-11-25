import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class HGTFraudDetector(nn.Module):
    def __init__(
        self,
        metadata,
        input_dims,
        hidden_channels=32,
        heads=2,
        dropout=0.3
    ):
        super().__init__()

        self.node_types = metadata[0]
        self.dropout = dropout

        self.input_projection = nn.ModuleDict({
            node_type: nn.Linear(input_dims[node_type], hidden_channels)
            for node_type in self.node_types
        })

        self.hgt1 = HGTConv(
            hidden_channels, hidden_channels, metadata, heads=heads
        )

        self.hgt2 = HGTConv(
            hidden_channels, hidden_channels, metadata, heads=heads
        )

        self.norm1 = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_channels)
            for node_type in self.node_types
        })

        self.norm2 = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_channels)
            for node_type in self.node_types
        })

        # Direct subscriber-feature pathway
        self.subscriber_skip = nn.Sequential(
            nn.Linear(input_dims["subscriber"], hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Combine graph representation and original subscriber features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x_dict, edge_index_dict):

        original_subscriber_features = x_dict["subscriber"]

        x_dict = {
            node_type: F.relu(self.input_projection[node_type](x))
            for node_type, x in x_dict.items()
        }

        h_dict = self.hgt1(x_dict, edge_index_dict)

        x_dict = {
            node_type: F.dropout(
                F.relu(
                    self.norm1[node_type](
                        h_dict[node_type] + x_dict[node_type]
                    )
                ),
                p=self.dropout,
                training=self.training
            )
            for node_type in h_dict
            if h_dict[node_type] is not None
        }

        h_dict = self.hgt2(x_dict, edge_index_dict)

        x_dict = {
            node_type: F.dropout(
                F.relu(
                    self.norm2[node_type](
                        h_dict[node_type] + x_dict[node_type]
                    )
                ),
                p=self.dropout,
                training=self.training
            )
            for node_type in h_dict
            if h_dict[node_type] is not None
        }

        graph_embedding = x_dict["subscriber"]

        feature_embedding = self.subscriber_skip(
            original_subscriber_features
        )

        combined = torch.cat(
            [graph_embedding, feature_embedding],
            dim=1
        )

        return self.classifier(combined).squeeze(-1)
