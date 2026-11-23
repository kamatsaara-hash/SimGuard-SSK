"""SIMGuard graph-aware Temporal Transformer for subscriber fraud classification.

Model inputs are (batch, time, 3, features) where each graph token is:
    0 = the subscriber's own features in the current window
    1 = weighted mean of OUTGOING neighbors in the same window
    2 = weighted mean of INCOMING neighbors in the same window

Graph attention learns how to combine these three tokens for every hour.
Causal temporal self-attention then models history through the observation end.
This is a *neighbor-summary* graph Transformer baseline, not a full HGT or an
edge-level message-passing Transformer. Fraud labels are never model inputs.
"""

import math

import torch
from torch import nn


class TemporalGraphTransformer(nn.Module):
    def __init__(self, feature_dim=17, hidden_dim=48, heads=4, layers=1, dropout=0.1):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.role_embedding = nn.Parameter(torch.zeros(1, 1, 3, hidden_dim))
        nn.init.normal_(self.role_embedding, std=0.02)
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.graph_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def sinusoidal_positions(length, width, device, dtype):
        pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        index = torch.arange(0, width, 2, device=device, dtype=dtype)
        frequencies = torch.exp(index * (-math.log(10000.0) / width))
        result = torch.zeros(length, width, device=device, dtype=dtype)
        result[:, 0::2] = torch.sin(pos * frequencies)
        # width need not be even.
        result[:, 1::2] = torch.cos(pos * frequencies[:width // 2])
        return result.unsqueeze(0)

    def forward(self, tokens, neighbor_available):
        """Return fraud logits.

        tokens: float32 (B,T,3,F); neighbor_available: bool (B,T,2),
        marking nonempty outgoing/incoming neighborhood summaries.
        Prediction is made at the END of the final observed time window.
        """
        if tokens.ndim != 4 or tokens.shape[2] != 3 or tokens.shape[3] != self.feature_dim:
            raise ValueError("tokens must be (batch, time, 3, feature_dim)")
        batch, steps, _, _ = tokens.shape
        if neighbor_available.shape != (batch, steps, 2):
            raise ValueError("neighbor_available must have shape (batch,time,2)")

        h = self.input_projection(tokens) + self.role_embedding
        h = h.reshape(batch * steps, 3, self.hidden_dim)
        # Always allow attending to self; absent neighbor summaries are masked.
        neighbor_padding = ~neighbor_available.reshape(batch * steps, 2)
        key_padding_mask = torch.cat(
            [torch.zeros(batch * steps, 1, dtype=torch.bool, device=tokens.device),
             neighbor_padding], dim=1,
        )
        graph_out, _ = self.graph_attention(
            query=h[:, :1, :], key=h, value=h,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        temporal = self.graph_norm(h[:, 0, :] + graph_out[:, 0, :])
        temporal = temporal.reshape(batch, steps, self.hidden_dim)
        temporal = temporal + self.sinusoidal_positions(
            steps, self.hidden_dim, tokens.device, temporal.dtype,
        )
        # True entries disallow future-to-past attention.
        future_mask = torch.ones(steps, steps, device=tokens.device, dtype=torch.bool).triu(1)
        temporal = self.temporal_encoder(temporal, mask=future_mask)
        return self.head(temporal[:, -1, :]).squeeze(-1)
