import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Encodes a sequence of historical states using causal attention."""

    def __init__(self, embed_dim=512, num_heads=8, num_layers=2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Time, Embed_dim)
        return self.transformer(x)
