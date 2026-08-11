import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Encodes a sequence of historical states using causal attention."""

    def __init__(self, embed_dim=512, num_heads=8, num_layers=2, context_frames=5):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, context_frames, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Time, Embed_dim)
        x = x + self.temporal_pos_embed[:, : x.size(1), :]
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        out = self.transformer(x, mask=causal_mask)
        return out
