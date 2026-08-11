import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """Fuses camera token embeddings with LiDAR token embeddings via Cross-Attention."""

    def __init__(self, embed_dim=512):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, camera_tokens: torch.Tensor, lidar_tokens: torch.Tensor) -> torch.Tensor:
        # If no lidar tokens, return camera tokens
        if lidar_tokens is None:
            return camera_tokens

        if lidar_tokens.dim() == 2:
            lidar_tokens = lidar_tokens.unsqueeze(1)

        # Treat camera as queries, lidar as keys/values
        attn_out, _ = self.cross_attn(camera_tokens, lidar_tokens, lidar_tokens)
        fused = self.norm(camera_tokens + attn_out)

        # FFN sub-layer
        fused = self.norm2(fused + self.ffn(fused))

        return fused
