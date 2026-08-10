import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    """Fuses camera token embeddings with LiDAR token embeddings via Cross-Attention."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, camera_tokens: torch.Tensor, lidar_tokens: torch.Tensor) -> torch.Tensor:
        # If no lidar tokens, return camera tokens
        if lidar_tokens is None:
            return camera_tokens
            
        # Treat camera as queries, lidar as keys/values
        attn_out, _ = self.cross_attn(camera_tokens, lidar_tokens, lidar_tokens)
        return self.norm(camera_tokens + attn_out)
