import torch
import torch.nn as nn


class LidarEncoder(nn.Module):
    """Encodes a LiDAR BEV grid into latent tokens."""

    def __init__(self, embed_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, embed_dim),
        )

    def forward(self, bev_grid: torch.Tensor) -> torch.Tensor:
        # Expected input: (Batch, 1, 224, 224)
        output = self.conv(bev_grid)
        return output.unsqueeze(1)
