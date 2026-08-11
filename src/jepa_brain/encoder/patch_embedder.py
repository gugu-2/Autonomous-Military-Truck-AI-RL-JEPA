"""Patch embedding layer for Vision Transformer — splits images into tokens."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedder(nn.Module):
    """Convolutional patch embedding."""

    def __init__(
        self, img_size=224, patch_size=16, in_channels: int = 3, in_chans: int = None, embed_dim=512
    ):
        super().__init__()
        if in_chans is not None:
            in_channels = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 2D Positional Embeddings
        self.register_buffer(
            "pos_embed", self._get_sinusoid_encoding_table(self.num_patches, embed_dim)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / math.pow(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = torch.tensor(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)], dtype=torch.float32
        )
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        return sinusoid_table.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        _, _, H_out, W_out = x.shape

        pos_embed = self.pos_embed
        if H_out * W_out != self.num_patches:
            grid_size = int(self.num_patches**0.5)
            pos_embed = pos_embed.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, size=(H_out, W_out), mode="bilinear", align_corners=False
            )
            pos_embed = pos_embed.flatten(2).transpose(1, 2)

        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = x + pos_embed
        return x


class CameraTokenizer(nn.Module):
    """Takes 4 camera images and produces flattened camera tokens."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=512, num_cameras=4):
        super().__init__()
        self.patch_embedder = PatchEmbedder(img_size, patch_size, in_channels, embed_dim)
        self.num_cameras = num_cameras
        self.cam_pos_embed = nn.Parameter(torch.randn(1, num_cameras, 1, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, num_cameras, C, H, W)
        B, num_cams, C, H, W = x.shape
        x = x.view(B * num_cams, C, H, W)

        # Patch embed
        patches = self.patch_embedder(x)  # (B*num_cams, N, D)

        # Reshape to (B, num_cams, N, D)
        _, N, D = patches.shape
        patches = patches.view(B, num_cams, N, D)

        # Add camera positional embedding
        patches = patches + self.cam_pos_embed

        # Flatten across cameras and patches: (B, num_cams*N, D)
        patches = patches.view(B, num_cams * N, D)
        return patches
