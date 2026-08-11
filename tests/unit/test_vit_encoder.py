"""Unit tests for ViT Encoder and PatchEmbedder."""

import pytest
import torch

try:
    from jepa_brain.encoder.patch_embedder import PatchEmbedder
    from jepa_brain.encoder.vit_encoder import ViTEncoder

    HAS_TORCH_MODULES = True
except ImportError:
    HAS_TORCH_MODULES = False


@pytest.mark.skipif(not HAS_TORCH_MODULES, reason="jepa_brain not importable")
def test_patch_embedder_shapes():
    """Test patch embedding produces correct token shape."""
    embedder = PatchEmbedder(img_size=256, patch_size=16, in_channels=3, embed_dim=512)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = embedder(dummy_input)
    # 256x256 / 16x16 = 256 patches
    assert output.shape == (2, 256, 512), f"Expected (2, 256, 512), got {output.shape}"


@pytest.mark.skipif(not HAS_TORCH_MODULES, reason="jepa_brain not importable")
def test_vit_encoder_forward():
    """Test the full ViT Encoder forward pass on tokenized input."""
    # First embed patches, THEN pass to ViTEncoder
    embedder = PatchEmbedder(img_size=256, patch_size=16, in_channels=3, embed_dim=512)
    encoder = ViTEncoder(embed_dim=512, depth=2, num_heads=4)

    dummy_input = torch.randn(2, 3, 256, 256)
    tokens = embedder(dummy_input)  # -> (2, 256, 512)
    output = encoder(tokens)  # -> (2, 256, 512)

    assert output.shape == (2, 256, 512)

    # Verify gradient flow
    loss = output.sum()
    loss.backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
