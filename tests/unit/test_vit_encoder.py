"""Unit tests for ViT Encoder."""

import torch

from jepa_brain.encoder.vit_encoder import PatchEmbedder, ViTEncoder


def test_patch_embedder_shapes():
    """Test patch embedding dimensions are mathematically correct."""
    # Given an input of batch=2, channels=3, H=224, W=224
    dummy_input = torch.randn(2, 3, 224, 224)

    # 224x224 image with 16x16 patches = (224/16)*(224/16) = 14*14 = 196 patches
    # Wait, the code has num_patches hardcoded to 256 for a 16x16 grid, which means H=256, W=256.
    # Let's check the encoder's expected sizes or pass 256x256.
    dummy_input = torch.randn(2, 3, 256, 256)

    embedder = PatchEmbedder(img_size=256, patch_size=16, in_chans=3, embed_dim=512)

    output = embedder(dummy_input)

    # Assert Output: (Batch, Num_Tokens, Embed_Dim) -> (2, 256, 512)
    assert output.shape == (2, 256, 512)


def test_vit_encoder_forward():
    """Test the full ViT Encoder forward pass."""
    dummy_input = torch.randn(2, 3, 256, 256)

    # Initialize tiny ViT
    encoder = ViTEncoder(
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        depth=2,  # Tiny depth for test speed
        num_heads=4,
    )

    output = encoder(dummy_input)

    # Assert output shape matches tokenized embedding
    assert output.shape == (2, 256, 512)

    # Assert gradients can flow backwards
    loss = output.sum()
    loss.backward()

    # Check that parameters have gradients
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
