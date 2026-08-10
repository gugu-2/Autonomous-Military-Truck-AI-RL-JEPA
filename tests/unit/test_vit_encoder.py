"""Unit tests for ViT Encoder."""
import pytest
import torch
import torch.nn as nn

class MockViTEncoder(nn.Module):
    def __init__(self, embed_dim=256, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)
        # x: (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)
        return x

def test_output_shape():
    encoder = MockViTEncoder(embed_dim=256)
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 196, 256)

def test_patch_count():
    encoder = MockViTEncoder(patch_size=16)
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape[1] == 196 # (224/16) * (224/16) = 14 * 14 = 196

def test_no_nan():
    encoder = MockViTEncoder()
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert not torch.any(torch.isnan(out))

def test_gradient_flows():
    encoder = MockViTEncoder()
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_device_consistency():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = MockViTEncoder().to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    out = encoder(x)
    assert out.device == x.device

def test_batch_size_1():
    encoder = MockViTEncoder()
    x = torch.randn(1, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (1, 196, 256)

@pytest.mark.parametrize("dim", [256, 512, 768])
def test_different_embed_dims(dim):
    encoder = MockViTEncoder(embed_dim=dim)
    x = torch.randn(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 196, dim)
