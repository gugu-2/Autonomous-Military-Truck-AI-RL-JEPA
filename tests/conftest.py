"""Shared pytest fixtures for OMNIDRIVE test suite."""
import pytest
import torch
import numpy as np
from omegaconf import OmegaConf

# Dummy classes to act as placeholders for actual OMNIDRIVE modules if they are not importable
class UnifiedWorldState:
    def __init__(self):
        self.camera_data = torch.zeros((4, 3, 224, 224))
        self.lidar_data = np.zeros((1000, 4))
        self.ego_speed = 0.0

class LatentState:
    def __init__(self, tensor):
        self.tokens = tensor

class DrivingAction:
    def __init__(self, steering, throttle, brake):
        self.steering = steering
        self.throttle = throttle
        self.brake = brake

class PatchEmbedder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, config.encoder.embed_dim, kernel_size=16, stride=16)
    def forward(self, x):
        return self.proj(x)

class ViTEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbedder(config)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(config.encoder.embed_dim, config.encoder.embed_dim) for _ in range(1)])
        self.norm = torch.nn.LayerNorm(config.encoder.embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class HazardEnergyComputer:
    def __init__(self, config):
        self.threshold = config.safety.hazard_threshold
        self.warn_threshold = config.safety.warn_threshold
        self.veto_threshold = config.safety.veto_threshold
        
    def compute(self, s_hat, s_target):
        return torch.mean((s_hat - s_target) ** 2, dim=-1)

@pytest.fixture(scope='function')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(scope='function')
def batch_size():
    return 2

@pytest.fixture(scope='function')
def dummy_camera_frames(batch_size):
    return torch.rand((batch_size, 4, 3, 224, 224))

@pytest.fixture(scope='function')
def dummy_lidar_points():
    points = np.random.rand(1000, 4) * 100
    points[:, 3] = np.random.rand(1000) # intensity
    return points

@pytest.fixture(scope='function')
def dummy_world_state():
    state = UnifiedWorldState()
    state.ego_speed = 15.0
    return state

@pytest.fixture(scope='function')
def dummy_latent_state(device):
    return LatentState(torch.rand((256, 512), device=device))

@pytest.fixture(scope='function')
def dummy_action():
    return DrivingAction(steering=0.0, throttle=0.5, brake=0.0)

@pytest.fixture(scope='function')
def dummy_config():
    return OmegaConf.create({
        "encoder": {"embed_dim": 256, "patch_size": 16},
        "safety": {"hazard_threshold": 0.2, "warn_threshold": 0.5, "veto_threshold": 0.7}
    })

@pytest.fixture(scope='function')
def robotaxi_config(dummy_config):
    cfg = dummy_config.copy()
    cfg.mode = "robotaxi"
    return cfg

@pytest.fixture(scope='function')
def truck_config(dummy_config):
    cfg = dummy_config.copy()
    cfg.mode = "truck"
    return cfg

@pytest.fixture(scope='function')
def military_config(dummy_config):
    cfg = dummy_config.copy()
    cfg.mode = "military"
    return cfg

@pytest.fixture(scope='function')
def patch_embedder(dummy_config, device):
    return PatchEmbedder(dummy_config).to(device)

@pytest.fixture(scope='function')
def vit_encoder(dummy_config, device):
    return ViTEncoder(dummy_config).to(device)

@pytest.fixture(scope='function')
def hazard_computer(dummy_config):
    return HazardEnergyComputer(dummy_config)
