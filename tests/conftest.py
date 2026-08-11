"""Shared pytest fixtures for OMNIDRIVE test suite."""

import numpy as np
import pytest
import torch
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
    def __init__(self, steering_angle=0.0, throttle=0.0, brake=0.0, gear=1):
        self.steering_angle = steering_angle
        self.throttle = throttle
        self.brake = brake
        self.gear = gear


try:
    from jepa_brain.encoder.patch_embedder import PatchEmbedder
    from jepa_brain.encoder.vit_encoder import ViTEncoder
    from jepa_brain.predictor.hazard_energy import HazardEnergyComputer
except ImportError:
    # Fallback stubs if torch not installed
    class PatchEmbedder:
        pass

    class ViTEncoder:
        pass

    class HazardEnergyComputer:
        def __init__(self, config=None):
            if config:
                self.threshold = config.safety.hazard_threshold
                self.warn_threshold = config.safety.warn_threshold
                self.veto_threshold = config.safety.veto_threshold
            else:
                self.threshold = 0.7

        def compute(self, s_target, s_hat):
            diff_sq = torch.sum((s_target - s_hat) ** 2, dim=-1)
            target_sq = torch.sum(s_target**2, dim=-1)
            eps = 1e-6
            return diff_sq / (target_sq + eps)


@pytest.fixture(scope="function")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="function")
def batch_size():
    return 2


@pytest.fixture(scope="function")
def dummy_camera_frames(batch_size):
    return torch.rand((batch_size, 4, 3, 224, 224))


@pytest.fixture(scope="function")
def dummy_lidar_points():
    points = np.random.rand(1000, 4) * 100
    points[:, 3] = np.random.rand(1000)  # intensity
    return points


@pytest.fixture(scope="function")
def dummy_world_state():
    state = UnifiedWorldState()
    state.ego_speed = 15.0
    return state


@pytest.fixture(scope="function")
def dummy_latent_state(device):
    return LatentState(torch.rand((256, 512), device=device))


@pytest.fixture(scope="function")
def dummy_action():
    return DrivingAction(steering_angle=0.0, throttle=0.5, brake=0.0, gear=1)


@pytest.fixture(scope="function")
def dummy_config():
    return OmegaConf.create(
        {
            "encoder": {"embed_dim": 256, "patch_size": 16},
            "safety": {"hazard_threshold": 0.2, "warn_threshold": 0.5, "veto_threshold": 0.7},
        }
    )


@pytest.fixture(scope="function")
def robotaxi_config(dummy_config):
    cfg = dummy_config.copy()
    cfg.mode = "robotaxi"
    return cfg



@pytest.fixture(scope="function")
def patch_embedder(dummy_config, device):
    return PatchEmbedder(img_size=224, patch_size=16, in_channels=3, embed_dim=dummy_config.encoder.embed_dim).to(device)


@pytest.fixture(scope="function")
def vit_encoder(dummy_config, device):
    return ViTEncoder(embed_dim=dummy_config.encoder.embed_dim, depth=2, num_heads=4).to(device)


@pytest.fixture(scope="function")
def hazard_computer(dummy_config):
    return HazardEnergyComputer(dummy_config)
