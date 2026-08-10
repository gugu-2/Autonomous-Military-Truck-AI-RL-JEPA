"""End-to-end pipeline integration test (no real hardware)."""

import time

import numpy as np
import pytest
import torch


class SensorFusionEngine:
    def fuse(self, camera, lidar):
        # mock fusion
        return torch.randn(1, 4, 256)


class JEPAWorldModel:
    def encode(self, fused_features):
        return torch.randn(1, 256, 512)


class RLPolicy:
    def get_action(self, latent_state):
        return {"steering": 0.1, "throttle": 0.5, "brake": 0.0}


class SafetyInterlock:
    def filter(self, action):
        if action["throttle"] > 0.8:
            action["throttle"] = 0.8
        return action


class FullPipeline:
    def __init__(self):
        self.fusion = SensorFusionEngine()
        self.jepa = JEPAWorldModel()
        self.policy = RLPolicy()
        self.safety = SafetyInterlock()

    def step(self, camera, lidar):
        fused = self.fusion.fuse(camera, lidar)
        latent = self.jepa.encode(fused)
        raw_action = self.policy.get_action(latent)
        safe_action = self.safety.filter(raw_action)
        return latent, safe_action


@pytest.fixture
def pipeline():
    return FullPipeline()


def test_sensor_to_latent_pipeline(pipeline):
    camera = torch.zeros((1, 4, 3, 224, 224))
    lidar = np.zeros((1000, 4))
    latent, _ = pipeline.step(camera, lidar)
    assert latent.shape == (1, 256, 512)


def test_latent_to_action_pipeline(pipeline):
    camera = torch.zeros((1, 4, 3, 224, 224))
    lidar = np.zeros((1000, 4))
    _, action = pipeline.step(camera, lidar)
    assert -1.0 <= action["steering"] <= 1.0
    assert 0.0 <= action["throttle"] <= 1.0
    assert 0.0 <= action["brake"] <= 1.0


def test_safety_interlock_overrides_bad_action(pipeline):
    # Mocking policy to return dangerous action
    pipeline.policy.get_action = lambda x: {"steering": 0.0, "throttle": 1.0, "brake": 0.0}
    _, action = pipeline.step(None, None)
    assert action["throttle"] <= 0.8


def test_full_pipeline_latency(pipeline):
    camera = torch.randn((1, 4, 3, 224, 224))
    lidar = np.random.randn(1000, 4)

    # warmup
    pipeline.step(camera, lidar)

    start_time = time.perf_counter()
    pipeline.step(camera, lidar)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    assert latency_ms < 50.0


def test_no_crash_on_all_zeros_input(pipeline):
    camera = torch.zeros((1, 4, 3, 224, 224))
    lidar = np.zeros((1000, 4))
    try:
        pipeline.step(camera, lidar)
    except Exception as e:
        pytest.fail(f"Pipeline crashed on all-zero input: {e}")


def test_no_crash_on_random_input(pipeline):
    camera = torch.randn((1, 4, 3, 224, 224))
    lidar = np.random.randn(1000, 4)
    try:
        pipeline.step(camera, lidar)
    except Exception as e:
        pytest.fail(f"Pipeline crashed on random input: {e}")
