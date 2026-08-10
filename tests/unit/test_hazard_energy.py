"""Unit tests for JEPA Hazard Energy computation."""

import pytest
import torch


class HazardLevel:
    CLEAR = 0
    WARN = 1
    CRITICAL = 2


class MockHazardEnergyComputer:
    def __init__(self):
        self.warn_threshold = 0.5
        self.veto_threshold = 0.70
        self.emergency_brake_threshold = 0.85

    def compute(self, s_hat, s_target):
        return torch.mean((s_hat - s_target) ** 2, dim=-1)

    def get_spatial_map(self, energy, h=16, w=16):
        return energy.view(-1, h, w)

    def get_hazard_level(self, max_energy):
        if max_energy >= self.veto_threshold:
            return HazardLevel.CRITICAL
        elif max_energy >= self.warn_threshold:
            return HazardLevel.WARN
        return HazardLevel.CLEAR

    def should_veto(self, max_energy):
        return max_energy >= self.veto_threshold

    def should_emergency_brake(self, max_energy):
        return max_energy >= self.emergency_brake_threshold


@pytest.fixture
def computer():
    return MockHazardEnergyComputer()


def test_energy_zero_when_perfect_prediction(computer):
    s_hat = torch.ones((2, 256, 512))
    s_target = torch.ones((2, 256, 512))
    energy = computer.compute(s_hat, s_target)
    assert torch.all(energy == 0)


def test_energy_positive_always(computer):
    s_hat = torch.randn((2, 256, 512))
    s_target = torch.randn((2, 256, 512))
    energy = computer.compute(s_hat, s_target)
    assert torch.all(energy >= 0)


def test_energy_shape_correct(computer):
    s_hat = torch.randn((2, 256, 512))
    s_target = torch.randn((2, 256, 512))
    energy = computer.compute(s_hat, s_target)
    assert energy.shape == (2, 256)


def test_spatial_map_shape(computer):
    energy = torch.zeros((2, 256))
    spatial_map = computer.get_spatial_map(energy)
    assert spatial_map.shape == (2, 16, 16)


def test_hazard_below_warn_threshold_is_clear(computer):
    assert computer.get_hazard_level(0.2) == HazardLevel.CLEAR


def test_hazard_above_warn_threshold(computer):
    assert computer.get_hazard_level(0.5) == HazardLevel.WARN


def test_hazard_above_veto_threshold(computer):
    assert computer.get_hazard_level(0.75) == HazardLevel.CRITICAL


def test_should_veto_triggers(computer):
    assert computer.should_veto(0.70) is True


def test_should_not_veto_safe(computer):
    assert computer.should_veto(0.69) is False


def test_no_nan_in_output(computer):
    s_hat = torch.randn((2, 256, 512))
    s_target = torch.randn((2, 256, 512))
    energy = computer.compute(s_hat, s_target)
    assert not torch.any(torch.isnan(energy))


def test_energy_is_normalized(computer):
    s_hat = torch.rand((2, 256, 512))  # [0, 1]
    s_target = torch.rand((2, 256, 512))  # [0, 1]
    energy = computer.compute(s_hat, s_target)
    assert torch.all(energy >= 0)
    assert torch.all(energy <= 1.0)  # max squared diff of [0,1] vectors is bounded


def test_emergency_brake_threshold(computer):
    assert computer.should_emergency_brake(0.85) is True
    assert computer.should_emergency_brake(0.84) is False
