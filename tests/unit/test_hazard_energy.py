"""Unit tests for Hazard Energy computation."""

import pytest
import torch

try:
    from jepa_brain.predictor.hazard_energy import HazardEnergyComputer
    from jepa_brain.world_model.trajectory_veto import TrajectoryVetoSystem

    HAS_JEPA = True
except ImportError:
    HAS_JEPA = False


@pytest.mark.skipif(not HAS_JEPA, reason="jepa_brain not importable")
def test_hazard_energy_perfect_prediction():
    """Energy must be 0 when prediction perfectly matches target."""
    computer = HazardEnergyComputer()
    s_target = torch.ones(2, 256, 512) * 2.0
    s_hat = torch.ones(2, 256, 512) * 2.0  # perfect match
    energy = computer.compute(s_target, s_hat)
    assert torch.allclose(energy, torch.zeros_like(energy), atol=1e-5)


@pytest.mark.skipif(not HAS_JEPA, reason='jepa_brain not importable')
def test_trajectory_veto_triggers_above_threshold():
    """TrajectoryVetoSystem must veto when energy exceeds threshold."""
    veto = TrajectoryVetoSystem(veto_threshold=0.70)
    # evaluate() returns string: 'VETO', 'CAUTION', or 'SAFE'
    high_energy = 0.85
    assert veto.evaluate(high_energy) == 'VETO', "Energy 0.85 > threshold 0.70 must return VETO"

    low_energy = 0.30
    assert veto.evaluate(low_energy) == 'SAFE', "Energy 0.30 < threshold must return SAFE"
