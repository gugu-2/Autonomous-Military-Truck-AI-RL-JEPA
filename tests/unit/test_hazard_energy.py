"""Unit tests for Hazard Energy Computation."""

import torch
from jepa_brain.predictor.trajectory_veto import TrajectoryVeto


def test_hazard_energy_calculation():
    """Test the L2 Norm mathematical formula for Hazard Energy."""
    veto_system = TrajectoryVeto(threshold=0.7)

    # s_target is the true world state (EMA encoded)
    # s_hat is the predicted world state
    # Energy E(t+k) = ||s_target - s_hat||^2 / ||s_target||^2

    # Create deterministic tensors for math verification
    s_target = torch.ones(2, 256, 512) * 2.0  # L2 norm per element is 2.0
    s_hat_safe = torch.ones(2, 256, 512) * 2.0  # Perfect match

    # 1. Perfect Prediction = 0 Energy (Safe)
    energy_safe, is_veto_safe = veto_system.evaluate_hazard(s_target, s_hat_safe)
    assert torch.allclose(energy_safe, torch.tensor(0.0)), "Perfect match must yield 0.0 energy"
    assert is_veto_safe is False

    # 2. Complete Mismatch = High Energy (Danger)
    s_hat_danger = torch.ones(2, 256, 512) * 4.0  # Double the value
    # ||2 - 4||^2 = 4
    # ||2||^2 = 4
    # Energy = 4 / 4 = 1.0 (100% mismatch)
    energy_danger, is_veto_danger = veto_system.evaluate_hazard(s_target, s_hat_danger)

    assert torch.allclose(
        energy_danger.mean(), torch.tensor(1.0)
    ), f"Expected 1.0 energy, got {energy_danger.mean()}"
    assert is_veto_danger is True, "Energy of 1.0 must trigger veto (threshold 0.7)"
