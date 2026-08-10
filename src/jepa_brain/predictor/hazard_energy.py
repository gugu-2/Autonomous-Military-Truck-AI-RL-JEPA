"""Spatial Anomaly Energy Discrepancy — computes per-region hazard scores."""

from enum import Enum

import torch


class HazardLevel(Enum):
    CLEAR = 0
    LOW = 1
    WARN = 2
    CRITICAL = 3


class HazardEnergyComputer:
    def __init__(self, veto_threshold: float = 0.8, emergency_threshold: float = 1.5):
        self.veto_threshold = veto_threshold
        self.emergency_threshold = emergency_threshold

    def compute(
        self, s_target: torch.Tensor, s_hat: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Computes normalized L2 energy: E = ||s_target - s_hat||^2 / (||s_target||^2 + eps).
        Input shapes: both (B, N, D). Output: (B, N) per-token energy.
        """
        diff_sq = torch.sum((s_target - s_hat) ** 2, dim=-1)
        target_sq = torch.sum(s_target**2, dim=-1)
        energy = diff_sq / (target_sq + eps)
        return energy

    def compute_spatial_map(self, energy: torch.Tensor, grid_size: int = 16) -> torch.Tensor:
        """Reshapes token energies to spatial grid (B, H, W)."""
        B, N = energy.shape
        assert N == grid_size * grid_size, f"Expected {grid_size*grid_size} tokens, got {N}"
        spatial_map = energy.view(B, grid_size, grid_size)
        return spatial_map

    def get_max_hazard(self, energy_map: torch.Tensor) -> tuple[float, int, int]:
        """Returns max energy + location."""
        if energy_map.dim() == 3:
            energy_map = energy_map[0]
        max_val = torch.max(energy_map).item()
        max_idx = torch.argmax(energy_map)
        h = (max_idx // energy_map.size(-1)).item()
        w = (max_idx % energy_map.size(-1)).item()
        return max_val, h, w

    def classify_hazard(self, energy: float) -> HazardLevel:
        if energy >= self.emergency_threshold:
            return HazardLevel.CRITICAL
        elif energy >= self.veto_threshold:
            return HazardLevel.WARN
        elif energy >= self.veto_threshold * 0.5:
            return HazardLevel.LOW
        return HazardLevel.CLEAR

    def should_veto(self, energy_map: torch.Tensor) -> bool:
        """Returns True if any region >= veto_threshold"""
        return torch.any(energy_map >= self.veto_threshold).item()

    def should_emergency_brake(self, energy_map: torch.Tensor) -> bool:
        """Returns True if any region >= emergency_threshold"""
        return torch.any(energy_map >= self.emergency_threshold).item()
