"""Imagination Engine — simulates 3 seconds of future driving before acting."""
import torch
from dataclasses import dataclass
from typing import List

@dataclass
class DrivingAction:
    action_id: int
    vector: torch.Tensor

@dataclass
class LatentState:
    tokens: torch.Tensor

@dataclass
class ImaginationResult:
    candidate_actions: List[DrivingAction]
    energy_maps: torch.Tensor 
    is_safe: List[bool]
    best_action: DrivingAction

class ImaginationEngine:
    def __init__(self, jepa_predictor, hazard_computer, config):
        self.jepa_predictor = jepa_predictor
        self.hazard_computer = hazard_computer
        self.config = config
        self.grid_size = config.get("grid_size", 16)
        
    def imagine(self, current_latent: LatentState, candidate_actions: List[DrivingAction]) -> ImaginationResult:
        """
        Simulate futures for all candidates and compute hazard maps.
        """
        num_candidates = len(candidate_actions)
        B, N, D = current_latent.tokens.shape
        
        context_batch = current_latent.tokens.repeat(num_candidates, 1, 1) # (B*num_candidates, N, D)
        
        action_vectors = torch.stack([a.vector for a in candidate_actions], dim=0) # (num_candidates, action_dim)
        action_vectors = action_vectors.repeat(B, 1)
        
        # Predict K steps ahead
        predictions = self.jepa_predictor(context_batch, action_vectors)
        _, K, _, _ = predictions.shape
        
        # Target representation usually comes from an invariant or ground truth. 
        # Using a surrogate zeroes tensor for imagination anomaly evaluation here.
        s_target = torch.zeros_like(predictions)
        
        flat_preds = predictions.view(B * num_candidates * K, N, D)
        flat_targets = s_target.view(B * num_candidates * K, N, D)
        
        energies = self.hazard_computer.compute(flat_targets, flat_preds)
        spatial_maps = self.hazard_computer.compute_spatial_map(energies, self.grid_size)
        
        spatial_maps = spatial_maps.view(num_candidates, B, K, self.grid_size, self.grid_size)
        energy_maps = spatial_maps[:, 0, :, :, :] # Take B=0
        
        is_safe = []
        best_action = None
        min_max_energy = float('inf')
        
        for i in range(num_candidates):
            candidate_energy = energy_maps[i] # (K, H, W)
            
            safe = not self.hazard_computer.should_veto(candidate_energy)
            is_safe.append(safe)
            
            max_energy = torch.max(candidate_energy).item()
            if max_energy < min_max_energy:
                min_max_energy = max_energy
                best_action = candidate_actions[i]
                
        return ImaginationResult(
            candidate_actions=candidate_actions,
            energy_maps=energy_maps,
            is_safe=is_safe,
            best_action=best_action
        )
        
    def get_safe_trajectories(self, result: ImaginationResult) -> List[DrivingAction]:
        return [act for act, safe in zip(result.candidate_actions, result.is_safe) if safe]
