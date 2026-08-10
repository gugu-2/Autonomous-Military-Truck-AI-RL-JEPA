import torch
from typing import List

class LatentTrajectory:
    """Stores the predicted sequence of future latent states."""
    def __init__(self):
        self.trajectory: List[torch.Tensor] = []
        
    def append(self, latent_state: torch.Tensor):
        self.trajectory.append(latent_state)
        
    def get_sequence(self) -> torch.Tensor:
        if not self.trajectory:
            return torch.empty(0)
        return torch.stack(self.trajectory, dim=1)
