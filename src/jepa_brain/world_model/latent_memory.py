import torch
from collections import deque

class LatentMemory:
    """Stores a rolling window of recent latent states for temporal context."""
    def __init__(self, maxlen=10):
        self.memory = deque(maxlen=maxlen)
        
    def add(self, state: torch.Tensor):
        self.memory.append(state)
        
    def get_context(self) -> torch.Tensor:
        if not self.memory:
            return torch.empty(0)
        return torch.stack(list(self.memory), dim=1)
