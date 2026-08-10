import torch
from typing import List

class MultiCameraStitcher:
    """Combines multiple camera feeds into a single batched tensor."""
    def stitch(self, processed_frames: List[torch.Tensor]) -> torch.Tensor:
        """Stacks frames into shape (N_cams, C, H, W)."""
        return torch.stack(processed_frames, dim=0)
