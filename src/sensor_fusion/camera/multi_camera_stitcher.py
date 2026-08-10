import torch


class MultiCameraStitcher:
    """Combines multiple camera feeds into a single batched tensor."""

    def stitch(self, processed_frames: list[torch.Tensor]) -> torch.Tensor:
        """Stacks frames into shape (N_cams, C, H, W)."""
        return torch.stack(processed_frames, dim=0)
