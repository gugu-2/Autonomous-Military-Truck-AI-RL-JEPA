import torch
import numpy as np
from typing import Any

class CameraPreprocessor:
    """Normalizes and resizes camera streams for the ViT encoder."""
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Converts raw RGB numpy array to normalized PyTorch tensor."""
        # Normalize and rearrange to (C, H, W)
        tensor = torch.from_numpy(frame).float() / 255.0
        return tensor.permute(2, 0, 1)
