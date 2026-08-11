"""Exponential Moving Average target encoder — prevents representation collapse."""

import copy

import torch
import torch.nn as nn


class EMATargetEncoder(nn.Module):
    def __init__(self, online_encoder: nn.Module, momentum: float = 0.996):
        super().__init__()
        self.momentum = momentum
        self.target_encoder = copy.deepcopy(online_encoder)
        object.__setattr__(self, "_online_encoder_ref", online_encoder)

        # Stop gradients for the target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    def update(self):
        """
        EMA update: theta_bar = tau * theta_bar + (1-tau) * theta.
        Does NOT update gradients. Call after every training step.
        """
        with torch.no_grad():
            for online_param, target_param in zip(
                self._online_encoder_ref.parameters(), self.target_encoder.parameters()
            ):
                target_param.data.mul_(self.momentum).add_(
                    online_param.data, alpha=1.0 - self.momentum
                )
            for (name, target_buf), (_, online_buf) in zip(
                self.target_encoder.named_buffers(), self._online_encoder_ref.named_buffers()
            ):
                target_buf.data.mul_(self.momentum).add_(online_buf.data * (1 - self.momentum))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through target encoder (with torch.no_grad())."""
        with torch.no_grad():
            return self.target_encoder(x)

    def set_momentum(self, momentum: float):
        """Dynamic momentum schedule."""
        self.momentum = momentum

    def get_momentum(self) -> float:
        return self.momentum
