import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class WorldModelRLWrapper(nn.Module):
    """
    Wrapper that connects the frozen JEPA World Model to the RL RSSM dynamics.
    The JEPA model handles vision -> embedding, and RSSM handles dynamics.
    """
    def __init__(self, jepa_model: nn.Module, rssm: nn.Module):
        super().__init__()
        self.jepa_model = jepa_model
        self.rssm = rssm
        
        # Ensure JEPA is frozen
        for param in self.jepa_model.parameters():
            param.requires_grad = False
            
    def encode_observation(self, camera_frames: torch.Tensor, lidar_bev: torch.Tensor = None) -> torch.Tensor:
        """
        Encode raw sensors into the latent state representation `s_t` using JEPA.
        """
        with torch.no_grad():
            # In actual implementation, JEPA handles multi-modal fusion.
            # Assuming camera_frames is shape (B, T, C, H, W)
            # Output is an embedding vector (B, embed_dim)
            if hasattr(self.jepa_model, 'encode_for_rl'):
                embed = self.jepa_model.encode_for_rl(camera_frames, lidar_bev)
            else:
                # Fallback for mock models
                # Take last frame, pass through encoder
                last_frame = camera_frames[:, -1]
                if hasattr(self.jepa_model, 'encoder'):
                    x = self.jepa_model.encoder(last_frame)
                    # Global average pooling if it outputs spatial tokens
                    if x.ndim == 3:
                        embed = x.mean(dim=1)
                    else:
                        embed = x
                else:
                    embed = torch.randn(camera_frames.size(0), self.rssm.embed_dim, device=camera_frames.device)
        return embed

    def forward(self, obs: torch.Tensor, prev_state: Dict[str, torch.Tensor], prev_action: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass: Encode observation -> Update RSSM state.
        """
        embed = self.encode_observation(obs)
        prior, post = self.rssm.obs_step(prev_state, prev_action, embed)
        return embed, post
