import torch

from rl_controller.dreamer.dreamerv3_agent import DreamerV3Agent
from rl_controller.dreamer.world_model_rl import WorldModelRLWrapper
from rl_controller.safety.safety_interlock import SafetyInterlock


class DrivingPolicy:
    """
    High-level policy interface for deploying the trained RL agent in the real world.
    Manages state persistence, action decoding, and safety interlocks.
    """

    def __init__(
        self,
        agent: DreamerV3Agent,
        world_model_wrapper: WorldModelRLWrapper,
        safety_interlock: SafetyInterlock,
        device: str = "cuda",
    ):
        self.agent = agent
        self.world_model = world_model_wrapper
        self.safety = safety_interlock
        self.device = device

        self.agent.eval()
        self.world_model.eval()

        self.reset()

    def reset(self):
        """Reset the internal recurrent state of the policy."""
        self.prev_state = self.agent.rssm.initial(1)
        self.prev_action = torch.zeros(1, self.agent.action_dim, device=self.device)

    @torch.no_grad()
    def get_action(
        self,
        camera_frames: torch.Tensor,
        lidar_bev: torch.Tensor = None,
        hazard_map: torch.Tensor = None,
    ) -> dict[str, float]:
        """
        Execute a single driving step.
        1. Encode sensors to latent state
        2. Get raw RL action
        3. Pass through safety interlock
        4. Decode to physical vehicle controls
        """
        # Ensure batch dim
        if camera_frames.ndim == 4:
            camera_frames = camera_frames.unsqueeze(0)

        # 1. Encode observations
        obs_embed, self.prev_state = self.world_model(
            camera_frames, self.prev_state, self.prev_action
        )

        # 2. Get raw policy action (deterministic for deployment)
        raw_action, _ = self.agent.policy(
            obs_embed, self.prev_state, self.prev_action, explore=False
        )
        self.prev_action = raw_action

        # 3. Decode action space mapping
        decoded_action = self.agent.action_space.decode(raw_action)

        # 4. Apply safety interlock (vetos / overrides based on hazard energy)
        if hazard_map is not None:
            max_hazard = float(hazard_map.max())
            if max_hazard > 0.70:
                decoded_action = self.safety.trigger_emergency_brake(
                    decoded_action, reason=f"Hazard E={max_hazard:.2f}"
                )
            elif max_hazard > 0.45:
                decoded_action = self.safety.apply_caution_limits(decoded_action)

        return decoded_action
