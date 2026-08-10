from typing import Any

import torch
import torch.nn as nn

from rl_controller.dreamer.actor_critic import Actor, Critic
from rl_controller.dreamer.rssm import RSSM
from rl_controller.policy.action_space import ActionSpace


class DreamerV3Agent(nn.Module):
    """
    Main DreamerV3 Agent integrating the World Model (RSSM) and Actor-Critic.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

        # Dimensions
        self.embed_dim = config.get("embed_dim", 512)
        self.deter_dim = config.get("deter_dim", 1024)
        self.stoch_dim = config.get("stoch_dim", 32)
        self.num_classes = config.get("num_classes", 32)
        self.action_dim = config.get("action_dim", 3)

        # Core components
        self.rssm = RSSM(
            deter_dim=self.deter_dim,
            stoch_dim=self.stoch_dim,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            action_dim=self.action_dim,
        )

        feat_dim = self.deter_dim + (self.stoch_dim * self.num_classes)
        self.actor = Actor(feat_dim=feat_dim, action_dim=self.action_dim)
        self.critic = Critic(feat_dim=feat_dim)

        # Action space for decoding network outputs
        self.action_space = ActionSpace(config.get("vehicle_mode", "robotaxi"))

    def policy(
        self,
        obs_embed: torch.Tensor,
        prev_state: dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        explore: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute action given an observation embedding from JEPA.
        Returns continuous action and updated state.
        """
        # 1. Update world model state with new observation
        _, post_state = self.rssm.obs_step(prev_state, prev_action, obs_embed)

        # 2. Extract features
        stoch = post_state["stoch"]
        deter = post_state["deter"]
        feat = self.rssm.get_feat(stoch, deter)

        # 3. Get action from actor
        action_dist = self.actor(feat)
        if explore:
            action = action_dist.sample()
        else:
            action = action_dist.mode

        return action, post_state

    def imagine_trajectory(
        self, start_state: dict[str, torch.Tensor], horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rollout the world model to imagine future states and actions.
        Returns features, actions, and values.
        """
        features, actions = [], []
        state = start_state

        for _ in range(horizon):
            # 1. Get action for current state
            feat = self.rssm.get_feat(state["stoch"], state["deter"])
            action = self.actor(feat).sample()

            # 2. Predict next state
            state = self.rssm.img_step(state, action)
            next_feat = self.rssm.get_feat(state["stoch"], state["deter"])

            features.append(next_feat)
            actions.append(action)

        features = torch.stack(features, dim=1)
        actions = torch.stack(actions, dim=1)

        # Evaluate states with critic
        values = self.critic(features).mode

        return features, actions, values
