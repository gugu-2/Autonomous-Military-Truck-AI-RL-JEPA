from typing import Any

import torch
import torch.nn as nn


class ImaginationTrainer:
    """
    Trains the DreamerV3 agent in the latent imagination of the world model.
    """

    def __init__(self, agent: nn.Module, config: dict[str, Any]):
        self.agent = agent
        self.config = config

        self.horizon = config.get("imagination_horizon", 15)
        self.gamma = config.get("gamma", 0.997)
        self.lmbda = config.get("lmbda", 0.95)

        # Optimizers
        self.opt_rssm = torch.optim.AdamW(
            self.agent.rssm.parameters(), lr=config.get("rssm_lr", 1e-4)
        )
        self.opt_actor = torch.optim.AdamW(
            self.agent.actor.parameters(), lr=config.get("actor_lr", 3e-5)
        )
        self.opt_critic = torch.optim.AdamW(
            self.agent.critic.parameters(), lr=config.get("critic_lr", 3e-5)
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.get("use_amp", True))

    def _compute_lambda_returns(
        self, rewards: torch.Tensor, values: torch.Tensor, continues: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lambda-returns for Actor-Critic training.
        """
        last_val = values[:, -1]
        returns = []
        for t in reversed(range(self.horizon)):
            next_val = values[:, t + 1] if t + 1 < self.horizon else last_val
            ret = rewards[:, t] + self.gamma * continues[:, t] * (
                (1 - self.lmbda) * next_val + self.lmbda * last_val
            )
            last_val = ret
            returns.insert(0, ret)
        return torch.stack(returns, dim=1)

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
    ) -> dict[str, float]:
        """
        Execute a single training step using a batch of trajectories from the replay buffer.
        """
        batch_size, seq_len = obs.shape[:2]
        device = obs.device
        metrics = {}

        # 1. World Model (RSSM) Update
        # ----------------------------
        # In a real implementation, we unroll the RSSM over the sequence and compute KL divergence
        # between prior (dynamics) and posterior (encoder) states.
        # For brevity in this architectural template:

        with torch.cuda.amp.autocast(enabled=self.config.get("use_amp", True)):
            # Dummy loss for template
            rssm_loss = torch.tensor(0.5, requires_grad=True, device=device)
            metrics["rssm_loss"] = rssm_loss.item()

        self.opt_rssm.zero_grad()
        self.scaler.scale(rssm_loss).backward()
        self.scaler.unscale_(self.opt_rssm)
        torch.nn.utils.clip_grad_norm_(self.agent.rssm.parameters(), 100.0)
        self.scaler.step(self.opt_rssm)

        # 2. Actor-Critic (Imagination) Update
        # ------------------------------------
        # We sample a start state from the posterior states computed above,
        # and roll out the policy in the world model for `self.horizon` steps.

        # Get start states from flattened posterior (mocking with zeros)
        start_state = self.agent.rssm.initial(batch_size * seq_len, device)

        with torch.cuda.amp.autocast(enabled=self.config.get("use_amp", True)):
            # Rollout in imagination
            imag_features, imag_actions, imag_values = self.agent.imagine_trajectory(
                start_state, self.horizon
            )

            # Predict rewards and continues from imagined features
            # Mocking reward predictor
            imag_rewards = torch.zeros((batch_size * seq_len, self.horizon), device=device)
            imag_continues = torch.ones((batch_size * seq_len, self.horizon), device=device)

            # Compute targets
            returns = self._compute_lambda_returns(imag_rewards, imag_values, imag_continues)

            # Actor loss (Maximize returns)
            actor_loss = -returns.mean()
            metrics["actor_loss"] = actor_loss.item()

            # Critic loss (Minimize cross-entropy to two-hot targets)
            critic_logits = self.agent.critic(imag_features)
            critic_loss = self.agent.critic.two_hot_loss(critic_logits, returns.detach())
            metrics["critic_loss"] = critic_loss.item()

        # Update Actor
        self.opt_actor.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.opt_actor)
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 100.0)
        self.scaler.step(self.opt_actor)

        # Update Critic
        self.opt_critic.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.opt_critic)
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 100.0)
        self.scaler.step(self.opt_critic)

        self.scaler.update()

        return metrics
