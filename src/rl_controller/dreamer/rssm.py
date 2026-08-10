"""Recurrent State Space Model (RSSM) — core latent dynamics model for DreamerV3."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RSSMState:
    deter: torch.Tensor  # Deterministic state (B, deter_dim)
    stoch: torch.Tensor  # Stochastic state (B, num_classes, num_classes) or (B, stoch_dim)
    logits: torch.Tensor  # Logits for categorical distribution
    mean: torch.Tensor  # Used if continuous
    std: torch.Tensor  # Used if continuous


class RSSM(nn.Module):
    def __init__(self, deter_dim=512, stoch_dim=32, num_classes=32, embed_dim=512, action_dim=3):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.num_classes = num_classes

        # RNN for deterministic state
        self.cell = nn.GRUCell(self.stoch_dim * self.num_classes + action_dim, self.deter_dim)

        # Prior network (predicts next stoch from deter)
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.deter_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, self.stoch_dim * self.num_classes),
        )

        # Posterior network (predicts stoch from deter + observation embed)
        self.post_mlp = nn.Sequential(
            nn.Linear(self.deter_dim + embed_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, self.stoch_dim * self.num_classes),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, self.num_classes, device=device),
            logits=torch.zeros(batch_size, self.stoch_dim, self.num_classes, device=device),
            mean=torch.zeros(batch_size, self.stoch_dim, device=device),
            std=torch.zeros(batch_size, self.stoch_dim, device=device),
        )

    def observe(
        self, embed: torch.Tensor, action: torch.Tensor, state: RSSMState
    ) -> tuple[RSSMState, RSSMState]:
        """Steps the RSSM with observations. Returns posterior and prior states."""
        prior = self.imagine(action, state)
        logits = self.post_mlp(torch.cat([prior.deter, embed], dim=-1))
        logits = logits.view(-1, self.stoch_dim, self.num_classes)
        stoch = self._straight_through_sample(logits)
        posterior = RSSMState(
            deter=prior.deter,
            stoch=stoch,
            logits=logits,
            mean=torch.zeros_like(stoch),
            std=torch.ones_like(stoch),
        )
        return posterior, prior

    def imagine(self, action: torch.Tensor, state: RSSMState) -> RSSMState:
        """Rolls out prior dynamics without observations."""
        flat_stoch = state.stoch.reshape(-1, self.stoch_dim * self.num_classes)
        x = torch.cat([flat_stoch, action], dim=-1)
        deter = self.cell(x, state.deter)

        logits = self.prior_mlp(deter)
        logits = logits.view(-1, self.stoch_dim, self.num_classes)
        stoch = self._straight_through_sample(logits)
        return RSSMState(
            deter=deter,
            stoch=stoch,
            logits=logits,
            mean=torch.zeros_like(stoch),
            std=torch.ones_like(stoch),
        )

    def _prior(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.prior_mlp(h)
        logits = logits.view(-1, self.stoch_dim, self.num_classes)
        stoch = self._straight_through_sample(logits)
        return logits, torch.zeros_like(stoch), stoch

    def _posterior(
        self, h: torch.Tensor, embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.post_mlp(torch.cat([h, embed], dim=-1))
        logits = logits.view(-1, self.stoch_dim, self.num_classes)
        stoch = self._straight_through_sample(logits)
        return logits, torch.zeros_like(stoch), stoch

    def _straight_through_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Straight-Through estimator for categorical distributions."""
        dist = torch.distributions.OneHotCategorical(logits=logits)
        stoch = dist.sample()
        stoch = stoch + dist.probs - dist.probs.detach()  # Straight-through gradient
        return stoch

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        """Concatenates deterministic and stochastic states for actor/critic."""
        flat_stoch = state.stoch.reshape(state.stoch.shape[0], -1)
        return torch.cat([state.deter, flat_stoch], dim=-1)

    def kl_loss(
        self, post_logits: torch.Tensor, prior_logits: torch.Tensor, free_nats: float = 3.0
    ) -> torch.Tensor:
        """Computes KL divergence with free nats clipping."""
        post = torch.distributions.Independent(
            torch.distributions.OneHotCategorical(logits=post_logits), 1
        )
        prior = torch.distributions.Independent(
            torch.distributions.OneHotCategorical(logits=prior_logits), 1
        )
        kl = torch.distributions.kl.kl_divergence(post, prior)
        kl = torch.max(kl, torch.tensor(free_nats, device=kl.device))
        return kl.mean()
