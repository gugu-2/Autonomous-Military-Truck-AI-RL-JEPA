"""Actor-Critic networks for DreamerV3 driving policy."""

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def lambda_return(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.997, lambda_: float = 0.95
) -> torch.Tensor:
    """Computes full lambda-return for DreamerV3."""
    returns = torch.zeros_like(values)
    last_return = values[-1]
    for t in reversed(range(rewards.shape[0])):
        next_val = values[t + 1] if t + 1 < values.shape[0] else values[-1]
        ret = rewards[t] + gamma * ((1 - lambda_) * next_val + lambda_ * last_return)
        returns[t] = ret
        last_return = ret
    return returns


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes: list[int], activation=nn.SiLU):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(activation())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden_sizes=[512, 512, 512], min_std=0.1):
        super().__init__()
        self.min_std = min_std
        self.net = MLP(feat_dim, action_dim * 2, hidden_sizes)

    def forward(self, feat: torch.Tensor) -> dist.Distribution:
        out = self.net(feat)
        mean, std = torch.chunk(out, 2, dim=-1)
        mean = torch.tanh(mean)  # Squashed mean
        std = F.softplus(std) + self.min_std

        # Normal distribution wrapped in independent (diagonal)
        base_dist = dist.Normal(mean, std)
        distribution = dist.Independent(base_dist, 1)
        # Store squashed mean for deterministic action selection
        distribution._squashed_mean = mean
        return distribution

    def log_prob(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(feat)
        # Inverse tanh trick for exact log prob of squashed action
        raw_action = torch.atanh(torch.clamp(action, -0.999999, 0.999999))
        log_prob = distribution.log_prob(raw_action)
        log_prob -= torch.sum(torch.log(1.0 - action**2 + 1e-6), dim=-1)
        return log_prob


class Critic(nn.Module):
    def __init__(self, feat_dim: int, num_bins=255, lower=-20, upper=20):
        super().__init__()
        self.num_bins = num_bins
        self.lower = lower
        self.upper = upper
        self.net = MLP(feat_dim, num_bins, hidden_sizes=[512, 512, 512])
        self.register_buffer("bins", torch.linspace(lower, upper, num_bins))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)

    def expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bins.to(logits.device)).sum(dim=-1, keepdim=True)

    def two_hot_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Vectorized two-hot encoding
        bins = self.bins.to(targets.device)
        below = (torch.bucketize(targets, bins) - 1).clamp(0, len(bins) - 2)
        above = below + 1
        lower = bins[below]
        upper = bins[above]
        # Linear interpolation weights
        upper_weight = ((targets - lower) / (upper - lower + 1e-8)).clamp(0, 1)
        lower_weight = 1.0 - upper_weight
        # Build two-hot target
        two_hot = torch.zeros(*targets.shape, len(bins), device=targets.device)
        two_hot.scatter_(-1, below.unsqueeze(-1), lower_weight.unsqueeze(-1))
        two_hot.scatter_(-1, above.unsqueeze(-1), upper_weight.unsqueeze(-1))
        # Cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        return -(two_hot * log_probs).sum(dim=-1).mean()
