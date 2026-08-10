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
        ret = rewards[t] + gamma * ((1 - lambda_) * values[t] + lambda_ * last_return)
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

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, dist.Distribution]:
        out = self.net(feat)
        mean, std = torch.chunk(out, 2, dim=-1)
        mean = torch.tanh(mean)  # Squashed mean
        std = F.softplus(std) + self.min_std

        # Normal distribution wrapped in independent (diagonal)
        base_dist = dist.Normal(mean, std)
        distribution = dist.Independent(base_dist, 1)
        action = distribution.rsample()  # Reparameterization trick

        # Squash action to [-1, 1] - usually TanhTransform is used but keeping it simple with direct tanh
        squashed_action = torch.tanh(action)
        return squashed_action, distribution

    def log_prob(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _, distribution = self.forward(feat)
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
        expected_symlog = torch.sum(probs * self.bins, dim=-1)
        return symexp(expected_symlog)

    def two_hot_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss on two-hot encoded target."""
        target = torch.clamp(target, self.lower, self.upper)

        # Compute two-hot targets
        diffs = torch.abs(self.bins.unsqueeze(0) - target.unsqueeze(1))
        # Find closest bins
        idx = torch.argmin(diffs, dim=-1)

        two_hot = torch.zeros_like(logits)
        for i in range(target.shape[0]):
            b = target[i]
            # Simple soft assignment for continuous target mapping to discrete bins
            for j in range(self.num_bins - 1):
                if self.bins[j] <= b <= self.bins[j + 1]:
                    w = (b - self.bins[j]) / (self.bins[j + 1] - self.bins[j])
                    two_hot[i, j] = 1.0 - w
                    two_hot[i, j + 1] = w
                    break

        return F.cross_entropy(logits, two_hot)
