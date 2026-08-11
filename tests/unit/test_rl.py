"""Unit tests for Model-Based RL (DreamerV3)."""

import torch

from rl_controller.dreamer.actor_critic import ActorCritic
from rl_controller.dreamer.rssm import RSSM


def test_rssm_initial_state():
    """Test the RSSM initial state generation shapes."""
    batch_size = 4
    rssm = RSSM(deter_dim=512, stoch_dim=32, num_classes=32)
    device = torch.device("cpu")

    state = rssm.initial_state(batch_size, device)

    assert state.deter.shape == (4, 512)
    assert state.stoch.shape == (4, 32, 32)
    assert state.logits.shape == (4, 32, 32)


def test_rssm_imagine():
    """Test RSSM prior imagination dynamics."""
    batch_size = 2
    rssm = RSSM(deter_dim=512, stoch_dim=32, num_classes=32, action_dim=3)

    state = rssm.initial_state(batch_size, torch.device("cpu"))
    dummy_action = torch.randn(batch_size, 3)  # Steer, Throttle, Brake

    next_state = rssm.imagine(dummy_action, state)

    # Ensure next state deter is deterministic GRU output
    assert next_state.deter.shape == (2, 512)
    # Ensure stoch is sampled categorically
    assert next_state.stoch.shape == (2, 32, 32)


def test_rssm_kl_loss():
    """Test mathematical computation of KL Divergence with free nats."""
    rssm = RSSM()
    # Create two logits distributions
    post_logits = torch.randn(2, 32, 32)
    prior_logits = torch.randn(2, 32, 32)

    # Calculate KL
    kl = rssm.kl_loss(post_logits, prior_logits, free_nats=3.0)

    # Assert KL is a scalar tensor and at least free_nats (clipped)
    assert kl.dim() == 0
    assert kl.item() >= 3.0


def test_actor_critic_shapes():
    """Test the Actor network bounds and outputs."""
    actor_critic = ActorCritic(feat_dim=512 + (32 * 32), action_dim=3)

    # Create dummy RSSM state concatenated feature
    dummy_feat = torch.randn(4, 512 + 1024)

    # Forward pass
    action_dist = actor_critic.actor(dummy_feat)
    value_pred = actor_critic.critic(dummy_feat)

    # Sample action
    action = action_dist.rsample()

    # Test shapes
    assert action.shape == (4, 3)
    assert value_pred.shape == (4, 1)

    # Test squashed normal bounds [-1, 1] for actions
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
