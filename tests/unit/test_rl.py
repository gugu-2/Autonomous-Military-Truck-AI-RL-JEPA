"""Unit tests for DreamerV3 RL Controller."""

import pytest
import torch

try:
    from rl_controller.dreamer.actor_critic import Actor, Critic
    from rl_controller.dreamer.rssm import RSSM

    HAS_RL = True
except ImportError:
    HAS_RL = False


@pytest.mark.skipif(not HAS_RL, reason="rl_controller not importable")
def test_rssm_initial_state():
    rssm = RSSM(deter_dim=512, stoch_dim=32, num_classes=32)
    state = rssm.initial_state(4, torch.device("cpu"))
    assert state.deter.shape == (4, 512)
    assert state.stoch.shape == (4, 32, 32)


@pytest.mark.skipif(not HAS_RL, reason="rl_controller not importable")
def test_rssm_imagine():
    rssm = RSSM(deter_dim=512, stoch_dim=32, num_classes=32, action_dim=3)
    state = rssm.initial_state(2, torch.device("cpu"))
    action = torch.randn(2, 3)
    next_state = rssm.imagine(action, state)
    assert next_state.deter.shape == (2, 512)
    assert next_state.stoch.shape == (2, 32, 32)


@pytest.mark.skipif(not HAS_RL, reason="rl_controller not importable")
def test_rssm_kl_loss():
    rssm = RSSM()
    post_logits = torch.randn(2, 32, 32)
    prior_logits = torch.randn(2, 32, 32)
    kl = rssm.kl_loss(post_logits, prior_logits, free_nats=3.0)
    assert kl.dim() == 0
    assert kl.item() >= 3.0


@pytest.mark.skipif(not HAS_RL, reason='rl_controller not importable')
def test_actor_output_shape():
    feat_dim = 512 + 32 * 32
    actor = Actor(feat_dim=feat_dim, action_dim=3)
    dummy_feat = torch.randn(4, feat_dim)
    dist = actor(dummy_feat)
    # rsample gives unbounded Normal sample; apply tanh to squash to [-1, 1]
    raw_sample = dist.rsample()
    action = torch.tanh(raw_sample)
    assert action.shape == (4, 3)
    # Squashed tanh bounds
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


@pytest.mark.skipif(not HAS_RL, reason="rl_controller not importable")
def test_critic_expected_value():
    feat_dim = 512 + 32 * 32
    critic = Critic(feat_dim=feat_dim)
    dummy_feat = torch.randn(4, feat_dim)
    logits = critic(dummy_feat)
    value = critic.expected_value(logits)
    assert value.shape == (4, 1)
