"""Separate gradient clipping for the actor and the critic."""
from __future__ import annotations

import pytest
import torch

from src.agents.split_clip import split_clip_context


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Linear(3, 3)
        self.value_net = torch.nn.Linear(3, 1)


def _norm(params):
    return torch.sqrt(sum((p.grad ** 2).sum() for p in params))


def _with_grads(actor_value, critic_value):
    pol = _Policy()
    for p in pol.policy_net.parameters():
        p.grad = torch.full_like(p, actor_value)
    for p in pol.value_net.parameters():
        p.grad = torch.full_like(p, critic_value)
    return pol


def test_joint_clipping_crushes_the_actor_but_split_clipping_does_not():
    joint = _with_grads(0.1, 1e4)
    torch.nn.utils.clip_grad_norm_(joint.parameters(), 0.5)              # what plain PPO does
    assert _norm(joint.policy_net.parameters()) < 1e-3

    split = _with_grads(0.1, 1e4)
    original = torch.nn.utils.clip_grad_norm_
    with split_clip_context(split):
        torch.nn.utils.clip_grad_norm_(split.parameters(), 0.5)
    assert torch.nn.utils.clip_grad_norm_ is original                    # the patch is undone
    assert _norm(split.policy_net.parameters()) == pytest.approx(0.3 * 1.0, rel=0.3) or \
        _norm(split.policy_net.parameters()) <= 0.5 + 1e-5
    assert _norm(split.policy_net.parameters()) > 0.1                    # the actor kept a usable gradient
    assert _norm(split.value_net.parameters()) == pytest.approx(0.5, rel=1e-3)


def test_other_clip_calls_are_left_alone():
    other = torch.nn.Linear(2, 2)
    for p in other.parameters():
        p.grad = torch.full_like(p, 10.0)
    with split_clip_context(_Policy()):
        n = torch.nn.utils.clip_grad_norm_(other.parameters(), 1.0)
    assert float(n) > 1.0 and _norm(other.parameters()) == pytest.approx(1.0, rel=1e-3)
