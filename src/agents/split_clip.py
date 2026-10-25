"""
split_clip.py
=============
PPO with SEPARATE gradient-norm clipping for the actor and the critic.

SB3 clips ONE global norm over every parameter (`clip_grad_norm_(policy.parameters(), max_grad_norm)`).
Measured on the phase-select run (artifacts/eda/rl_behavior_forensics/data/grad_norms.json): critic grad
norm 3,830 vs actor 0.19 against a 0.5 cap -> every update scaled by 1.3e-4, the actor's gradient fell
below Adam's eps, and the policy stayed ~90% of maximum entropy for millions of steps. Clipping the two
groups independently (the "separate" arm of clip_ablation.py) moved the policy ~10x further per update.

Implemented by swapping torch's clip function for the duration of ``train()`` only; the call that passes the
complete policy parameter set is split in two, every other call is untouched.
"""
from __future__ import annotations

import contextlib

import torch as th
from stable_baselines3 import PPO


def _is_critic(name: str) -> bool:
    return "value_net" in name


@contextlib.contextmanager
def split_clip_context(policy, record=None):
    named = list(policy.named_parameters())
    critic = [p for n, p in named if _is_critic(n)]
    actor = [p for n, p in named if not _is_critic(n)]
    full_ids = {id(p) for _, p in named}
    original = th.nn.utils.clip_grad_norm_

    def clip(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        if {id(p) for p in params} != full_ids:
            return original(params, max_norm, *args, **kwargs)
        n_actor = original(actor, max_norm, *args, **kwargs)
        n_critic = original(critic, max_norm, *args, **kwargs)
        if record is not None:
            record(float(n_actor), float(n_critic))
        return th.sqrt(n_actor ** 2 + n_critic ** 2)

    th.nn.utils.clip_grad_norm_ = clip
    try:
        yield
    finally:
        th.nn.utils.clip_grad_norm_ = original


class _SplitClipMixin:
    def train(self) -> None:  # noqa: D401 - SB3 hook
        norms: list[tuple[float, float]] = []
        with split_clip_context(self.policy, record=lambda a, c: norms.append((a, c))):
            super().train()
        if norms:  # pre-clip norms averaged over the update's minibatches
            self.logger.record("train/grad_norm_actor", sum(a for a, _ in norms) / len(norms))
            self.logger.record("train/grad_norm_critic", sum(c for _, c in norms) / len(norms))


class SplitClipPPO(_SplitClipMixin, PPO):
    pass
