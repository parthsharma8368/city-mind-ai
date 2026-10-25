"""The shipped configs reproduce the recipe the reported model was trained with, and every key they hold is consumed."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from src.agents.ippo_agent import _fields_of
from src.environment.local_observation import LocalObservationConfig
from src.environment.local_reward import LocalRewardConfig
from src.environment.safety import SafetyConfig

CONFIGS = Path(__file__).parent.parent / "configs"


def _env(name="environment.yaml"):
    return yaml.safe_load((CONFIGS / name).read_text())


def test_environment_yaml_holds_the_trained_recipe():
    cfg = _env()
    rew = LocalRewardConfig(**_fields_of(LocalRewardConfig, cfg["reward"]))
    assert (rew.queue_level_weight, rew.downstream_level_weight, rew.switching_penalty) == (1.0, 2.0, 1.0)
    assert (rew.backlog_weight, rew.global_backlog_weight, rew.reward_scale, rew.reward_clip) == (0.5, 0.5, 0.1, 200)
    obs = LocalObservationConfig(**_fields_of(LocalObservationConfig, cfg["observation"]))
    assert (obs.queue_reach_m, obs.timing_source, obs.pending_in_demand) == (300, "program", False)
    saf = SafetyConfig(**cfg["safety"])
    assert (saf.min_green, saf.max_green, saf.yellow_duration, saf.all_red_duration) == (5, 60, 3, 1)
    assert (cfg["sumo"]["begin"], cfg["sumo"]["end"], cfg["decision_interval"]) == (25200, 28800, 5)


def test_every_reward_and_observation_key_is_a_real_dataclass_field():
    for name in ("environment.yaml", "environment_pending.yaml"):
        cfg = _env(name)
        for section, cls in (("reward", LocalRewardConfig), ("observation", LocalObservationConfig)):
            assert set(cfg[section]) <= {f.name for f in dataclasses.fields(cls)}, (name, section)


def test_pending_variant_differs_only_in_pending_in_demand():
    a, b = _env("environment.yaml"), _env("environment_pending.yaml")
    assert b["observation"].pop("pending_in_demand") is True and a["observation"].pop("pending_in_demand") is False
    assert a == b


def test_ppo_yaml_samples_per_update_and_update_count_are_consistent():
    p = yaml.safe_load((CONFIGS / "ippo.yaml").read_text())
    samples = p["n_steps"] * 98 * p["num_vec_envs"]
    assert samples % p["batch_size"] == 0 and samples // p["batch_size"] == 8     # 8 minibatches
    assert p["total_timesteps"] % samples == 0 and p["total_timesteps"] // samples == 100   # 100 PPO updates
    assert p["split_clip"] is True


def test_finetune_stage_is_ten_updates_with_otherwise_identical_hyperparameters():
    base = yaml.safe_load((CONFIGS / "ippo.yaml").read_text())
    ft = yaml.safe_load((CONFIGS / "ippo_finetune.yaml").read_text())
    samples = base["n_steps"] * 98 * base["num_vec_envs"]
    assert ft.pop("total_timesteps") == 10 * samples
    base.pop("total_timesteps")
    assert ft == base
