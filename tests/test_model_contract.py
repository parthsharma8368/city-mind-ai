"""Model/config artifact contract (rl-core-011/012) and the agent plumbing around it."""
from __future__ import annotations

import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import yaml

from src.agents.model_contract import (
    ModelContractError,
    build_artifact,
    validate_model_contract,
    run_env_settings,
    write_run_artifacts,
)
from src.environment.local_observation import OBS_SIZE
from src.environment.schema import METRICS_SCHEMA_VERSION, OBS_SCHEMA_VERSION
from src.scenario.tls_metadata import MAX_GREEN_PHASES, PhaseSpec, TLSSpec


def _fake_model(obs_n, act_n=None):
    return SimpleNamespace(
        observation_space=gym.spaces.Box(0, 1, (obs_n,), np.float32),
        action_space=gym.spaces.Discrete(act_n) if act_n else gym.spaces.MultiBinary(3),
    )


def test_mismatched_obs_size_raises_a_clear_error_before_sumo():
    with pytest.raises(ModelContractError, match=r"observation size \(4,\).*\(22,\)"):
        validate_model_contract(_fake_model(4), OBS_SIZE, model_path="m.zip")
    validate_model_contract(_fake_model(OBS_SIZE), OBS_SIZE)           # matching passes


def test_action_space_mismatch_is_rejected():
    with pytest.raises(ModelContractError, match="Discrete"):
        validate_model_contract(_fake_model(OBS_SIZE, 3), OBS_SIZE, expected_action_n=2)
    validate_model_contract(_fake_model(OBS_SIZE, 2), OBS_SIZE, expected_action_n=2)


def test_run_config_schema_version_is_checked(tmp_path):
    run = tmp_path / "run"
    ckpt = run / "checkpoints"
    ckpt.mkdir(parents=True)
    write_run_artifacts(run, {"obs_size": OBS_SIZE, "obs_schema_version": OBS_SCHEMA_VERSION + 1, "git_commit": "abc"})
    with pytest.raises(ModelContractError, match="obs_schema_version"):
        validate_model_contract(_fake_model(OBS_SIZE), OBS_SIZE, model_path=ckpt / "m.zip")     # config one dir up
    write_run_artifacts(run, {"obs_size": 17, "obs_schema_version": OBS_SCHEMA_VERSION})
    with pytest.raises(ModelContractError, match="obs_size=17"):
        validate_model_contract(_fake_model(OBS_SIZE), OBS_SIZE, model_path=run / "model_final.zip")
    write_run_artifacts(run, {"obs_size": OBS_SIZE, "obs_schema_version": OBS_SCHEMA_VERSION})
    validate_model_contract(_fake_model(OBS_SIZE), OBS_SIZE, model_path=run / "model_final.zip")


def test_legacy_model_without_schema_version_loads_with_a_warning(tmp_path, caplog):
    (tmp_path / "config.yaml").write_text(yaml.dump({"algorithm": "IPPO", "total_timesteps": 5}))
    with caplog.at_level("WARNING"):
        validate_model_contract(_fake_model(OBS_SIZE), OBS_SIZE, model_path=tmp_path / "model_final.zip")
    assert "legacy" in caplog.text
    validate_model_contract(_fake_model(OBS_SIZE), OBS_SIZE, model_path=tmp_path / "does_not_exist" / "m.zip")


def test_build_artifact_records_everything_needed_to_audit_a_run(tmp_path):
    art = build_artifact(algorithm="IPPO", seed=3, device="cpu", demand_id="1.0x", ppo_cfg={"a": 1},
                         env_cfg={"b": 2}, obs_size=OBS_SIZE, action_space="Discrete(2)",
                         requested_timesteps=1000, num_timesteps=980, status="finished")
    assert art["obs_size"] == OBS_SIZE and art["action_space"] == "Discrete(2)"
    assert art["obs_schema_version"] == OBS_SCHEMA_VERSION
    assert art["metrics_schema_version"] == METRICS_SCHEMA_VERSION
    assert art["git_commit"] and art["status"] == "finished"
    assert art["total_timesteps"] == 980 and art["requested_timesteps"] == 1000   # ACTUAL count
    started = build_artifact(algorithm="x", seed=1, device="cpu", demand_id="d", ppo_cfg={}, env_cfg={},
                             obs_size=1, action_space="a", requested_timesteps=1000, status="started")
    assert started["total_timesteps"] == 1000
    write_run_artifacts(tmp_path / "r", art)
    assert yaml.safe_load((tmp_path / "r" / "config.yaml").read_text())["obs_size"] == OBS_SIZE
    assert (tmp_path / "r" / "git_commit.txt").read_text() == art["git_commit"]


def test_agent_constructors_do_not_create_run_directories(tmp_path):
    """eval-scripts-14: building an agent just to load a model left an empty run behind."""
    from src.agents.ippo_agent import IPPOAgent

    out = tmp_path / "exp"
    IPPOAgent("x.sumocfg", "m.json", output_dir=str(out / "ippo"))
    assert not out.exists()


def test_ippo_agent_load_rejects_a_model_with_the_wrong_obs_size(tmp_path):
    from stable_baselines3 import PPO

    from src.agents.ippo_agent import IPPOAgent

    model = PPO("MlpPolicy", gym.make("CartPole-v1"), device="cpu", n_steps=8, batch_size=8)   # obs (4,)
    path = tmp_path / "cartpole.zip"
    model.save(str(path))
    agent = IPPOAgent("x.sumocfg", "m.json", output_dir=str(tmp_path / "o"))
    with pytest.raises(ModelContractError, match=r"\(4,\)"):
        agent.load(str(path))
    assert agent.model is None                       # a rejected model is never installed


def test_agent_config_is_written_at_start_and_updated_at_end(tmp_path):
    from stable_baselines3 import PPO

    from src.agents.ippo_agent import IPPOAgent

    agent = IPPOAgent("x.sumocfg", "m.json", output_dir=str(tmp_path / "o"))
    agent.model = PPO("MlpPolicy", gym.make("CartPole-v1"), device="cpu", n_steps=8, batch_size=8)
    agent._save_config_artifact(500, status="started")
    cfg = yaml.safe_load((agent.run_dir / "config.yaml").read_text())
    assert cfg["status"] == "started" and cfg["obs_size"] == 4 and "Discrete" in cfg["action_space"]
    assert cfg["total_timesteps"] == 500 and cfg["git_commit"] and cfg["local_obs_size"] == OBS_SIZE
    agent._save_config_artifact(500, status="finished")
    assert yaml.safe_load((agent.run_dir / "config.yaml").read_text())["status"] == "finished"


def test_run_env_settings_roundtrips_through_the_saved_run_config(tmp_path):
    """Regression: the helper once read 'env_cfg' while build_artifact writes 'env_config', so every model was silently
    evaluated with default observation options."""
    art = build_artifact(algorithm="x", seed=1, device="cpu", demand_id="1.0x", ppo_cfg={},
                         env_cfg={"observation": {"queue_reach_m": 300, "timing_source": "program", "pending_in_demand": True}},
                         obs_size=OBS_SIZE, action_space="Discrete(2)", requested_timesteps=1)
    write_run_artifacts(tmp_path, art)
    assert run_env_settings(tmp_path / "model_final.zip") == {
        "queue_reach_m": 300.0, "timing_source": "program", "pending_in_demand": True}
    assert run_env_settings(tmp_path / "checkpoints" / "ckpt.zip")["queue_reach_m"] == 300.0   # checkpoint one level down
    empty = tmp_path.parent / (tmp_path.name + "_legacy")
    empty.mkdir()
    assert run_env_settings(empty / "model.zip") == {"queue_reach_m": 0.0, "timing_source": "safety", "pending_in_demand": False}
