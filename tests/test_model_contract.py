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
    write_run_artifacts,
)
from src.environment.action import ActionDecoder
from src.environment.local_observation import LOCAL_OBS_SIZE
from src.environment.schema import METRICS_SCHEMA_VERSION, OBS_SCHEMA_VERSION
from src.scenario.tls_metadata import MAX_GREEN_PHASES, PhaseSpec, TLSSpec


def _fake_model(obs_n, act_n=None):
    return SimpleNamespace(
        observation_space=gym.spaces.Box(0, 1, (obs_n,), np.float32),
        action_space=gym.spaces.Discrete(act_n) if act_n else gym.spaces.MultiBinary(3),
    )


def test_mismatched_obs_size_raises_a_clear_error_before_sumo():
    with pytest.raises(ModelContractError, match=r"observation size \(4,\).*\(18,\)"):
        validate_model_contract(_fake_model(4), LOCAL_OBS_SIZE, model_path="m.zip")
    validate_model_contract(_fake_model(LOCAL_OBS_SIZE), LOCAL_OBS_SIZE)           # matching passes


def test_action_space_mismatch_is_rejected():
    with pytest.raises(ModelContractError, match="Discrete"):
        validate_model_contract(_fake_model(18, 2), 18, expected_action_n=MAX_GREEN_PHASES)
    validate_model_contract(_fake_model(18, MAX_GREEN_PHASES), 18, expected_action_n=MAX_GREEN_PHASES)


def test_run_config_schema_version_is_checked(tmp_path):
    run = tmp_path / "run"
    ckpt = run / "checkpoints"
    ckpt.mkdir(parents=True)
    write_run_artifacts(run, {"obs_size": 18, "obs_schema_version": OBS_SCHEMA_VERSION + 1, "git_commit": "abc"})
    with pytest.raises(ModelContractError, match="obs_schema_version"):
        validate_model_contract(_fake_model(18), 18, model_path=ckpt / "m.zip")     # config one dir up
    write_run_artifacts(run, {"obs_size": 17, "obs_schema_version": OBS_SCHEMA_VERSION})
    with pytest.raises(ModelContractError, match="obs_size=17"):
        validate_model_contract(_fake_model(18), 18, model_path=run / "model_final.zip")
    write_run_artifacts(run, {"obs_size": 18, "obs_schema_version": OBS_SCHEMA_VERSION})
    validate_model_contract(_fake_model(18), 18, model_path=run / "model_final.zip")


def test_legacy_model_without_schema_version_loads_with_a_warning(tmp_path, caplog):
    (tmp_path / "config.yaml").write_text(yaml.dump({"algorithm": "IPPO", "total_timesteps": 5}))
    with caplog.at_level("WARNING"):
        validate_model_contract(_fake_model(18), 18, model_path=tmp_path / "model_final.zip")
    assert "legacy" in caplog.text
    validate_model_contract(_fake_model(18), 18, model_path=tmp_path / "does_not_exist" / "m.zip")


def test_build_artifact_records_everything_needed_to_audit_a_run(tmp_path):
    art = build_artifact(algorithm="IPPO", seed=3, device="cpu", demand_id="1.0x", ppo_cfg={"a": 1},
                         env_cfg={"b": 2}, obs_size=18, action_space="Discrete(6)",
                         requested_timesteps=1000, num_timesteps=980, status="finished")
    assert art["obs_size"] == 18 and art["action_space"] == "Discrete(6)"
    assert art["obs_schema_version"] == OBS_SCHEMA_VERSION
    assert art["metrics_schema_version"] == METRICS_SCHEMA_VERSION
    assert art["git_commit"] and art["status"] == "finished"
    assert art["total_timesteps"] == 980 and art["requested_timesteps"] == 1000   # ACTUAL count
    started = build_artifact(algorithm="x", seed=1, device="cpu", demand_id="d", ppo_cfg={}, env_cfg={},
                             obs_size=1, action_space="a", requested_timesteps=1000, status="started")
    assert started["total_timesteps"] == 1000
    write_run_artifacts(tmp_path / "r", art)
    assert yaml.safe_load((tmp_path / "r" / "config.yaml").read_text())["obs_size"] == 18
    assert (tmp_path / "r" / "git_commit.txt").read_text() == art["git_commit"]


def test_agent_constructors_do_not_create_run_directories(tmp_path):
    """eval-scripts-14: building an agent just to load a model left an empty run behind."""
    from src.agents.ippo_agent import IPPOAgent
    from src.agents.ippo_maskable_agent import IPPOMaskableAgent
    from src.agents.ppo_agent import PPOAgent

    out = tmp_path / "exp"
    PPOAgent("x.sumocfg", "m.json", output_dir=str(out / "ppo"))
    IPPOAgent("x.sumocfg", "m.json", output_dir=str(out / "ippo"))
    IPPOMaskableAgent("x.sumocfg", "m.json", output_dir=str(out / "mask"))
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
    assert cfg["total_timesteps"] == 500 and cfg["git_commit"] and cfg["local_obs_size"] == LOCAL_OBS_SIZE
    agent._save_config_artifact(500, status="finished")
    assert yaml.safe_load((agent.run_dir / "config.yaml").read_text())["status"] == "finished"


def test_maskable_wrapper_exposes_one_mask_per_agent_and_masks_match_the_specs(tmp_path):
    from src.agents.ippo_maskable_agent import MaskableActionWrapper
    from src.environment.decentralized_sumo_env import DecentralizedSumoEnv

    entries = [{"tls_id": t, "program_id": "0", "type": "static",
                "phases": [{"index": i, "state": s, "duration": 10.0}
                           for i, s in enumerate(["GGrr", "yyrr", "rrGG", "rryy"][: 2 * n])],
                "controlled_lanes": ["a"]} for t, n in (("A", 1), ("B", 2))]
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps(entries))
    env = DecentralizedSumoEnv(str(tmp_path / "s.sumocfg"), str(meta))
    masks = env.action_masks()
    assert masks["A"].tolist() == [True] + [False] * 5
    assert masks["B"].tolist() == [True, True] + [False] * 4
    assert all(m.dtype == bool and m.shape == (MAX_GREEN_PHASES,) for m in masks.values())

    inner = SimpleNamespace(par_env=env)
    wrapper = MaskableActionWrapper.__new__(MaskableActionWrapper)
    wrapper.venv = SimpleNamespace(venv=SimpleNamespace(vec_envs=[inner]))
    listed = wrapper.env_method("action_masks")
    assert [m.tolist() for m in listed] == [masks["A"].tolist(), masks["B"].tolist()]
    assert wrapper.has_attr("action_masks") is True


def test_action_decoder_length_and_order():
    def spec(t):
        return TLSSpec(t, "0", "static", t, [PhaseSpec(0, "G", 5.0)], ["l"])
    dec = ActionDecoder([spec("b"), spec("a"), spec("c")])
    assert dec.decode(np.array([1, 0, 1])) == {"a": 1, "b": 0, "c": 1}   # sorted by tls_id
    with pytest.raises(ValueError, match="mismatch"):
        dec.decode(np.array([1, 0]))
    assert dec.action_space().n == 3
