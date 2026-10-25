"""
model_contract.py
=================
What a saved model must record and what a loader must verify (rl-core-011/012):

  * config.yaml is written at the START of training (so a crashed run still has
    one) and updated at the end with the real ``num_timesteps``;
  * it records obs_size, action space, the observation/metrics schema versions,
    the git commit and the exact SUMO command inputs;
  * ``validate_model_contract`` refuses to use a model whose observation (or
    action) space does not match the environment, BEFORE SUMO is started, with a
    message that says what is wrong.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import yaml

from src.environment.schema import METRICS_SCHEMA_VERSION, OBS_SCHEMA_VERSION

log = logging.getLogger(__name__)


class ModelContractError(ValueError):
    """A saved model does not match the environment it is being loaded for."""


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001 - not in a git checkout / git missing
        return "unknown"


def build_artifact(
    *,
    algorithm: str,
    seed: int,
    device: str,
    demand_id: str,
    ppo_cfg: dict,
    env_cfg: dict,
    obs_size: int | None,
    action_space: str | None,
    requested_timesteps: int,
    num_timesteps: int = 0,
    status: str = "started",
    extra: dict | None = None,
) -> dict:
    artifact = {
        "algorithm": algorithm,
        "status": status,  # started | finished | failed
        "seed": seed,
        # total_timesteps keeps its old meaning for readers (API/experiments list):
        # the timesteps ACTUALLY trained once finished (model.num_timesteps),
        # the requested target while the run is only started / failed.
        "total_timesteps": num_timesteps if (status == "finished" and num_timesteps > 0) else requested_timesteps,
        "requested_timesteps": requested_timesteps,
        "num_timesteps": num_timesteps,
        "device": device,
        "demand_id": demand_id,
        "obs_size": obs_size,
        "action_space": action_space,
        "obs_schema_version": OBS_SCHEMA_VERSION,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "git_commit": current_git_commit(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ppo_config": ppo_cfg,
        "env_config": env_cfg,
    }
    if extra:
        artifact.update(extra)
    return artifact


def write_run_artifacts(run_dir: Path, artifact: dict) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(artifact, f, default_flow_style=False)
    (run_dir / "git_commit.txt").write_text(str(artifact.get("git_commit", "unknown")))


def _find_run_config(model_path: str | Path | None) -> dict | None:
    if model_path is None:
        return None
    p = Path(model_path)
    for candidate in (p.parent / "config.yaml", p.parent.parent / "config.yaml"):
        if candidate.is_file():
            try:
                return yaml.safe_load(candidate.read_text()) or {}
            except yaml.YAMLError:
                return None
    return None


def run_env_settings(model_path: str | Path | None) -> dict:
    """The observation options a model was TRAINED with, from its run's config.yaml: queue_reach_m, timing_source,
    pending_in_demand. Evaluating a model with other options is silent when the sizes match, so the evaluator reads this
    instead of trusting defaults. A run without a config gets the defaults."""
    obs = ((_find_run_config(model_path) or {}).get("env_config") or {}).get("observation") or {}
    return {
        "queue_reach_m": float(obs.get("queue_reach_m", 0.0)),
        "timing_source": str(obs.get("timing_source", "safety")),
        "pending_in_demand": bool(obs.get("pending_in_demand", False)),
    }


def validate_model_contract(
    model,
    expected_obs_size: int,
    *,
    model_path: str | Path | None = None,
    expected_action_n: int | None = None,
) -> None:
    """Raise ModelContractError if ``model`` cannot drive the environment.

    Legacy models (no recorded schema version) are accepted with a warning when
    their shapes match: the shape is the only thing that can be checked.
    """
    where = f" ({model_path})" if model_path else ""
    got = tuple(model.observation_space.shape)
    if got != (expected_obs_size,):
        raise ModelContractError(
            f"Model{where} was trained with observation size {got}, but this environment "
            f"produces ({expected_obs_size},). Re-train, or load it with the matching "
            "environment/feature version (obs size changes when the observation features, "
            "TLS metadata or OBS_SIZE change)."
        )
    if expected_action_n is not None:
        n = getattr(model.action_space, "n", None)
        if n != expected_action_n:
            raise ModelContractError(
                f"Model{where} has action space {model.action_space}, expected "
                f"Discrete({expected_action_n}). Wrong checkpoint type for this agent "
                "(binary keep/advance vs phase-selection)?"
            )
    cfg = _find_run_config(model_path)
    if cfg is None or "obs_schema_version" not in cfg:
        log.warning("Model%s has no recorded obs_schema_version (legacy run); only the "
                    "observation size could be verified.", where)
        return
    recorded_size = cfg.get("obs_size")
    if recorded_size is not None and int(recorded_size) != expected_obs_size:
        raise ModelContractError(
            f"Model{where}: run config records obs_size={recorded_size}, env produces {expected_obs_size}."
        )
    if int(cfg["obs_schema_version"]) != OBS_SCHEMA_VERSION:
        raise ModelContractError(
            f"Model{where} was trained with obs_schema_version={cfg['obs_schema_version']}, "
            f"current code is {OBS_SCHEMA_VERSION}: same-size features may mean something different."
        )
