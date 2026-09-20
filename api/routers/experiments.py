"""experiments.py — List and retrieve experiment runs."""
from __future__ import annotations
import json
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException
from api.models import ExperimentRun

router = APIRouter()

_PROJECT_ROOT = Path(__file__).parent.parent.parent
METRICS_DIR = _PROJECT_ROOT / "artifacts" / "metrics"
EXPERIMENTS_ROOTS = {
    "ppo": _PROJECT_ROOT / "artifacts" / "experiments" / "ppo",
    "ippo": _PROJECT_ROOT / "artifacts" / "experiments" / "ippo",
}
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _resolve_run_dir(run_id: str) -> Path:
    if not _SAFE_RUN_ID.match(run_id):
        raise HTTPException(400, f"Invalid run_id: {run_id!r}")
    for root in EXPERIMENTS_ROOTS.values():
        candidate = (root / run_id).resolve()
        # The regex above already blocks "/", but a bare ".." is still a
        # valid token that resolves to the parent directory - confirm the
        # resolved path is still actually inside root before trusting it.
        if root.resolve() not in candidate.parents and candidate != root.resolve():
            continue
        if candidate.exists():
            return candidate
    # Fall back to the ppo root so the 404 message is stable when neither has it.
    return EXPERIMENTS_ROOTS["ppo"] / run_id


def _load_yaml(text: str) -> dict:
    import yaml
    try:
        return yaml.safe_load(text) or {}
    except Exception:
        try:
            return yaml.load(text, Loader=yaml.Loader) or {}
        except Exception:
            return {}

def _sanitize_json(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [_sanitize_json(v) for v in obj]
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    else:
        return str(obj)

@router.get("/", response_model=list[ExperimentRun])
def list_experiments():
    runs = []
    for root in EXPERIMENTS_ROOTS.values():
        if not root.exists():
            continue
        for run_dir in run_dirs_sorted(root):
            cfg_path = run_dir / "config.yaml"
            if not cfg_path.exists():
                continue
            cfg = _load_yaml(cfg_path.read_text())
            sanitized_cfg = _sanitize_json(cfg)
            model_path = run_dir / "model_final.zip"
            runs.append(ExperimentRun(
                run_id=run_dir.name,
                algorithm=str(cfg.get("algorithm", "PPO")),
                seed=int(cfg.get("seed", 0)),
                total_timesteps=int(cfg.get("total_timesteps", 0)),
                model_path=str(model_path) if model_path.exists() else None,
                config=sanitized_cfg,
            ))
    runs.sort(key=lambda r: r.run_id, reverse=True)
    return runs


def run_dirs_sorted(root: Path):
    return sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)

@router.get("/{run_id}")
def get_experiment(run_id: str):
    run_dir = _resolve_run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(404, f"Run {run_id} not found")
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, "Config not found")
    cfg = _sanitize_json(_load_yaml(cfg_path.read_text()))
    model_path = run_dir / "model_final.zip"
    return {
        **cfg,
        "run_id": run_id,
        "has_model": model_path.exists(),
        "model_path": str(model_path) if model_path.exists() else None,
    }

@router.get("/{run_id}/summary")
def get_experiment_summary(run_id: str):
    """
    Return metrics summary for a specific experiment run — only if a saved
    evaluation actually recorded coming from *this* run_id's checkpoint
    (source_run_id). Evaluation JSON files are shared per-algorithm
    filenames (ppo_seed42_1.0x.json etc.), not per-run, so without this
    check every run of the same algorithm would show whichever evaluation
    ran last, even if it came from a different run entirely.
    """
    run_dir = _resolve_run_dir(run_id)
    cfg_path = run_dir / "config.yaml"
    algorithm = _load_yaml(cfg_path.read_text()).get("algorithm", "PPO") if cfg_path.exists() else "PPO"
    prefix = "ippo" if "ippo" in str(algorithm).lower() else "ppo"

    for json_file in METRICS_DIR.glob(f"{prefix}*.json"):
        try:
            data = json.loads(json_file.read_text())
        except Exception:
            continue
        if data.get("source_run_id") != run_id:
            continue
        return {
            "controller": data.get("controller", "ppo"),
            "seed": data.get("seed"),
            "demand_id": data.get("demand_id"),
            "total_steps": data.get("total_steps", 0),
            "final_sim_time": data.get("episode_duration", 3600.0),
            "average_waiting_time": data.get("mean_waiting_time", 0),
            "total_waiting_time": data.get("total_waiting_time", 0),
            "average_queue_length": data.get("mean_queue", 0),
            "maximum_queue_length": data.get("max_queue", 0),
            "average_speed": data.get("mean_speed", 0),
            "throughput": data.get("throughput", 0),
            "total_switches": data.get("total_switches", 0),
            "cumulative_reward": data.get("cumulative_reward", 0),
        }
    raise HTTPException(404, f"No evaluation has been run against {run_id}'s own checkpoint yet.")

