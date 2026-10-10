"""experiments.py — List and retrieve experiment runs."""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
import yaml
from fastapi import APIRouter, HTTPException
from api.models import ExperimentRun
from api.util import read_json_file, sanitize_nan

logger = logging.getLogger("citymind.api.experiments")
router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = _PROJECT_ROOT / "artifacts" / "metrics"
EXPERIMENTS_ROOTS = {
    "ppo": _PROJECT_ROOT / "artifacts" / "experiments" / "ppo",
    "ippo": _PROJECT_ROOT / "artifacts" / "experiments" / "ippo",
    "ippo_phase_select": _PROJECT_ROOT / "artifacts" / "experiments" / "ippo_phase_select",
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
        if root.resolve() not in candidate.parents:
            continue
        if candidate.exists():
            return candidate
    # Fall back to the ppo root so the 404 message is stable when neither has it.
    return EXPERIMENTS_ROOTS["ppo"] / f"_missing_{run_id}"


class _TolerantSafeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates python/* tags (older configs) without executing them."""


_TolerantSafeLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", lambda _loader, suffix, _node: f"<python:{suffix}>"
)


def _load_yaml(text: str) -> dict:
    """Parse config text safely. Never uses yaml.Loader/unsafe_load; returns {} on any problem."""
    try:
        cfg = yaml.load(text, Loader=_TolerantSafeLoader)  # noqa: S506 - SafeLoader subclass
    except Exception as exc:
        logger.warning("malformed config.yaml ignored: %s", exc)
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _rel(path: Path) -> str:
    """Path relative to the repo root (never leak absolute server paths)."""
    try:
        return path.resolve().relative_to(_PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


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
            try:
                cfg = _load_yaml(cfg_path.read_text())
                sanitized_cfg = _sanitize_json(cfg)
                model_path = run_dir / "model_final.zip"
                runs.append(ExperimentRun(
                    run_id=run_dir.name,
                    algorithm=str(cfg.get("algorithm", "PPO")),
                    seed=_to_int(cfg.get("seed", 0)),
                    total_timesteps=_to_int(cfg.get("total_timesteps", 0)),
                    model_path=_rel(model_path) if model_path.exists() else None,
                    config=sanitized_cfg,
                ))
            except Exception as exc:  # one bad run must not 500 the whole list
                logger.warning("skipping run %s: %s", run_dir.name, exc)
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
        "model_path": _rel(model_path) if model_path.exists() else None,
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
    # Derive the metrics-filename prefix from which experiments root the run
    # actually lives under (run_dir.parent.name), not from parsing the
    # algorithm string — "MaskablePPO (parameter-shared, phase-selection)"
    # doesn't contain the substring "ippo", so string-matching it against
    # "ippo"/"ppo" would silently misroute ippo_phase_select runs to the
    # wrong metrics files.
    prefix = run_dir.parent.name if run_dir.parent.name in EXPERIMENTS_ROOTS else "ppo"

    for json_file in METRICS_DIR.glob(f"{prefix}*.json"):
        data = read_json_file(json_file)
        if not isinstance(data, dict):
            continue
        if data.get("source_run_id") != run_id:
            continue
        return sanitize_nan({
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
        })
    raise HTTPException(404, f"No evaluation has been run against {run_id}'s own checkpoint yet.")

