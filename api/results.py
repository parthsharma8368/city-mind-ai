"""results.py — The evaluation results the dashboard shows: SUMO actuated, fixed time and the two final IPPO models.

Layout (written by scripts/evaluate.py, summarised by scripts/report_eval.py)::

    <root>/<id>/s<seed>/metrics/<type>_seed<seed>_<demand>.json     one run's metrics
    <root>/<id>/s<seed>/traj/<type>_seed<seed>_<demand>.parquet     its recorded vehicles + signal phases
    <root>/report/{results.csv, summary.csv}                        10-seed table and paired statistics

``<id>`` is the name shown in the dashboard (``sumo_actuated``, ``fixed_time``, ``ippo_A``, ``ippo_B``); ``<type>`` is the
controller type inside the file name (``ippo`` for both models). Set CITYMIND_RESULTS_ROOT to show another run.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from api.util import read_json_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = Path(os.environ.get("CITYMIND_RESULTS_ROOT") or PROJECT_ROOT / "artifacts" / "eval10_run").resolve()

BASELINE = "sumo_actuated"
# Display order, label, controller type used in the file names.
CONTROLLERS: dict[str, tuple[str, str]] = {
    "sumo_actuated": ("SUMO actuated (baseline)", "sumo_actuated"),
    "fixed_time": ("Fixed time", "fixed_time"),
    "ippo_A": ("Model A (final)", "ippo"),
    "ippo_B": ("Model B (final)", "ippo"),
}
MODELS = ("ippo_A", "ippo_B")
DEFAULT_SEED = 42


def label(controller: str) -> str:
    return CONTROLLERS.get(controller, (controller, ""))[0]


def _stem(controller: str, seed: int, demand_id: str) -> str:
    return f"{CONTROLLERS[controller][1]}_seed{seed}_{demand_id}"


def metrics_path(controller: str, seed: int = DEFAULT_SEED, demand_id: str = "1.0x") -> Path:
    return RESULTS_ROOT / controller / f"s{seed}" / "metrics" / f"{_stem(controller, seed, demand_id)}.json"


def trajectory_path(controller: str, seed: int = DEFAULT_SEED, demand_id: str = "1.0x") -> Path:
    return RESULTS_ROOT / controller / f"s{seed}" / "traj" / f"{_stem(controller, seed, demand_id)}.parquet"


def load_metrics(controller: str, seed: int = DEFAULT_SEED, demand_id: str = "1.0x") -> dict[str, Any]:
    """One run's metrics JSON ({} if that run does not exist)."""
    if controller not in CONTROLLERS:
        return {}
    data = read_json_file(metrics_path(controller, seed, demand_id)) if metrics_path(controller, seed, demand_id).is_file() else None
    return data if isinstance(data, dict) else {}


def available_controllers() -> list[str]:
    return [c for c in CONTROLLERS if (RESULTS_ROOT / c).is_dir()]


def seeds() -> list[int]:
    found = {int(p.name[1:]) for c in available_controllers() for p in (RESULTS_ROOT / c).glob("s*") if p.name[1:].isdigit()}
    return sorted(found)


def per_seed_table() -> pd.DataFrame | None:
    """One row per controller and seed, one column per measure (report/results.csv)."""
    path = RESULTS_ROOT / "report" / "results.csv"
    return pd.read_csv(path) if path.is_file() else None


def paired_summary() -> pd.DataFrame | None:
    """Paired model-vs-baseline statistics per measure (report/summary.csv)."""
    path = RESULTS_ROOT / "report" / "summary.csv"
    return pd.read_csv(path) if path.is_file() else None
