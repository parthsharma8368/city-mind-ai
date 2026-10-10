"""
metrics.py — Real metrics endpoints for CityMind AI dashboard and comparison.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
import pandas as pd

from api.util import read_json_file, sanitize_nan

logger = logging.getLogger("citymind.api.metrics")

router = APIRouter()

_PROJECT_ROOT = Path(__file__).parent.parent.parent
METRICS_DIR = _PROJECT_ROOT / "artifacts" / "metrics"
EXPERIMENTS_DIR = _PROJECT_ROOT / "artifacts" / "experiments"


def _load_metric_json(controller: str, seed: int = 42) -> dict[str, Any]:
    """Load metrics JSON for a controller. Each controller now has one file
    per evaluation seed (fixed_time_seed42_1.0x.json, _seed123_, _seed777_)
    — pick seed 42 deterministically (the project's convention default)
    rather than whatever a directory glob happens to return first."""
    matches = []
    skipped = False
    for json_file in METRICS_DIR.glob("*.json"):
        data = read_json_file(json_file)
        if data is None:
            skipped = True
            continue
        if isinstance(data, dict) and data.get("controller") == controller:
            matches.append(data)
    if not matches:
        return {}
    exact = next((m for m in matches if m.get("seed") == seed), None)
    if exact is not None:
        return exact
    if skipped:
        # The seed's file is probably mid-write: do not silently substitute another seed.
        logger.warning("seed %s for %s unavailable (a metrics file was unreadable)", seed, controller)
        return {}
    return matches[0]


def _f(value: Any) -> float:
    """float() that tolerates None/garbage (returns 0.0); NaN is kept and sanitised at output."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _i(value: Any) -> int:
    """int() that tolerates None/NaN/inf (returns 0)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0
    return int(f) if math.isfinite(f) else 0


@router.get("/comparison")
def get_comparison():
    """
    Return aggregated comparison metrics across all controllers.
    Reads from artifacts/metrics/*.json files.
    """
    controllers = ["ppo", "ippo", "ippo_phase_select", "fixed_time", "sumo_actuated", "max_pressure"]
    labels = {
        "ppo": "PPO (Centralized)",
        "ippo": "IPPO (Parameter-Shared)",
        "ippo_phase_select": "IPPO Phase-Select",
        "fixed_time": "Fixed-Time (static)",
        "sumo_actuated": "SUMO-Actuated (calibrated)",
        "max_pressure": "Max-Pressure",
    }
    results = []
    for ctrl in controllers:
        data = _load_metric_json(ctrl)
        if not data:
            continue
        results.append({
            "controller": ctrl,
            "label": labels.get(ctrl, ctrl),
            "seed": data.get("seed"),
            "mean_waiting_time": round(_f(data.get("mean_waiting_time", 0)), 2),
            "mean_queue": round(_f(data.get("mean_queue", 0)), 3),
            "throughput": round(_f(data.get("throughput", 0)), 4),
            "mean_speed": round(_f(data.get("mean_speed", 0)), 3),
            "total_arrived": _i(data.get("total_arrived", 0)),
            "total_steps": _i(data.get("total_steps", 0)),
            "cumulative_reward": round(_f(data.get("cumulative_reward", 0)), 1),
            "backlog": _i(data.get("backlog", 0)),
            "total_teleports": _i(data.get("total_teleports", 0)),
            "trips_completed": _i(data.get("trips_completed", 0)),
            "mean_trip_timeloss": round(_f(data.get("mean_trip_timeloss", 0)), 2),
        })

    if not results:
        # Return empty but valid response
        return {"results": [], "message": "No evaluation results found. Run run_evaluation.py first."}

    # Compute improvement % for whichever RL controller currently has a
    # result and is actually the best of them, against each baseline that
    # has a result. sumo_actuated is the harder, calibrated-adaptive bar;
    # fixed_time is the weaker static one — report both rather than only
    # the easier comparison, so the headline can't quietly pick the flattering one.
    # Ranked by trips_completed, NOT mean_waiting_time — the same starvation-
    # hiding bug fixed in /summary (2026-09-28) was still live here: picking
    # by min(mean_waiting_time) would keep selecting whichever controller
    # starves the most vehicles, since that's exactly what depresses the
    # in-network average. See get_summary()'s docstring for the full reasoning.
    rl_candidates = [r for r in results if r["controller"] in ("ppo", "ippo", "ippo_phase_select")]
    best_rl = max(rl_candidates, key=lambda r: r["trips_completed"], default=None)

    def _improvement_vs(baseline_name: str) -> dict:
        baseline = next((r for r in results if r["controller"] == baseline_name), None)
        if not (baseline and best_rl and baseline["mean_waiting_time"] > 0):
            return {}
        trips_gap_pct = round(
            (best_rl["trips_completed"] - baseline["trips_completed"]) / baseline["trips_completed"] * 100
            if baseline.get("trips_completed") else 0, 1
        )
        return {
            "controller": best_rl["controller"],
            "baseline": baseline_name,
            "waiting_time_improvement_pct": round(
                (baseline["mean_waiting_time"] - best_rl["mean_waiting_time"]) / baseline["mean_waiting_time"] * 100, 1
            ),
            "throughput_improvement_pct": round(
                (best_rl["throughput"] - baseline["throughput"]) / baseline["throughput"] * 100
                if baseline["throughput"] > 0 else 0, 1
            ),
            # Negative means the RL controller completed FEWER trips than the
            # baseline — a lower mean_waiting_time alongside a negative gap
            # here is starvation (fewer vehicles finishing, not faster
            # service), not a real improvement. See release gate "no starvation".
            "trips_completed_gap_pct": trips_gap_pct,
            "starvation_flag": trips_gap_pct < -2.0,
        }

    improvement = _improvement_vs("fixed_time")
    improvement_vs_actuated = _improvement_vs("sumo_actuated")

    return sanitize_nan(
        {"results": results, "improvement": improvement, "improvement_vs_actuated": improvement_vs_actuated}
    )


@router.get("/summary")
def get_summary():
    """
    Return the dashboard's headline RL controller summary.

    Ranked by trips_completed (descending), NOT mean_waiting_time: the
    corrected evaluation (2026-09-23) showed IPPO's lower mean_waiting_time
    was a starvation artifact — it completed 18% fewer trips than the
    actuated baseline while leaving 26% more vehicles stuck in the network.
    mean_waiting_time only reflects vehicles currently in the network, so a
    controller that lets fewer of them finish can look artificially good on
    it. Ranking by trips completed is what the "no starvation" release gate
    actually checks, so the Dashboard's headline can't repeat that mistake.
    """
    ppo_data = _load_metric_json("ppo")
    ippo_data = _load_metric_json("ippo")
    ippo_phase_select_data = _load_metric_json("ippo_phase_select")

    candidates = [
        (name, data) for name, data in (
            ("ppo", ppo_data), ("ippo", ippo_data), ("ippo_phase_select", ippo_phase_select_data),
        ) if data
    ]
    if not candidates:
        # Fallback to reading from evaluation_results.parquet
        parquet_path = METRICS_DIR / "evaluation_results.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, engine="pyarrow")
                ppo_df = df[df["controller"] == "ppo"]
                if not ppo_df.empty:
                    candidates = [("ppo", ppo_df.iloc[0].to_dict())]
            except Exception:
                pass

    if not candidates:
        raise HTTPException(404, "No metrics found. Run evaluation first.")

    best_name, best_data = max(
        candidates, key=lambda item: _f(item[1].get("trips_completed", 0))
    )

    baseline = _load_metric_json("sumo_actuated") or _load_metric_json("fixed_time")
    trips_gap_pct = None
    if baseline and baseline.get("trips_completed"):
        trips_gap_pct = round(
            (_f(best_data.get("trips_completed", 0)) - _f(baseline["trips_completed"]))
            / _f(baseline["trips_completed"]) * 100, 1
        )

    return sanitize_nan({
        "controller": best_name,
        "mean_waiting_time": round(_f(best_data.get("mean_waiting_time", 0)), 2),
        "mean_queue": round(_f(best_data.get("mean_queue", 0)), 3),
        "throughput": round(_f(best_data.get("throughput", 0)), 4),
        "mean_speed": round(_f(best_data.get("mean_speed", 0)), 3),
        "total_arrived": _i(best_data.get("total_arrived", 0)),
        "total_steps": _i(best_data.get("total_steps", 0)),
        "cumulative_reward": round(_f(best_data.get("cumulative_reward", 0)), 1),
        "total_switches": _i(best_data.get("total_switches", 0)),
        "backlog": _i(best_data.get("backlog", 0)),
        "total_teleports": _i(best_data.get("total_teleports", 0)),
        "trips_completed": _i(best_data.get("trips_completed", 0)),
        "mean_trip_timeloss": round(_f(best_data.get("mean_trip_timeloss", 0)), 2),
        "trips_completed_gap_pct": trips_gap_pct,
        "starvation_flag": (trips_gap_pct is not None and trips_gap_pct < -2.0),
    })


@router.get("/experiments-list")
def list_experiment_metrics():
    """List all available experiment metric JSON files."""
    result = []
    for f in METRICS_DIR.glob("*.json"):
        data = read_json_file(f)
        if not isinstance(data, dict):
            continue
        try:
            result.append({
                "filename": f.name,
                "controller": data.get("controller", "unknown"),
                "seed": data.get("seed"),
                "demand_id": data.get("demand_id"),
                "mean_waiting_time": data.get("mean_waiting_time"),
                "throughput": data.get("throughput"),
            })
        except Exception:
            pass
    return sanitize_nan(result)
