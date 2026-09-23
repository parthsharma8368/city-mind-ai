"""
metrics.py — Real metrics endpoints for CityMind AI dashboard and comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
import pandas as pd

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
    for json_file in METRICS_DIR.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            if data.get("controller") == controller:
                matches.append(data)
        except Exception:
            pass
    if not matches:
        return {}
    return next((m for m in matches if m.get("seed") == seed), matches[0])


@router.get("/comparison")
def get_comparison():
    """
    Return aggregated comparison metrics across all controllers.
    Reads from artifacts/metrics/*.json files.
    """
    controllers = ["ppo", "ippo", "fixed_time", "sumo_actuated", "max_pressure"]
    labels = {
        "ppo": "PPO (Centralized)",
        "ippo": "IPPO (Parameter-Shared)",
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
            "mean_waiting_time": round(float(data.get("mean_waiting_time", 0)), 2),
            "mean_queue": round(float(data.get("mean_queue", 0)), 3),
            "throughput": round(float(data.get("throughput", 0)), 4),
            "mean_speed": round(float(data.get("mean_speed", 0)), 3),
            "total_arrived": int(data.get("total_arrived", 0)),
            "total_steps": int(data.get("total_steps", 0)),
            "cumulative_reward": round(float(data.get("cumulative_reward", 0)), 1),
            "backlog": int(data.get("backlog", 0)),
            "total_teleports": int(data.get("total_teleports", 0)),
            "trips_completed": int(data.get("trips_completed", 0)),
            "mean_trip_timeloss": round(float(data.get("mean_trip_timeloss", 0)), 2),
        })

    if not results:
        # Return empty but valid response
        return {"results": [], "message": "No evaluation results found. Run run_evaluation.py first."}

    # Compute improvement % for whichever RL controller currently has a
    # result and (if both do) is actually better, against each baseline that
    # has a result. sumo_actuated is the harder, calibrated-adaptive bar;
    # fixed_time is the weaker static one — report both rather than only
    # the easier comparison, so the headline can't quietly pick the flattering one.
    rl_candidates = [r for r in results if r["controller"] in ("ppo", "ippo")]
    best_rl = min(rl_candidates, key=lambda r: r["mean_waiting_time"], default=None)

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

    return {"results": results, "improvement": improvement, "improvement_vs_actuated": improvement_vs_actuated}


@router.get("/summary")
def get_summary():
    """
    Return the dashboard's headline controller summary — whichever RL
    controller (ppo or ippo) currently has the lower mean_waiting_time, not
    always "ppo". Same best-RL selection as /comparison, so the Dashboard
    and Comparison page never show two different "headline" results.
    """
    ppo_data = _load_metric_json("ppo")
    ippo_data = _load_metric_json("ippo")

    candidates = [(name, data) for name, data in (("ppo", ppo_data), ("ippo", ippo_data)) if data]
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

    best_name, best_data = min(
        candidates, key=lambda item: float(item[1].get("mean_waiting_time", float("inf")))
    )

    return {
        "controller": best_name,
        "mean_waiting_time": round(float(best_data.get("mean_waiting_time", 0)), 2),
        "mean_queue": round(float(best_data.get("mean_queue", 0)), 3),
        "throughput": round(float(best_data.get("throughput", 0)), 4),
        "mean_speed": round(float(best_data.get("mean_speed", 0)), 3),
        "total_arrived": int(best_data.get("total_arrived", 0)),
        "total_steps": int(best_data.get("total_steps", 0)),
        "cumulative_reward": round(float(best_data.get("cumulative_reward", 0)), 1),
        "total_switches": int(best_data.get("total_switches", 0)),
    }


@router.get("/experiments-list")
def list_experiment_metrics():
    """List all available experiment metric JSON files."""
    result = []
    for f in METRICS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
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
    return result
