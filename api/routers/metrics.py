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


def _load_metric_json(controller: str) -> dict[str, Any]:
    """Load metrics JSON for a controller (ppo, fixed_time, max_pressure)."""
    # Try exact match first
    for json_file in METRICS_DIR.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            if data.get("controller") == controller:
                return data
        except Exception:
            pass
    # Fallback: filename prefix
    candidates = list(METRICS_DIR.glob(f"{controller}*.json"))
    if candidates:
        try:
            return json.loads(candidates[0].read_text())
        except Exception:
            pass
    return {}


@router.get("/comparison")
def get_comparison():
    """
    Return aggregated comparison metrics across all controllers.
    Reads from artifacts/metrics/*.json files.
    """
    controllers = ["ppo", "fixed_time", "max_pressure"]
    results = []
    for ctrl in controllers:
        data = _load_metric_json(ctrl)
        if not data:
            continue
        results.append({
            "controller": ctrl,
            "label": {"ppo": "PPO (RL)", "fixed_time": "Fixed-Time", "max_pressure": "Max-Pressure"}.get(ctrl, ctrl),
            "mean_waiting_time": round(float(data.get("mean_waiting_time", 0)), 2),
            "mean_queue": round(float(data.get("mean_queue", 0)), 3),
            "throughput": round(float(data.get("throughput", 0)), 4),
            "mean_speed": round(float(data.get("mean_speed", 0)), 3),
            "total_arrived": int(data.get("total_arrived", 0)),
            "total_steps": int(data.get("total_steps", 0)),
            "cumulative_reward": round(float(data.get("cumulative_reward", 0)), 1),
        })

    if not results:
        # Return empty but valid response
        return {"results": [], "message": "No evaluation results found. Run run_evaluation.py first."}

    # Compute improvement % relative to fixed_time baseline
    baseline = next((r for r in results if r["controller"] == "fixed_time"), None)
    ppo = next((r for r in results if r["controller"] == "ppo"), None)

    improvement = {}
    if baseline and ppo and baseline["mean_waiting_time"] > 0:
        improvement["waiting_time_improvement_pct"] = round(
            (baseline["mean_waiting_time"] - ppo["mean_waiting_time"]) / baseline["mean_waiting_time"] * 100, 1
        )
        improvement["throughput_improvement_pct"] = round(
            (ppo["throughput"] - baseline["throughput"]) / baseline["throughput"] * 100
            if baseline["throughput"] > 0 else 0, 1
        )

    return {"results": results, "improvement": improvement}


@router.get("/summary")
def get_summary():
    """
    Return aggregated dashboard summary — best metrics across all runs.
    Uses latest PPO metrics as primary.
    """
    ppo_data = _load_metric_json("ppo")
    fixed_data = _load_metric_json("fixed_time")

    if not ppo_data and not fixed_data:
        # Fallback to reading from evaluation_results.parquet
        parquet_path = METRICS_DIR / "evaluation_results.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, engine="pyarrow")
                ppo_df = df[df["controller"] == "ppo"]
                if not ppo_df.empty:
                    ppo_data = ppo_df.iloc[0].to_dict()
            except Exception:
                pass

    if not ppo_data:
        raise HTTPException(404, "No metrics found. Run evaluation first.")

    return {
        "controller": "ppo",
        "mean_waiting_time": round(float(ppo_data.get("mean_waiting_time", 0)), 2),
        "mean_queue": round(float(ppo_data.get("mean_queue", 0)), 3),
        "throughput": round(float(ppo_data.get("throughput", 0)), 4),
        "mean_speed": round(float(ppo_data.get("mean_speed", 0)), 3),
        "total_arrived": int(ppo_data.get("total_arrived", 0)),
        "total_steps": int(ppo_data.get("total_steps", 0)),
        "cumulative_reward": round(float(ppo_data.get("cumulative_reward", 0)), 1),
        "total_switches": int(ppo_data.get("total_switches", 0)),
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
