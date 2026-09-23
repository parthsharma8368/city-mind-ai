"""comparison.py — Aggregated controller comparison endpoints."""
from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException
import pandas as pd

router = APIRouter()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = _PROJECT_ROOT / "artifacts" / "metrics"
_CONTROLLERS = {"fixed_time", "sumo_actuated", "max_pressure", "ppo", "ippo"}
_METRICS = {
    "mean_waiting_time", "mean_queue", "throughput", "mean_speed", "total_arrived",
    "cumulative_reward", "total_switches", "backlog", "total_teleports",
    "trips_completed", "mean_trip_timeloss",
}


def _read_parquet_or_404(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(404, "Evaluation results not found. Run run_evaluation.py first.")
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise HTTPException(500, f"Could not read {path.name}: {exc}") from exc


@router.get("/summary")
def get_comparison_summary(
    demand_id: str = "1.0x",
):
    """Return mean metrics per controller for a given demand level."""
    path = METRICS_DIR / "evaluation_results.parquet"
    df = _read_parquet_or_404(path)
    if demand_id != "all":
        df = df[df["demand_id"] == demand_id]

    records = []
    for ctrl, group in df.groupby("controller"):
        records.append({
            "controller": ctrl,
            "mean_waiting_time": float(group["mean_waiting_time"].mean()) if "mean_waiting_time" in group else 0.0,
            "std_waiting_time": float(group["mean_waiting_time"].std()) if len(group) > 1 and "mean_waiting_time" in group else 0.0,
            "mean_queue": float(group["mean_queue"].mean()) if "mean_queue" in group else 0.0,
            "std_queue": float(group["mean_queue"].std()) if len(group) > 1 and "mean_queue" in group else 0.0,
            "throughput": float(group["throughput"].mean()) if "throughput" in group else 0.0,
            "std_throughput": float(group["throughput"].std()) if len(group) > 1 and "throughput" in group else 0.0,
            "total_arrived": int(group["total_arrived"].sum()) if "total_arrived" in group else 0,
            "runs": len(group),
        })
    return {"results": records}

@router.get("/improvement")
def get_improvement(
    baseline: str = "fixed_time",
    target: str = "ppo",
    metric: str = "mean_waiting_time",
    demand_id: str = "1.0x",
):
    """Compute % improvement of target vs baseline."""
    if baseline not in _CONTROLLERS or target not in _CONTROLLERS:
        raise HTTPException(400, f"baseline/target must be one of {sorted(_CONTROLLERS)}")
    if metric not in _METRICS:
        raise HTTPException(400, f"metric must be one of {sorted(_METRICS)}")

    path = METRICS_DIR / "evaluation_results.parquet"
    df = _read_parquet_or_404(path)
    if demand_id != "all":
        df = df[df["demand_id"] == demand_id]

    baseline_rows = df[df["controller"] == baseline][metric]
    target_rows = df[df["controller"] == target][metric]
    if baseline_rows.empty or target_rows.empty:
        raise HTTPException(404, f"No evaluation rows for baseline={baseline!r} / target={target!r} at demand_id={demand_id!r}.")

    baseline_mean = baseline_rows.mean()
    target_mean = target_rows.mean()
    improvement = (baseline_mean - target_mean) / baseline_mean * 100 if baseline_mean else 0

    return {
        "baseline": baseline,
        "target": target,
        "metric": metric,
        "demand_id": demand_id,
        "baseline_mean": float(baseline_mean),
        "target_mean": float(target_mean),
        "improvement_pct": float(improvement),
    }
