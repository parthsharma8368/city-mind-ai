"""comparison.py — Aggregated controller comparison endpoints."""
from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException
import pandas as pd
from api.models import ComparisonResult

router = APIRouter()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_METRICS_DIR = _PROJECT_ROOT / "artifacts" / "metrics"

@router.get("/summary")
def get_comparison_summary(
    metrics_dir: str | None = None,
    demand_id: str = "1.0x",
):
    """Return mean metrics per controller for a given demand level."""
    base = Path(metrics_dir) if metrics_dir else DEFAULT_METRICS_DIR
    path = base / "evaluation_results.parquet"
    if not path.exists():
        raise HTTPException(404, "Evaluation results not found. Run run_evaluation.py first.")


    df = pd.read_parquet(path)
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
    metrics_dir: str | None = None,
):
    """Compute % improvement of target vs baseline."""
    base = Path(metrics_dir) if metrics_dir else DEFAULT_METRICS_DIR
    path = base / "evaluation_results.parquet"
    if not path.exists():
        raise HTTPException(404, "Evaluation results not found.")


    df = pd.read_parquet(path)
    if demand_id != "all":
        df = df[df["demand_id"] == demand_id]

    baseline_mean = df[df["controller"] == baseline][metric].mean()
    target_mean = df[df["controller"] == target][metric].mean()
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
