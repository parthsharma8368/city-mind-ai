"""
metrics.py — Results of the final models against the baselines, for the dashboard.

Everything is read from the evaluation results (api/results.py): one run per controller and seed for the replayed episode,
the 10-seed means and the paired statistics (scripts/report_eval.py) for the comparison.
"""
from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, HTTPException

from api import results
from api.util import sanitize_nan
from scripts.report_eval import METRICS as MEASURES  # (key, label, "lower"|"higher", format)

router = APIRouter()
_STARVATION_GAP_PCT = -2.0   # completing >2% fewer trips than the baseline is starvation, whatever the wait time says


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
    return int(round(f)) if math.isfinite(f) else 0


def _ten_seed_means() -> dict[str, dict[str, float]]:
    """controller -> measure -> mean over all seeds (report/results.csv)."""
    table = results.per_seed_table()
    if table is None:
        return {}
    means = table.groupby("name").mean(numeric_only=True)
    return {name: row.to_dict() for name, row in means.iterrows() if name in results.CONTROLLERS}


def _improvement(baseline: dict, model: dict, model_name: str, baseline_name: str) -> dict:
    if not baseline or not model or not baseline.get("mean_waiting_time"):
        return {}
    trips_gap = (model["trips_completed"] - baseline["trips_completed"]) / baseline["trips_completed"] * 100
    return {
        "controller": model_name,
        "baseline": baseline_name,
        "waiting_time_improvement_pct": round((baseline["mean_waiting_time"] - model["mean_waiting_time"]) / baseline["mean_waiting_time"] * 100, 1),
        "throughput_improvement_pct": round((model["throughput"] - baseline["throughput"]) / baseline["throughput"] * 100, 1),
        # Negative = the model completed FEWER trips than the baseline; a lower wait next to that is starvation, not speed.
        "trips_completed_gap_pct": round(trips_gap, 1),
        "starvation_flag": trips_gap < _STARVATION_GAP_PCT,
    }


@router.get("/comparison")
def get_comparison():
    """Mean of every seed per controller: SUMO actuated (baseline), fixed time and the two final models."""
    means = _ten_seed_means()
    if not means:
        return {"results": [], "message": "No evaluation results found. Run scripts/evaluate.py and scripts/report_eval.py first."}
    n_seeds = len(results.seeds())
    rows = []
    for name in results.CONTROLLERS:
        m = means.get(name)
        if not m:
            continue
        rows.append({
            "controller": name,
            "label": results.label(name),
            "seed": None,
            "n_seeds": n_seeds,
            "mean_waiting_time": round(_f(m["mean_waiting_time"]), 2),
            "mean_queue": round(_f(m["mean_queue"]), 3),
            "throughput": round(_f(m["throughput"]), 4),
            "mean_speed": round(_f(m["mean_speed"]), 3),
            "total_arrived": _i(m["trips_completed"]),
            "total_steps": 720,
            "cumulative_reward": None,
            "backlog": _i(m["backlog"]),
            "total_teleports": round(_f(m["teleports"]), 1),
            "trips_completed": _i(m["trips_completed"]),
            "mean_trip_timeloss": round(_f(m["mean_timeloss_cars_nonzero"]), 2),
            "lag_per_scheduled_car_s": round(_f(m["lag_per_scheduled_car_s"]), 1),
            "never_inserted": _i(m["never_inserted"]),
            "due_completion_ratio": round(_f(m["due_completion_ratio"]), 3),
        })
    by_name = {r["controller"]: r for r in rows}
    models = [r for r in rows if r["controller"] in results.MODELS]
    best = max(models, key=lambda r: r["trips_completed"], default=None)
    vs = lambda base: _improvement(by_name.get(base, {}), best or {}, best["controller"] if best else "", base)  # noqa: E731
    return sanitize_nan({"results": rows, "improvement": vs("fixed_time"), "improvement_vs_actuated": vs(results.BASELINE)})


@router.get("/summary")
def get_summary():
    """Headline run of the dashboard: the final model that completes the most trips over all seeds, shown on seed 42 - the run the replay animates."""
    candidates = [(name, results.load_metrics(name)) for name in results.MODELS]
    candidates = [(name, data) for name, data in candidates if data]
    if not candidates:
        raise HTTPException(404, "No metrics found. Run evaluation first.")
    means = _ten_seed_means()   # choose on every seed, not on the one seed that is replayed
    best_name, best = max(
        candidates, key=lambda item: _f(means.get(item[0], {}).get("trips_completed", item[1].get("trips_completed", 0)))
    )
    baseline = results.load_metrics(results.BASELINE)
    trips_gap_pct = None
    if baseline.get("trips_completed"):
        trips_gap_pct = round((_f(best.get("trips_completed")) - _f(baseline["trips_completed"])) / _f(baseline["trips_completed"]) * 100, 1)
    return sanitize_nan({
        "controller": best_name,
        "seed": best.get("seed"),
        "mean_waiting_time": round(_f(best.get("mean_waiting_time")), 2),
        "mean_queue": round(_f(best.get("mean_queue")), 3),
        "throughput": round(_f(best.get("throughput")), 4),
        "mean_speed": round(_f(best.get("mean_speed")), 3),
        "total_arrived": _i(best.get("total_arrived")),
        "total_steps": _i(best.get("total_steps")),
        "cumulative_reward": round(_f(best.get("cumulative_reward")), 1),
        "total_switches": _i(best.get("total_switches")),
        "backlog": _i(best.get("backlog")),
        "total_teleports": _i(best.get("total_teleports")),
        "trips_completed": _i(best.get("trips_completed")),
        "mean_trip_timeloss": round(_f(best.get("mean_trip_timeloss")), 2),
        "trips_completed_gap_pct": trips_gap_pct,
        "starvation_flag": trips_gap_pct is not None and trips_gap_pct < _STARVATION_GAP_PCT,
    })


@router.get("/ten-seed")
def get_ten_seed():
    """Paired statistics of each final model against SUMO actuated, every measure (report/summary.csv)."""
    table = results.paired_summary()
    if table is None:
        raise HTTPException(404, "No paired report found. Run scripts/report_eval.py first.")
    labels = {key: (label, better) for key, label, better, _ in MEASURES}
    out: dict[str, Any] = {"baseline": results.BASELINE, "seeds": results.seeds(), "models": {}}
    for name, rows in table.groupby("model"):
        out["models"][name] = {
            "label": results.label(name),
            "measures": [
                {
                    "key": r.metric, "label": labels[r.metric][0], "better": labels[r.metric][1],
                    "baseline_mean": r.baseline_mean, "model_mean": r.model_mean, "pct_diff": r.pct_diff,
                    "ci_lo": r.t_ci_lo, "ci_hi": r.t_ci_hi, "seeds_won": int(r.seeds_won), "n_seeds": int(r.n_seeds),
                    "sign_test_p": r.sign_test_p, "verdict": r.verdict,
                }
                for r in rows.itertuples() if r.metric in labels
            ],
        }
    return sanitize_nan(out)
