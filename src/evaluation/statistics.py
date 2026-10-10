"""
statistics.py
=============
Statistical aggregation for evaluation results.

Design notes (read before quoting any number from here):

* The evaluation "seeds" (42/123/777) only change SUMO's random draws
  (speed factors, rerouting devices); the demand files are fixed and each RL
  controller was trained ONCE. So between-seed spread measures simulation
  noise only - it says nothing about training-seed variance.
* With n = 3 seeds a hypothesis test has almost no power, and p-values from
  it are not evidence of significance. This module therefore reports effect
  sizes, t-based confidence intervals (n and df stated), and the minimum
  detectable effect (MDE) for the paired-by-seed design, and *deliberately
  computes no p-values*. "Exceeds MDE" is the strongest statement it makes.
* Comparisons are paired by seed (same SUMO seed => same speed factors for
  the same vehicle), which removes the shared seed effect from the difference.
* A percentile bootstrap over n = 3 paired differences has only 10 distinct
  resamples; it is available for larger n and flagged unreliable below
  ``MIN_N_BOOTSTRAP``.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

MIN_N_BOOTSTRAP = 6

# two-sided 95% t quantiles for df = 1..30 (fallback when scipy is unavailable)
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
         2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
         2.052, 2.048, 2.045, 2.042]


def load_results(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def aggregate_by_controller(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std (sample std; NaN for a single seed) for each (controller, demand_id) pair."""
    numeric_cols = [
        "mean_waiting_time", "total_waiting_time",
        "mean_queue", "max_queue",
        "throughput", "total_arrived", "total_departed",
        "mean_speed", "total_switches",
        "total_teleports", "backlog",
        "trips_completed", "mean_trip_duration", "mean_trip_timeloss",
        # drain-aware protocol (metrics_schema_version >= 2)
        "lag_per_scheduled_car_s", "due_completion_ratio", "teleports",
        "mean_timeloss_cars_nonzero", "never_inserted",
    ]
    agg = {}
    for col in numeric_cols:
        if col in df.columns:
            agg[col] = ["mean", "std"]

    result = df.groupby(["controller", "demand_id"]).agg(agg)
    result.columns = ["_".join(c).strip() for c in result.columns]
    return result.reset_index()


def _fmt(x: float, spec: str) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, spec)


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a human-readable comparison table (headline: lag per scheduled car when available)."""
    agg = aggregate_by_controller(df)
    cols = [("Trips", 8, "trips_completed_mean", ".0f"),
            ("Lag/car s", 10, "lag_per_scheduled_car_s_mean", ".1f"),
            ("±", 6, "lag_per_scheduled_car_s_std", ".1f"),
            ("DueDone", 8, "due_completion_ratio_mean", ".3f"),
            ("TimeLoss", 9, "mean_trip_timeloss_mean", ".1f"),
            ("±", 6, "mean_trip_timeloss_std", ".1f"),
            ("Backlog", 8, "backlog_mean", ".0f"),
            ("Teleports", 9, "total_teleports_mean", ".1f")]
    header = f"{'Controller':<18}" + " ".join(f"{n:>{w}}" for n, w, _, _ in cols)
    rule = "-" * len(header)
    print("\n" + "=" * len(header))
    print("EVALUATION RESULTS — mean ± sample std across evaluation seeds (n per cell in 'n')")
    print("Simulation noise only: one trained model per RL controller, no significance implied.")
    print("=" * len(header))
    for demand_id, group in agg.groupby("demand_id"):
        n_seeds = df[df.demand_id == demand_id].groupby("controller").size().to_dict()
        print(f"\nDemand: {demand_id}")
        print(header + "   n")
        print(rule + "----")
        for _, row in group.iterrows():
            cells = " ".join(f"{_fmt(row.get(key, float('nan')), spec):>{w}}" for _, w, key, spec in cols)
            print(f"{row['controller']:<18}{cells} {n_seeds.get(row['controller'], 0):>3}")
    print("=" * len(header) + "\n")


def improvement_pct(baseline: float, target: float, lower_is_better: bool = True) -> float:
    """% improvement of target over baseline; NaN (never inf) when the baseline is 0/non-finite."""
    if not (math.isfinite(baseline) and math.isfinite(target)) or baseline == 0:
        return float("nan")
    gain = (baseline - target) if lower_is_better else (target - baseline)
    return float(gain / abs(baseline) * 100.0)


def compute_improvement_over_baseline(
    df: pd.DataFrame,
    baseline: str = "sumo_actuated",
    target: str = "ippo_phase_select",
    metric: str = "lag_per_scheduled_car_s",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """% improvement of target vs baseline per demand level.

    Raises ValueError (instead of silently omitting the column) if the metric,
    baseline or target is absent. A zero/non-finite baseline mean gives NaN.
    """
    if metric not in df.columns:
        raise ValueError(f"metric {metric!r} not in results (columns: {sorted(df.columns)})")
    present = set(df["controller"].unique())
    for name, role in ((baseline, "baseline"), (target, "target")):
        if name not in present:
            raise ValueError(f"{role} controller {name!r} not in results (have {sorted(present)})")
    agg = df.groupby(["controller", "demand_id"])[metric].mean().reset_index()
    pivot = agg.pivot(index="demand_id", columns="controller", values=metric)
    pivot["improvement_pct"] = [
        improvement_pct(b, t, lower_is_better) for b, t in zip(pivot[baseline], pivot[target])
    ]
    return pivot


# ─── paired-by-seed inference ────────────────────────────────────────────────
def _t_quantile(df_: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df_))
    except ImportError:  # pragma: no cover - scipy is present in the project venv
        return _T975[df_ - 1] if 1 <= df_ <= len(_T975) else 1.96


def t_confidence_interval(diffs: Iterable[float], confidence: float = 0.95) -> tuple[float, float, int]:
    """t-based CI of the mean paired difference. Returns (lo, hi, n); NaN bounds if n < 2."""
    d = np.asarray(list(diffs), dtype=float)
    n = int(d.size)
    if n < 2:
        return float("nan"), float("nan"), n
    if confidence != 0.95:
        from scipy import stats
        tq = float(stats.t.ppf(0.5 + confidence / 2, n - 1))
    else:
        tq = _t_quantile(n - 1)
    half = tq * d.std(ddof=1) / math.sqrt(n)
    m = float(d.mean())
    return m - half, m + half, n


def bootstrap_ci(
    diffs: Iterable[float], confidence: float = 0.95, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float, bool]:
    """Percentile bootstrap CI of the mean paired difference.

    Returns (lo, hi, reliable). ``reliable`` is False for n < MIN_N_BOOTSTRAP:
    with n = 3 there are only 10 distinct resamples, so the interval is
    degenerate and the t-based interval should be used instead.
    """
    d = np.asarray(list(diffs), dtype=float)
    if d.size < 2:
        return float("nan"), float("nan"), False
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return float(lo), float(hi), bool(d.size >= MIN_N_BOOTSTRAP)


def minimum_detectable_effect(sd_diff: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true mean paired difference a two-sided one-sample t-test on n
    paired differences (with SD ``sd_diff``) detects with the given power.

    Exact via the noncentral t when scipy is present; otherwise the
    normal-approximation (z_{1-a/2} + z_power) * sd / sqrt(n) (optimistic for
    small n).
    """
    if n < 2 or not math.isfinite(sd_diff):
        return float("nan")
    try:
        from scipy import integrate, optimize, stats
    except ImportError:  # pragma: no cover
        z = 1.959964 + 0.841621
        return float(z * sd_diff / math.sqrt(n))
    dof = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, dof)

    def _power(nc: float) -> float:
        def g(v: float) -> float:
            s = tcrit * math.sqrt(v / dof)
            return stats.chi2.pdf(v, dof) * (1 - stats.norm.cdf(s - nc) + stats.norm.cdf(-s - nc))
        return integrate.quad(g, 0, np.inf, limit=200)[0]

    nc = optimize.brentq(lambda x: _power(x) - power, 0.05, 80.0)
    return float(nc * sd_diff / math.sqrt(n))


def paired_by_seed_comparison(
    df: pd.DataFrame,
    metric: str,
    baseline: str,
    controllers: Iterable[str] | None = None,
    demand_id: str | None = None,
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Paired-by-seed differences (controller - baseline) for one metric.

    Columns: n, mean_diff, per-seed diffs, sd_diff, t-CI, bootstrap CI (+reliable
    flag), MDE (80% power, alpha 0.05 two-sided, n paired seeds), exceeds_mde.
    No p-values - see the module docstring.
    """
    if metric not in df.columns:
        raise ValueError(f"metric {metric!r} not in results")
    d = df if demand_id is None else df[df.demand_id == demand_id]
    wide = d.pivot_table(index=seed_col, columns="controller", values=metric, aggfunc="mean")
    if baseline not in wide.columns:
        raise ValueError(f"baseline {baseline!r} not in results (have {sorted(wide.columns)})")
    ctrls = [c for c in (controllers if controllers is not None else wide.columns) if c != baseline]
    rows = []
    for c in ctrls:
        if c not in wide.columns:
            raise ValueError(f"controller {c!r} not in results")
        both = wide[[c, baseline]].dropna()
        diffs = (both[c] - both[baseline]).to_numpy()
        n = int(diffs.size)
        sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
        lo, hi, _ = t_confidence_interval(diffs)
        blo, bhi, reliable = bootstrap_ci(diffs)
        mde = minimum_detectable_effect(sd, n)
        mean_diff = float(diffs.mean()) if n else float("nan")
        rows.append({
            "controller": c, "baseline": baseline, "metric": metric, "n_seeds": n,
            "mean_diff": mean_diff, "per_seed_diffs": [float(x) for x in diffs], "sd_diff": sd,
            "t_ci_lo": lo, "t_ci_hi": hi, "boot_ci_lo": blo, "boot_ci_hi": bhi, "boot_reliable": reliable,
            "mde_80pct_power": mde,
            "exceeds_mde": bool(math.isfinite(mde) and abs(mean_diff) > mde),
        })
    return pd.DataFrame(rows)
