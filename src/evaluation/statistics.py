"""
statistics.py
=============
Statistical aggregation for evaluation results.
Computes mean ± std per controller and demand level.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_results(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def aggregate_by_controller(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std for each (controller, demand_id) pair."""
    numeric_cols = [
        "mean_waiting_time", "total_waiting_time",
        "mean_queue", "max_queue",
        "throughput", "total_arrived", "total_departed",
        "mean_speed", "total_switches",
        "total_teleports", "backlog",
        "trips_completed", "mean_trip_duration", "mean_trip_timeloss",
    ]
    agg = {}
    for col in numeric_cols:
        if col in df.columns:
            agg[col] = ["mean", "std"]

    result = df.groupby(["controller", "demand_id"]).agg(agg)
    result.columns = ["_".join(c).strip() for c in result.columns]
    return result.reset_index()


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a human-readable comparison table."""
    agg = aggregate_by_controller(df)
    print("\n" + "="*80)
    print("EVALUATION RESULTS — Mean ± Std across seeds")
    print("="*80)

    metrics = [
        ("mean_waiting_time", "Avg Wait Time (s)"),
        ("mean_queue", "Avg Queue (vehicles)"),
        ("throughput", "Throughput (ratio)"),
        ("total_arrived", "Arrived Vehicles"),
    ]

    for demand_id, group in agg.groupby("demand_id"):
        print(f"\nDemand: {demand_id}")
        print(f"{'Controller':<16} {'Trips':>8} {'TimeLoss':>10} {'±':>6} "
              f"{'Backlog':>9} {'Teleports':>10} {'Avg Wait':>10} {'±':>6}")
        print("-" * 90)
        for _, row in group.iterrows():
            trips_m = row.get("trips_completed_mean", 0.0)
            tl_m = row.get("mean_trip_timeloss_mean", 0.0)
            tl_s = row.get("mean_trip_timeloss_std", 0.0)
            backlog_m = row.get("backlog_mean", 0.0)
            tel_m = row.get("total_teleports_mean", 0.0)
            wt_m = row.get("mean_waiting_time_mean", 0.0)
            wt_s = row.get("mean_waiting_time_std", 0.0)
            print(f"{row['controller']:<16} {trips_m:>8.0f} {tl_m:>10.2f} {tl_s:>6.2f} "
                  f"{backlog_m:>9.1f} {tel_m:>10.1f} {wt_m:>10.2f} {wt_s:>6.2f}")
    print("="*90 + "\n")


def compute_improvement_over_baseline(
    df: pd.DataFrame,
    baseline: str = "fixed_time",
    target: str = "ppo",
    metric: str = "mean_waiting_time",
) -> pd.DataFrame:
    """Compute % improvement of target vs baseline per demand level."""
    agg = df.groupby(["controller", "demand_id"])[metric].mean().reset_index()
    pivot = agg.pivot(index="demand_id", columns="controller", values=metric)
    if baseline in pivot.columns and target in pivot.columns:
        pivot["improvement_pct"] = (
            (pivot[baseline] - pivot[target]) / pivot[baseline] * 100
        )
    return pivot
