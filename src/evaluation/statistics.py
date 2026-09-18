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
        print(f"{'Controller':<20} {'Avg Wait':>12} {'±':>4} {'Avg Queue':>12} {'±':>4} {'Throughput':>12} {'±':>4}")
        print("-" * 80)
        for _, row in group.iterrows():
            wt_m = row.get("mean_waiting_time_mean", 0.0)
            wt_s = row.get("mean_waiting_time_std", 0.0)
            q_m  = row.get("mean_queue_mean", 0.0)
            q_s  = row.get("mean_queue_std", 0.0)
            tp_m = row.get("throughput_mean", 0.0)
            tp_s = row.get("throughput_std", 0.0)
            print(f"{row['controller']:<20} {wt_m:>12.2f} {wt_s:>4.2f} "
                  f"{q_m:>12.2f} {q_s:>4.2f} {tp_m:>12.4f} {tp_s:>4.4f}")
    print("="*80 + "\n")


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
