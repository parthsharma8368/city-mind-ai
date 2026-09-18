"""
compare.py
==========
Generates comparison plots and tables from evaluation results.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for headless environments

from src.evaluation.statistics import load_results, aggregate_by_controller, compute_improvement_over_baseline


def plot_comparison_bars(
    results_path: str,
    output_dir: str = "artifacts/reports",
    metrics: list[str] | None = None,
) -> None:
    """Generate bar charts comparing controllers across metrics."""
    df = load_results(results_path)
    agg = aggregate_by_controller(df)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics = ["mean_waiting_time", "mean_queue", "throughput"]

    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in agg.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        controllers = agg["controller"].unique()
        colors = {"fixed_time": "#E74C3C", "max_pressure": "#F39C12", "ppo": "#27AE60"}
        x = np.arange(len(agg["demand_id"].unique()))
        width = 0.25

        for i, ctrl in enumerate(controllers):
            ctrl_data = agg[agg["controller"] == ctrl]
            means = ctrl_data[mean_col].values
            stds = ctrl_data[std_col].values if std_col in ctrl_data.columns else np.zeros_like(means)
            ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                   label=ctrl.replace("_", " ").title(),
                   color=colors.get(ctrl, "#888888"), alpha=0.85)

        ax.set_xlabel("Demand Level")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Controller Comparison — {metric.replace('_', ' ').title()}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(agg["demand_id"].unique())
        ax.legend()
        plt.tight_layout()

        out_path = Path(output_dir) / f"comparison_{metric}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved plot: {out_path}")
