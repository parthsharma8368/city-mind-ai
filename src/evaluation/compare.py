"""
compare.py
==========
Generates comparison plots and tables from evaluation results.

Default metrics are the ones that survive the survivor-bias critique
(see drain_metrics.py): trips completed, drain-aware lag per scheduled car,
due-completion ratio and backlog. ``mean_waiting_time`` is intentionally NOT a
default - it is an in-network average that rewards controllers for keeping
vehicles out of the network.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.statistics import aggregate_by_controller, load_results  # noqa: E402

DEFAULT_METRICS = [
    "trips_completed",
    "lag_per_scheduled_car_s",
    "due_completion_ratio",
    "backlog",
]

# Fixed colours for the known controllers; anything else gets a distinct tab10 colour.
CONTROLLER_COLORS = {
    "sumo_actuated": "#2C3E50",
    "fixed_time": "#E74C3C",
    "max_pressure": "#F39C12",
    "ppo": "#27AE60",
    "ippo": "#9B59B6",
    "ippo_phase_select": "#1ABC9C",
}


def controller_color(name: str, known_extra: list[str] | None = None) -> str:
    """Stable colour per controller; unknown names get distinct tab10 colours, never a shared grey."""
    if name in CONTROLLER_COLORS:
        return CONTROLLER_COLORS[name]
    extras = sorted((set(known_extra or []) | {name}) - set(CONTROLLER_COLORS))
    return matplotlib.colormaps["tab10"](extras.index(name) % 10)


def plot_comparison_bars(
    results_path: str,
    output_dir: str = "artifacts/reports",
    metrics: list[str] | None = None,
) -> list[Path]:
    """Generate grouped bar charts (controllers side by side per demand level) with ±1 std error bars."""
    df = load_results(results_path)
    agg = aggregate_by_controller(df)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics = metrics or DEFAULT_METRICS
    written: list[Path] = []

    controllers = list(agg["controller"].unique())
    demands = list(agg["demand_id"].unique())
    n_seeds = int(df.groupby(["controller", "demand_id"]).size().min())

    for metric in metrics:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col not in agg.columns:
            print(f"  (skipping {metric}: not in results)")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(demands))
        width = 0.8 / len(controllers)
        for i, ctrl in enumerate(controllers):
            ctrl_data = agg[agg["controller"] == ctrl].set_index("demand_id").reindex(demands)
            means = ctrl_data[mean_col].to_numpy(dtype=float)
            stds = np.nan_to_num(ctrl_data[std_col].to_numpy(dtype=float)) if std_col in ctrl_data else np.zeros_like(means)
            ax.bar(x + (i - (len(controllers) - 1) / 2) * width, means, width, yerr=stds, capsize=3,
                   label=ctrl.replace("_", " ").title(), color=controller_color(ctrl, controllers), alpha=0.9)

        ax.set_xlabel("Demand Level")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Controller comparison - {metric.replace('_', ' ')}\n"
                     f"mean ± std over {n_seeds} simulation seeds (one trained model per RL controller)")
        ax.set_xticks(x)
        ax.set_xticklabels(demands)
        ax.legend(fontsize=8)
        plt.tight_layout()

        out_path = Path(output_dir) / f"comparison_{metric}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        written.append(out_path)
        print(f"  Saved plot: {out_path}")
    return written
