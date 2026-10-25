"""Figures for the professor report. Reads artifacts/eval10_run/report/{results,summary}.csv. Run from the repository root."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "fig"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASE, A, B, FIXED = "#8a8985", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def box(ax, xy, w, h, text, fc, ec=None, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec or fc, lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")


def arrow(ax, a, b, text=None, rad=0.0, dy=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.5, color=MUTED, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha="center", va="center", fontsize=9, color=MUTED,
                bbox={"fc": "white", "ec": "none", "pad": 1.5})


def fig_loop():
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    ax.set_xlim(-0.15, 10.15)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    box(ax, (0.0, 1.0), 3.3, 2.2, "Agent\nONE neural network,\nshared by all 98 signals\n(book Ch.1, p.9-11)", "#dbe8f9", A, 9.5, True)
    box(ax, (6.7, 1.0), 3.3, 2.2, "Environment = SUMO\n98 traffic lights,\n~15,000 cars, Ingolstadt,\n07:00-08:00", "#e6e5e1", BASE, 9.5, True)
    arrow(ax, (3.45, 2.8), (6.55, 2.8), "action (every 5 s)", dy=0.22)
    arrow(ax, (6.55, 2.1), (3.45, 2.1), "state (22 numbers)", dy=0.22)
    arrow(ax, (6.55, 1.4), (3.45, 1.4), "reward (minus waiting cars)", dy=0.22)
    ax.text(5, 3.8, "The agent-environment loop (repeats 720 times per simulated hour)", ha="center", fontsize=10.5, color=INK, weight="bold")
    ax.text(5, 0.35, "A safety layer sits between action and traffic light: minimum/maximum green, yellow and all-red are never skipped.",
            ha="center", fontsize=9, color=MUTED)
    fig.savefig(OUT / "fig_loop.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ppo_clip():
    eps = 0.2
    r = np.linspace(0, 2, 400)
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=False)
    for ax, adv, title in ((axs[0], 1, "Action was GOOD (advantage > 0)"), (axs[1], -1, "Action was BAD (advantage < 0)")):
        plain = r * adv
        clipped = np.clip(r, 1 - eps, 1 + eps) * adv
        ax.plot(r, plain, color=BASE, lw=1.4, ls="--", label="r x advantage (unclipped)")
        ax.plot(r, np.minimum(plain, clipped), color=A, lw=2.4, label="PPO objective = min(unclipped, clipped)")
        ax.axvline(1, color=GRID, lw=1)
        ax.axvspan(1 - eps, 1 + eps, color="#dbe8f9", alpha=0.5, lw=0)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("r = new policy / old policy probability of the action")
        ax.grid(color=GRID, lw=0.6)
    axs[0].text(1.22, 0.15, "gain stops\nrising here", fontsize=8.5, color=MUTED)
    axs[1].text(0.05, -1.75, "no extra credit\nfor going below 0.8", fontsize=8.5, color=MUTED)
    axs[0].set_ylabel("objective")
    axs[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ppo_clip.png", dpi=200)
    plt.close(fig)


def fig_gradnorm():
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    names = ["Critic gradient\n(value network)", "Actor gradient\n(policy network)"]
    vals = [3830, 0.19]
    ax.barh(names, vals, color=[B, A], height=0.45)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 2e4)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("gradient norm before clipping (log scale)")
    ax.text(3830 * 1.15, 0, "~3,830", va="center", fontsize=9.5, color=INK)
    ax.text(0.19 * 1.25, 1, "~0.19  (shrunk ~7,600x when clipped jointly)", va="center", fontsize=9.5, color=INK)
    ax.axvline(0.5, color=MUTED, lw=1.2, ls="--")
    ax.text(0.56, 0.62, "clip limit 0.5", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gradnorm.png", dpi=200)
    plt.close(fig)


def fig_seed_lag():
    d = pd.read_csv("artifacts/eval10_run/report/results.csv")
    order = [("sumo_actuated", "SUMO actuated\n(baseline)", BASE), ("ippo_A", "Model A", A), ("ippo_B", "Model B", B)]
    w = d.pivot_table(index="seed", columns="name", values="lag_per_scheduled_car_s")
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for s, row in w.iterrows():
        ax.plot([0, 1, 2], [row[k] for k, _, _ in order], color="#c4c3be", lw=1.2, zorder=1)
    for i, (k, lab, c) in enumerate(order):
        ax.scatter(np.full(len(w), i), w[k], s=34, color=c, zorder=3, edgecolor="white", linewidth=0.8)
        m = w[k].mean()
        ax.plot([i - 0.22, i + 0.22], [m, m], color=INK, lw=2, zorder=4)
        ax.text(i + 0.27, m, f"mean {m:.1f}", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for _, l, _ in order])
    ax.set_xlim(-0.4, 2.8)
    ax.set_ylabel("lag per scheduled car (s), lower is better")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Each grey line = one of the 10 random seeds (same traffic for all three controllers)", fontsize=9.5, color=MUTED, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_seed_lag.png", dpi=200)
    plt.close(fig)


def fig_improvement():
    s = pd.read_csv("artifacts/eval10_run/report/summary.csv")
    labels = {"lag_per_scheduled_car_s": "Lag per scheduled car", "mean_timeloss_cars_nonzero": "Time loss per finished car",
              "backlog": "Backlog at 08:00", "mean_waiting_time": "Mean waiting time", "mean_queue": "Mean queue",
              "mean_speed": "Mean speed", "due_completion_ratio": "Due completion", "trips_completed": "Trips completed",
              "throughput": "Throughput", "never_inserted": "Cars never inserted", "mean_depart_delay_cars": "Depart delay",
              "teleports": "Teleports (gridlock)", "mean_trip_duration_nonzero": "Trip duration", "overdue_cars": "Overdue cars"}
    higher = {"mean_speed", "due_completion_ratio", "trips_completed", "throughput"}
    keys = list(labels)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    y = np.arange(len(keys))
    h = 0.38
    for off, (m, c, n) in zip((-h / 2, h / 2), (("ippo_A", A, "Model A"), ("ippo_B", B, "Model B"))):
        sub = s[s.model == m].set_index("metric")
        v = [sub.loc[k, "pct_diff"] * (1 if k in higher else -1) for k in keys]
        ax.barh(y + off, v, height=h, color=c, label=n)
        for yy, k, val in zip(y + off, keys, v):
            ax.text(max(val, 0) + 0.6, yy, f"{int(sub.loc[k, 'seeds_won'])}/10", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in keys])
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("improvement over SUMO actuated (%): longer bar = better")
    ax.set_title("Small grey number = seeds (out of 10) on which the model beat the baseline", fontsize=9, color=MUTED, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_improvement.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_loop()
    fig_ppo_clip()
    fig_gradnorm()
    fig_seed_lag()
    fig_improvement()
    print(sorted(p.name for p in OUT.iterdir()))
