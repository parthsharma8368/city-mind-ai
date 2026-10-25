"""Multi-seed test report: paired-by-seed comparison of trained models against a baseline, every metric.

Reads ``<root>/<name>/s<seed>/metrics/<controller>_seed<seed>_1.0x.json`` (written by scripts/evaluate.py, one job per
controller and seed) and writes ``<root>/report/results.csv`` (one row per controller, seed, metric), ``summary.csv``
(one row per model, metric) and ``summary.md``.

For each metric and model: mean over seeds, mean paired difference to the baseline (model - baseline, same seed), 95% t-interval,
bootstrap 95% interval, seeds won (model strictly better), paired-t and sign-test p-values, minimum detectable effect (80% power).
The sign test is exact and makes no distribution assumption; the t-test and its interval assume the per-seed differences are roughly normal.
Verdict "Beaten" = the t-interval excludes zero on the good side AND the sign test p <= 0.05 (9 or 10 of 10 seeds won).

  python scripts/report_eval.py --root artifacts/eval10_run --baseline sumo_actuated --models ippo_A ippo_B [--extra fixed_time]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.statistics import bootstrap_ci, minimum_detectable_effect, t_confidence_interval  # noqa: E402

# (key, label, better, format); a key is looked up at the top level of the result JSON, then under "drain_metrics".
METRICS = [
    ("lag_per_scheduled_car_s", "Lag per scheduled car (s)", "lower", "{:.1f}"),
    ("mean_timeloss_cars_nonzero", "Time loss per finished car (s)", "lower", "{:.1f}"),
    ("backlog", "Backlog at end of window (cars)", "lower", "{:.0f}"),
    ("mean_waiting_time", "Mean waiting time (s)", "lower", "{:.2f}"),
    ("mean_queue", "Mean queue", "lower", "{:.3f}"),
    ("mean_speed", "Mean speed (m/s)", "higher", "{:.2f}"),
    ("due_completion_ratio", "Due completion", "higher", "{:.3f}"),
    ("trips_completed", "Trips completed", "higher", "{:.0f}"),
    ("throughput", "Throughput (arrived / departed)", "higher", "{:.3f}"),
    ("never_inserted", "Cars never inserted", "lower", "{:.0f}"),
    ("mean_depart_delay_cars", "Depart delay (s)", "lower", "{:.1f}"),
    ("teleports", "Teleports (gridlock)", "lower", "{:.1f}"),
    ("mean_trip_duration_nonzero", "Trip duration (s)", "lower", "{:.1f}"),
    ("overdue_cars", "Overdue cars", "lower", "{:.0f}"),
]


def _get(d: dict, key: str):
    if key in d and not isinstance(d[key], (dict, list)):
        return d[key]
    return (d.get("drain_metrics") or {}).get(key)


def load(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob("*/s*/metrics/*_seed*_1.0x.json")):
        name, seed_dir = f.parts[-4], f.parts[-3]
        d = json.loads(f.read_text())
        row = {"name": name, "seed": int(seed_dir[1:])}
        for key, *_ in METRICS:
            v = _get(d, key)
            row[key] = float(v) if v is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, baseline: str, model: str, key: str, better: str) -> dict | None:
    wide = df.pivot_table(index="seed", columns="name", values=key)
    if baseline not in wide or model not in wide:
        return None
    both = wide[[model, baseline]].dropna()
    diffs = (both[model] - both[baseline]).to_numpy()
    n = len(diffs)
    if n < 2:
        return None
    sign = -1.0 if better == "lower" else 1.0           # sign * diff > 0 means the model is better
    wins = int((sign * diffs > 0).sum())
    ties = int((diffs == 0).sum())
    sd = float(diffs.std(ddof=1))
    lo, hi, _ = t_confidence_interval(diffs)
    blo, bhi, reliable = bootstrap_ci(diffs)
    t_p = float(sps.ttest_1samp(diffs, 0.0).pvalue) if sd > 0 else (0.0 if diffs.mean() != 0 else 1.0)
    informative = n - ties
    sign_p = float(sps.binomtest(wins, informative, 0.5).pvalue) if informative > 0 else 1.0
    mean_diff = float(diffs.mean())
    ci_excludes_zero = (lo > 0) or (hi < 0)
    # "Beaten" needs BOTH the interval to exclude zero and the exact sign test to be significant (>= 9 of 10 seeds), so one
    # skewed seed cannot carry a verdict. Interval alone gives "Better (interval only)".
    if ci_excludes_zero and sign * mean_diff > 0:
        verdict = "Beaten" if sign_p <= 0.05 else "Better (interval only)"
    elif ci_excludes_zero:
        verdict = "Worse" if sign_p <= 0.05 else "Worse (interval only)"
    elif sign * mean_diff > 0:
        verdict = "Better, not significant"
    else:
        verdict = "Level" if abs(mean_diff) < 1e-12 or sign_p > 0.2 else "Worse, not significant"
    return {
        "model": model, "baseline": baseline, "metric": key, "n_seeds": n,
        "baseline_mean": float(both[baseline].mean()), "model_mean": float(both[model].mean()),
        "mean_diff": mean_diff, "pct_diff": mean_diff / float(both[baseline].mean()) * 100 if both[baseline].mean() else math.nan,
        "sd_diff": sd, "t_ci_lo": lo, "t_ci_hi": hi, "boot_ci_lo": blo, "boot_ci_hi": bhi,
        "seeds_won": wins, "seeds_tied": ties, "t_test_p": t_p, "sign_test_p": sign_p,
        "mde_80pct": minimum_detectable_effect(sd, n), "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", default="sumo_actuated")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="further controllers shown for context (e.g. fixed_time)")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)

    df = load(root)
    df.to_csv(out / "results.csv", index=False)
    labels = {k: (lab, better, fmt) for k, lab, better, fmt in METRICS}
    rows = []
    for m in a.models + a.extra:
        for key, _, better, _ in METRICS:
            r = compare(df, a.baseline, m, key, better)
            if r:
                rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)

    seeds = sorted(df.seed.unique())
    md = [f"# {len(seeds)}-seed test report\n", f"Seeds: {seeds}. Baseline: `{a.baseline}`. Paired by seed (model minus baseline).\n"]
    for m in a.models + a.extra:
        s = summ[summ.model == m]
        md += [f"\n## {m} vs {a.baseline}\n",
               "| Metric | Better | Baseline | Model | Diff | 95% interval | Seeds won | p (sign) | Verdict |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            lab, better, fmt = labels[r.metric]
            md.append(f"| {lab} | {better} | {fmt.format(r.baseline_mean)} | {fmt.format(r.model_mean)} | {r.mean_diff:+.3g} ({r.pct_diff:+.1f}%) | "
                      f"{r.t_ci_lo:+.3g} to {r.t_ci_hi:+.3g} | {int(r.seeds_won)}/{int(r.n_seeds)} | {r.sign_test_p:.3f} | {r.verdict} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
