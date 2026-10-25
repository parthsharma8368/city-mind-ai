"""
rescore_drain_metrics.py — re-score existing tripinfo artifacts with the
drain-aware protocol. ZERO simulation: reads artifacts/metrics/tripinfo/*.xml
(+ the metrics JSONs next to them, + the scenario route/net files).

Writes (next to the metrics, API-safe: the JSON is not a controller record):
  artifacts/metrics/drain_scores.json   full scores incl. entry-edge/cohort tables
  artifacts/metrics/drain_scores.csv    one flat row per (controller, seed)
  artifacts/metrics/drain_scores.md     per-controller markdown table

Usage:
  python scripts/rescore_drain_metrics.py
  python scripts/rescore_drain_metrics.py --sumo-cfg data/raw/intas/scenario/InTAS_train.sumocfg \
      --metrics-dir artifacts/metrics --begin 25200 --end 28800
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

CTRL_ORDER = ["sumo_actuated", "fixed_time", "max_pressure", "ppo", "ippo", "ippo_phase_select"]
_NAME = re.compile(r"^(?P<ctrl>.+)_seed(?P<seed>\d+)_(?P<demand>[\d.]+x)\.xml$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline drain-aware re-scoring of stored tripinfo files")
    p.add_argument("--metrics-dir", default="artifacts/metrics", help="dir with tripinfo/ and the per-run JSONs")
    p.add_argument("--tripinfo-dir", default=None, help="default: <metrics-dir>/tripinfo")
    p.add_argument("--trajectory-dir", default="artifacts/trajectories",
                   help="optional parquet trajectories (seed 42 only) to split never-inserted vs in-network")
    p.add_argument("--sumo-cfg", default="data/raw/intas/scenario/InTAS_train.sumocfg",
                   help="used only to locate the net + route files (identical in InTAS_train/InTAS_buildings)")
    p.add_argument("--begin", type=float, default=25200.0)
    p.add_argument("--end", type=float, default=28800.0)
    p.add_argument("--demand", default="1.0x")
    p.add_argument("--cache-dir", default=None, help="default: <metrics-dir>/_drain_cache")
    p.add_argument("--out-prefix", default=None, help="default: <metrics-dir>/drain_scores")
    p.add_argument("--baseline", default="sumo_actuated", help="baseline for the paired-by-seed comparison")
    p.add_argument("--no-trajectories", action="store_true")
    p.add_argument("--default-calibration", action="store_true",
                   help="use the EDA's free-flow calibration constants instead of re-fitting on the tripinfo")
    return p.parse_args()


def _md_table(summary: pd.DataFrame) -> str:
    cols = [("controller", "Controller"), ("lag_mean", "Lag/scheduled car (s)"), ("lag_sd", "SD"),
            ("due_mean", "Due-completion"), ("cars_nz_mean", "Completed (non-zero cars)"),
            ("never_ins_mean", "Never inserted"), ("tele_mean", "Teleports"), ("tl_nz_mean", "TimeLoss non-zero (s)")]
    head = "| " + " | ".join(h for _, h in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for _, r in summary.iterrows():
        cells = []
        for k, _ in cols:
            v = r[k]
            if k == "controller":
                cells.append(str(v))
            elif k == "due_mean":
                cells.append(f"{v:.3f}")
            elif isinstance(v, float) and v != v:
                cells.append("n/a")
            elif k in ("lag_mean", "lag_sd", "tl_nz_mean", "tele_mean"):
                cells.append(f"{v:.1f}")
            else:
                cells.append(f"{v:.0f}")
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def main() -> None:
    args = parse_args()
    from src.evaluation.drain_metrics import (
        PROTOCOL_ID, build_scheduled_demand, score_tripinfo_file, trips_frame,
    )
    from src.environment.schema import METRICS_SCHEMA_VERSION
    from src.evaluation.provenance import read_sumocfg
    from src.evaluation.statistics import paired_by_seed_comparison

    mdir = Path(args.metrics_dir)
    tdir = Path(args.tripinfo_dir) if args.tripinfo_dir else mdir / "tripinfo"
    cache = Path(args.cache_dir) if args.cache_dir else mdir / "_drain_cache"
    prefix = Path(args.out_prefix) if args.out_prefix else mdir / "drain_scores"

    files = sorted(f for f in tdir.glob("*.xml") if (m := _NAME.match(f.name)) and m["demand"] == args.demand)
    if not files:
        raise SystemExit(f"no tripinfo files for demand {args.demand} in {tdir}")

    cfg = read_sumocfg(args.sumo_cfg)
    print(f"[rescore] {len(files)} tripinfo files; scenario {args.sumo_cfg}")

    # Strict parse of every file first (fails loudly on a truncated/empty file).
    trips = {}
    for f in files:
        m = _NAME.match(f.name)
        trips[(m["ctrl"], int(m["seed"]))] = trips_frame(f)
    calib_trips = pd.concat(
        [t[~t.is_bus][["id", "rerouteNo", "routeLength", "duration", "timeLoss"]] for t in trips.values()],
        ignore_index=True,
    )

    print("[rescore] building scheduled-demand table (cached) ...")
    demand = build_scheduled_demand(
        cfg["route_files"], cfg["net_file"], args.begin, args.end,
        calibration_trips=None if args.default_calibration else calib_trips, cache_dir=cache,
    )
    print(f"[rescore] scheduled cars={demand.n_cars} (zero-length {int(demand.cars.zero_len.sum())}), "
          f"buses={demand.scheduled_buses}; free-flow calibration a={demand.calibration['a']:.4f} "
          f"c={demand.calibration['c']:.2f} ({demand.calibration_source})")

    runs = []
    for f in files:
        m = _NAME.match(f.name)
        ctrl, seed = m["ctrl"], int(m["seed"])
        jpath = mdir / f"{ctrl}_seed{seed}_{args.demand}.json"
        j = json.loads(jpath.read_text()) if jpath.exists() else {}
        traj = Path(args.trajectory_dir) / f"{ctrl}_seed{seed}_{args.demand}.parquet"
        score = score_tripinfo_file(
            demand, f,
            trajectory_path=None if args.no_trajectories else traj,
            inserted=j.get("total_departed"), json_teleports=j.get("total_teleports"), backlog=j.get("backlog"),
        )
        # score_tripinfo_file parsed the file again; reuse the in-memory frame for nothing else.
        runs.append({"controller": ctrl, "seed": seed, "demand_id": args.demand,
                     "source_metrics_schema_version": j.get("metrics_schema_version", 1), **score})
        print(f"  {ctrl:18s} seed {seed:3d}  lag/car={score['lag_per_scheduled_car_s']:.1f}s  "
              f"due-done={score['due_completion_ratio']:.4f}  never_inserted={score.get('never_inserted')}")

    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in runs])
    df["ctrl_rank"] = df.controller.map({c: i for i, c in enumerate(CTRL_ORDER)}).fillna(99)
    df = df.sort_values(["ctrl_rank", "seed"]).drop(columns="ctrl_rank")
    df.to_csv(f"{prefix}.csv", index=False)

    g = df.groupby("controller")
    summary = pd.DataFrame({
        "lag_mean": g.lag_per_scheduled_car_s.mean(), "lag_sd": g.lag_per_scheduled_car_s.std(),
        "due_mean": g.due_completion_ratio.mean(), "cars_nz_mean": g.cars_completed_nonzero.mean(),
        "never_ins_mean": g.never_inserted.mean() if "never_inserted" in df else float("nan"),
        "tele_mean": g.teleports.mean() if "teleports" in df else float("nan"),
        "tl_nz_mean": g.mean_timeloss_cars_nonzero.mean(),
        "n_seeds": g.size(),
    }).reset_index()
    summary["o"] = summary.controller.map({c: i for i, c in enumerate(CTRL_ORDER)}).fillna(99)
    summary = summary.sort_values("o").drop(columns="o")

    paired = {}
    for metric in ("lag_per_scheduled_car_s", "due_completion_ratio"):
        if args.baseline in set(df.controller):
            paired[metric] = paired_by_seed_comparison(df, metric, args.baseline).to_dict("records")

    table = _md_table(summary)
    md = (
        f"# Drain-aware scores ({PROTOCOL_ID})\n\n"
        f"Window {args.begin:.0f}-{args.end:.0f} s, {demand.n_cars} scheduled cars "
        f"({int(demand.cars.zero_len.sum())} zero-length), n = {int(summary.n_seeds.min())} evaluation seeds per "
        "controller (simulation noise only; one trained model per RL controller). Lag/scheduled car = mean over ALL "
        "scheduled cars of (actual completion, or 08:00 if unfinished/never inserted) minus (zero-delay completion "
        "time); lower is better.\n\n" + table
    )
    Path(f"{prefix}.md").write_text(md)
    out = {
        "drain_protocol": PROTOCOL_ID, "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "source_metrics_schema_versions": sorted({r["source_metrics_schema_version"] for r in runs}),
        "window": [args.begin, args.end], "scenario_sumocfg": args.sumo_cfg,
        "scheduled": {"cars": demand.n_cars, "zero_length_cars": int(demand.cars.zero_len.sum()),
                      "buses": demand.scheduled_buses},
        "ff_calibration": {**demand.calibration, "source": demand.calibration_source},
        "summary": summary.to_dict("records"), "paired_vs_baseline": paired, "runs": runs,
    }
    Path(f"{prefix}.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n" + md)
    print(f"[rescore] wrote {prefix}.json / .csv / .md")


if __name__ == "__main__":
    main()
