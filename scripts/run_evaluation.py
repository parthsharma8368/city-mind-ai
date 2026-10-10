"""
run_evaluation.py — CLI entry point for paired evaluation.

Usage:
  python scripts/run_evaluation.py \
    --sumo-cfg data/raw/intas/scenario/InTAS_train.sumocfg \
    --tls-json artifacts/scenario/tls_metadata.json \
    [--ppo-model artifacts/experiments/ppo/run_.../model_final.zip] \
    [--ippo-model ...] [--phase-select-model ...] [--controllers sumo_actuated ippo_phase_select]

Output location: --output-dir > the eval config's output_dir > artifacts/metrics
(canonical 07:00-08:00 window only; any other window goes to
artifacts/metrics/runs/b<begin>_d<duration>/). Existing results of a different
or unknown configuration are never overwritten without --overwrite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate controllers on InTAS/SUMO")
    p.add_argument("--sumo-cfg", required=True)
    p.add_argument("--tls-json", required=True)
    p.add_argument("--eval-config", default="configs/evaluation.yaml")
    p.add_argument("--ppo-model", default=None)
    p.add_argument("--ippo-model", default=None)
    p.add_argument("--phase-select-model", default=None, help="MaskablePPO phase-selection IPPO model.zip")
    p.add_argument("--controllers", nargs="+", default=None,
                   help="subset to run (default: the 3 baselines + every controller with a model path)")
    p.add_argument("--output-dir", default=None, help="default: eval config output_dir, else artifacts/metrics")
    p.add_argument("--trajectory-dir", default=None, help="default: eval config trajectory_dir")
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing results of a different/unknown configuration")
    p.add_argument("--no-drain", action="store_true", help="skip the drain-aware metrics (not recommended)")
    p.add_argument("--plot", action="store_true", help="Generate comparison plots")
    return p.parse_args()


def main():
    args = parse_args()
    from src.evaluation.evaluate import EvaluationRunner
    from src.evaluation.statistics import print_comparison_table, load_results

    runner = EvaluationRunner(
        sumo_cfg=args.sumo_cfg,
        tls_metadata_json=args.tls_json,
        eval_config_path=args.eval_config,
        ppo_model_path=args.ppo_model,
        ippo_model_path=args.ippo_model,
        ippo_phase_select_model_path=args.phase_select_model,
        output_dir=args.output_dir,
        trajectory_dir=args.trajectory_dir,
        controllers=args.controllers,
        overwrite=args.overwrite,
        compute_drain=not args.no_drain,
    )
    runner.run_all()

    # Print comparison table (a missing/unreadable parquet is an error, not a skipped step)
    results_parquet = str(runner.output_dir / "evaluation_results.parquet")
    print_comparison_table(load_results(results_parquet))

    if args.plot:
        from src.evaluation.compare import plot_comparison_bars
        plot_comparison_bars(results_parquet, output_dir=runner.cfg.get("report_dir", "artifacts/reports"))


if __name__ == "__main__":
    main()
