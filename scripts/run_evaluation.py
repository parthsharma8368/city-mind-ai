"""
run_evaluation.py — CLI entry point for paired evaluation.

Usage:
  python scripts/run_evaluation.py \
    --sumo-cfg data/raw/intas/ingolstadt.sumocfg \
    --tls-json artifacts/scenario/tls_metadata.json \
    [--ppo-model artifacts/experiments/ppo/run_.../model_final.zip]
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
    p.add_argument("--output-dir", default="artifacts/metrics")
    p.add_argument("--trajectory-dir", default="artifacts/trajectories")
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
        output_dir=args.output_dir,
        trajectory_dir=args.trajectory_dir,
    )
    results = runner.run_all()

    # Print comparison table
    results_parquet = f"{args.output_dir}/evaluation_results.parquet"
    try:
        df = load_results(results_parquet)
        print_comparison_table(df)
    except Exception as e:
        print(f"Could not print table: {e}")

    if args.plot:
        from src.evaluation.compare import plot_comparison_bars
        plot_comparison_bars(results_parquet, output_dir="artifacts/reports")


if __name__ == "__main__":
    main()
