"""
run_ippo_maskable_eval.py
==========================
Evaluates a trained MaskablePPO (phase-selection) IPPO model on the same
peak-window protocol as scripts/run_evaluation.py's other controllers.

This is a thin wrapper around EvaluationRunner(controllers=["ippo_phase_select"]):
the same loop, the same result schema (drain-aware metrics + provenance), the
same overwrite guard and the same output naming as every other controller, so
results are directly comparable to the canonical artifacts/metrics/ files and
cannot drift from them.

IMPORTANT: use the SAME --sumo-cfg as the baselines you compare against. The
stored 18 canonical runs were produced with InTAS_train.sumocfg (route-steps
100, no buildings polygons); InTAS_buildings.sumocfg has different dynamics.
The sumocfg sha256 is recorded in each result's provenance block.

Usage:
  python scripts/run_ippo_maskable_eval.py \
    --sumo-cfg data/raw/intas/scenario/InTAS_train.sumocfg \
    --tls-json artifacts/scenario/tls_metadata.json \
    --model artifacts/experiments/ippo_phase_select/run_.../model_final.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a phase-selection MaskablePPO IPPO model")
    p.add_argument("--sumo-cfg", required=True)
    p.add_argument("--tls-json", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--eval-config", default="configs/evaluation.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="override the eval config's seeds")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--trajectory-dir", default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="allow replacing results of a different/unknown configuration")
    return p.parse_args()


def main():
    args = parse_args()
    import numpy as np
    from src.evaluation.evaluate import EvaluationRunner

    runner = EvaluationRunner(
        sumo_cfg=args.sumo_cfg,
        tls_metadata_json=args.tls_json,
        eval_config_path=args.eval_config,
        ippo_phase_select_model_path=args.model,
        output_dir=args.output_dir,
        trajectory_dir=args.trajectory_dir,
        controllers=["ippo_phase_select"],
        overwrite=args.overwrite,
    )
    if args.seeds:
        runner.seeds = list(args.seeds)
    results = runner.run_all()

    print("\n" + "=" * 70)
    print(f"PHASE-SELECTION IPPO — {len(results)}-SEED SUMMARY (simulation noise only, one trained model)")
    print("=" * 70)
    for label, vals in (
        ("trips_completed", [r.trips_completed for r in results]),
        ("mean_trip_timeloss", [r.mean_trip_timeloss for r in results]),
        ("backlog", [r.backlog for r in results]),
        ("mean_waiting_time", [r.mean_waiting_time for r in results]),
    ):
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        print(f"{label}: {np.mean(vals):.2f} +/- {sd:.2f} (n={len(vals)})")
    print("Drain-aware scores: python scripts/rescore_drain_metrics.py")


if __name__ == "__main__":
    main()
