"""
run_xai.py
==========
CLI entry point for running Explainable AI (XAI) analysis on trained PPO traffic controllers.

Usage:
  python scripts/run_xai.py \
    --model artifacts/experiments/ppo/run_20260917_172613_seed42/model_final.zip \
    --tls-json artifacts/scenario/tls_metadata.json \
    --output-dir artifacts/reports/xai
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.evaluation.xai import XAIExplainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run XAI analysis on trained PPO traffic controller")
    p.add_argument("--model", required=True, help="Path to trained PPO model.zip")
    p.add_argument("--tls-json", default="artifacts/scenario/tls_metadata.json", help="Path to tls_metadata.json")
    p.add_argument("--tls-id", default=None, help="Specific TLS ID to explain (default: top 3 key junctions)")
    p.add_argument("--output-dir", default="artifacts/reports/xai", help="Directory for plots and reports")
    p.add_argument("--steps", type=int, default=15, help="Integrated gradients steps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("CityMind AI — Explainable AI (XAI) Policy Analyzer")
    print(f"{'='*60}\n")
    print(f"Loading model: {args.model}")
    print(f"Loading TLS metadata: {args.tls_json}")

    explainer = XAIExplainer(
        model_path=args.model,
        tls_json_path=args.tls_json,
    )

    # Use a synthetic representative traffic observation
    # (High queue and density on inbound lanes, moderate elapsed phase)
    obs = np.random.uniform(0.1, 0.4, size=(explainer.num_features,)).astype(np.float32)

    target_tls = [args.tls_id] if args.tls_id else explainer.tls_ids[:3]

    reports = []
    for tid in target_tls:
        print(f"\nExplaining decision for Junction {tid}...")
        explanation = explainer.explain_junction(
            obs=obs,
            tls_id=tid,
            top_k=8,
            method="integrated_gradients",
        )
        reports.append(explanation)
        print(f"Decision: {'SWITCH' if explanation['action'] == 1 else 'HOLD'}")
        print(f"Narrative: {explanation['narrative']}")

        plot_path = out_dir / f"drivers_{tid}.png"
        explainer.plot_top_drivers(explanation, str(plot_path))
        print(f"Saved driver plot: {plot_path}")

    # Compute network coordination matrix
    print("\nComputing full network inter-junction coordination matrix...")
    coord_mat = explainer.compute_coordination_matrix(obs, method="input_x_gradient")
    coord_plot_path = out_dir / "coordination_heatmap.png"
    explainer.plot_coordination_heatmap(coord_mat, str(coord_plot_path), top_k=20)
    print(f"Saved coordination heatmap: {coord_plot_path}")

    # Save structured JSON
    json_path = out_dir / "xai_summary.json"
    with open(json_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"Saved XAI structured summary: {json_path}")

    print(f"\n[XAI] Analysis complete. All artifacts written to: {out_dir}\n")


if __name__ == "__main__":
    main()
