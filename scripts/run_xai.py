"""
run_xai.py
==========
CLI entry point for running Explainable AI (XAI) analysis on trained
CityMind AI traffic controllers — both the centralized PPO model and the
parameter-shared decentralized IPPO model.

Usage:
  python scripts/run_xai.py --controller ppo \
    --model artifacts/experiments/ppo/run_20260917_172613_seed42/model_final.zip \
    --tls-json artifacts/scenario/tls_metadata.json

  python scripts/run_xai.py --controller ippo \
    --model artifacts/experiments/ippo/run_20260919_134931_seed42/model_final.zip \
    --tls-json artifacts/scenario/tls_metadata.json \
    --trajectory artifacts/trajectories/ippo_seed42_1.0x.parquet \
    --geojson data/processed/network.geojson
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.evaluation.xai import (
    XAIExplainer, IPPOExplainer, compute_spillback_correlation,
    plot_top_drivers, plot_coordination_heatmap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run XAI analysis on a trained CityMind AI traffic controller")
    p.add_argument("--controller", choices=["ppo", "ippo"], default="ppo", help="Which trained model to explain")
    p.add_argument("--model", required=True, help="Path to trained model.zip")
    p.add_argument("--tls-json", default="artifacts/scenario/tls_metadata.json", help="Path to tls_metadata.json")
    p.add_argument("--tls-id", default=None, help="Specific TLS ID to explain (default: top 3 key junctions)")
    p.add_argument("--output-dir", default=None, help="Directory for plots and reports (default: artifacts/reports/xai for ppo, artifacts/reports/xai/ippo for ippo)")
    p.add_argument("--steps", type=int, default=15, help="Integrated gradients steps")
    p.add_argument("--trajectory", default=None, help="IPPO only: recorded trajectory parquet, for the empirical spillback-correlation heatmap")
    p.add_argument("--geojson", default="data/processed/network.geojson", help="IPPO only: network GeoJSON, for TLS positions")
    p.add_argument("--k-neighbors", type=int, default=3, help="IPPO only: nearest-neighbor count for the spillback heatmap")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/reports/xai") / ("ippo" if args.controller == "ippo" else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CityMind AI — Explainable AI (XAI) Policy Analyzer [{args.controller.upper()}]")
    print(f"{'='*60}\n")
    print(f"Loading model: {args.model}")
    print(f"Loading TLS metadata: {args.tls_json}")

    reports = []

    if args.controller == "ppo":
        explainer = XAIExplainer(model_path=args.model, tls_json_path=args.tls_json)
        # Representative synthetic observation (not a recorded one — same
        # simplification the original PPO XAI analysis has always used).
        obs = np.random.uniform(0.1, 0.4, size=(explainer.num_features,)).astype(np.float32)
        target_tls = [args.tls_id] if args.tls_id else explainer.tls_ids[:3]

        for tid in target_tls:
            print(f"\nExplaining decision for Junction {tid}...")
            explanation = explainer.explain_junction(obs=obs, tls_id=tid, top_k=8, method="integrated_gradients")
            reports.append(explanation)
            print(f"Decision: {'SWITCH' if explanation['action'] == 1 else 'HOLD'}")
            print(f"Narrative: {explanation['narrative']}")
            plot_path = out_dir / f"drivers_{tid}.png"
            plot_top_drivers(explanation, str(plot_path))
            print(f"Saved driver plot: {plot_path}")

        print("\nComputing full network inter-junction coordination matrix (model attribution)...")
        coord_mat = explainer.compute_coordination_matrix(obs, method="input_x_gradient")
        coord_plot_path = out_dir / "coordination_heatmap.png"
        plot_coordination_heatmap(
            explainer.tls_ids, coord_mat, str(coord_plot_path), top_k=20,
            title="Top 20 Inter-Junction Coordination Influence Matrix (PPO, model attribution)",
            cmap="magma", colorbar_label="Decision Influence Weight",
        )
        print(f"Saved coordination heatmap: {coord_plot_path}")

    else:
        explainer = IPPOExplainer(model_path=args.model, tls_json_path=args.tls_json)
        obs = np.random.uniform(0.1, 0.4, size=(explainer.num_features,)).astype(np.float32)
        target_tls = [args.tls_id] if args.tls_id else explainer.tls_ids[:3]

        for tid in target_tls:
            print(f"\nExplaining decision for Junction {tid}...")
            explanation = explainer.explain_junction_local(obs=obs, tls_id=tid, top_k=8, method="integrated_gradients")
            reports.append(explanation)
            print(f"Decision: {'SWITCH' if explanation['action'] == 1 else 'HOLD'}")
            print(f"Narrative: {explanation['narrative']}")
            plot_path = out_dir / f"drivers_{tid}.png"
            plot_top_drivers(explanation, str(plot_path))
            print(f"Saved driver plot: {plot_path}")

        if args.trajectory:
            print("\nComputing empirical spillback correlation (recorded switch-timing, not model attribution)...")
            tls_ids, corr_mat = compute_spillback_correlation(
                trajectory_path=args.trajectory, network_geojson_path=args.geojson, k_neighbors=args.k_neighbors,
            )
            coord_plot_path = out_dir / "coordination_heatmap.png"
            plot_coordination_heatmap(
                tls_ids, corr_mat, str(coord_plot_path), top_k=20,
                title="Empirical Spillback Correlation — Nearest-Neighbor Switch Timing (IPPO, recorded outcomes)",
                cmap="RdBu_r", colorbar_label="Pearson Correlation of Switch Events",
            )
            print(f"Saved spillback correlation heatmap: {coord_plot_path}")
        else:
            print("\n[Skipped] No --trajectory given — pass a recorded IPPO trajectory parquet to also generate the spillback correlation heatmap.")

    json_path = out_dir / "xai_summary.json"
    with open(json_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"Saved XAI structured summary: {json_path}")

    print(f"\n[XAI] Analysis complete. All artifacts written to: {out_dir}\n")


if __name__ == "__main__":
    main()
