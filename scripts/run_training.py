"""
run_training.py
===============
CLI entry point for PPO training.

Usage:
  python scripts/run_training.py \
    --sumo-cfg data/raw/intas/ingolstadt.sumocfg \
    --tls-json artifacts/scenario/tls_metadata.json \
    --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on InTAS/SUMO")
    p.add_argument("--sumo-cfg", required=True, help="Path to .sumocfg")
    p.add_argument("--tls-json", required=True, help="Path to tls_metadata.json")
    p.add_argument("--ppo-config", default="configs/ppo.yaml")
    p.add_argument("--env-config", default="configs/environment.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demand-id", default="1.0x")
    p.add_argument("--output-dir", default="artifacts/experiments/ppo")
    p.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Multiple seeds (overrides --seed; runs sequentially)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src.agents.ppo_agent import PPOAgent

    seeds = args.seeds if args.seeds else [args.seed]

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"PPO Training — Seed {seed}")
        print(f"{'='*60}\n")

        agent = PPOAgent(
            sumo_cfg=args.sumo_cfg,
            tls_metadata_json=args.tls_json,
            ppo_config_path=args.ppo_config,
            env_config_path=args.env_config,
            seed=seed,
            demand_id=args.demand_id,
            output_dir=args.output_dir,
        )
        agent.train()
        print(f"\nSeed {seed} complete.\n")


if __name__ == "__main__":
    main()
