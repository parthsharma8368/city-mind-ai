"""
run_ippo_training.py
=====================
CLI entry point for parameter-shared IPPO training (DecentralizedSumoEnv).

Usage:
  python scripts/run_ippo_training.py \
    --sumo-cfg data/raw/intas/scenario/InTAS_buildings.sumocfg \
    --tls-json artifacts/scenario/tls_metadata.json \
    --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train parameter-shared IPPO on InTAS/SUMO")
    p.add_argument("--sumo-cfg", required=True)
    p.add_argument("--tls-json", required=True)
    p.add_argument("--ppo-config", default="configs/ippo.yaml")
    p.add_argument("--env-config", default="configs/environment.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demand-id", default="1.0x")
    p.add_argument("--output-dir", default="artifacts/experiments/ippo")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src.agents.ippo_agent import IPPOAgent

    agent = IPPOAgent(
        sumo_cfg=args.sumo_cfg,
        tls_metadata_json=args.tls_json,
        ppo_config_path=args.ppo_config,
        env_config_path=args.env_config,
        seed=args.seed,
        demand_id=args.demand_id,
        output_dir=args.output_dir,
    )
    agent.train()


if __name__ == "__main__":
    main()
