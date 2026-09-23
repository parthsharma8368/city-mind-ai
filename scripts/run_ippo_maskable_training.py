"""
run_ippo_maskable_training.py
==============================
CLI entry point for parameter-shared IPPO training with the phase-selection
action space + MaskablePPO (RL-strategy Phase 2).

Usage:
  python scripts/run_ippo_maskable_training.py \
    --sumo-cfg data/raw/intas/scenario/InTAS_train.sumocfg \
    --tls-json artifacts/scenario/tls_metadata.json \
    --ppo-config configs/ippo_phase_select_test.yaml \
    --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train parameter-shared, phase-selection IPPO on InTAS/SUMO")
    p.add_argument("--sumo-cfg", required=True)
    p.add_argument("--tls-json", required=True)
    p.add_argument("--ppo-config", default="configs/ippo_phase_select.yaml")
    p.add_argument("--env-config", default="configs/environment_phase_select.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demand-id", default="1.0x")
    p.add_argument("--output-dir", default="artifacts/experiments/ippo_phase_select")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src.agents.ippo_maskable_agent import IPPOMaskableAgent

    agent = IPPOMaskableAgent(
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
