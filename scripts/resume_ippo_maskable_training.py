"""
resume_ippo_maskable_training.py
=================================
Continues phase-selection IPPO training from a saved checkpoint after an
interruption (the process was killed by something outside the run itself
— e.g. a machine restart — not a normal completion). Not a substitute for
run_ippo_maskable_training.py on a fresh run.

Usage:
  python scripts/resume_ippo_maskable_training.py \
    --sumo-cfg data/raw/intas/scenario/InTAS_train.sumocfg \
    --tls-json artifacts/scenario/tls_metadata.json \
    --checkpoint artifacts/experiments/ippo_phase_select/run_.../checkpoints/ippo_phase_select_2822400_steps.zip \
    --run-dir artifacts/experiments/ippo_phase_select/run_...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resume phase-selection IPPO training from a checkpoint")
    p.add_argument("--sumo-cfg", required=True)
    p.add_argument("--tls-json", required=True)
    p.add_argument("--checkpoint", required=True, help="Path to the checkpoint .zip to resume from")
    p.add_argument("--run-dir", required=True,
                   help="The ORIGINAL run's directory (artifacts/experiments/ippo_phase_select/run_...) "
                        "— new checkpoints/model_final.zip are written here, continuing that run "
                        "rather than starting a new run_<timestamp> directory.")
    p.add_argument("--ppo-config", default="configs/ippo_phase_select.yaml")
    p.add_argument("--env-config", default="configs/environment_phase_select.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demand-id", default="1.0x")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src.agents.ippo_maskable_agent import IPPOMaskableAgent

    run_dir = Path(args.run_dir)
    agent = IPPOMaskableAgent(
        sumo_cfg=args.sumo_cfg,
        tls_metadata_json=args.tls_json,
        ppo_config_path=args.ppo_config,
        env_config_path=args.env_config,
        seed=args.seed,
        demand_id=args.demand_id,
        output_dir=str(run_dir.parent),
    )
    # IPPOMaskableAgent.__init__ always creates a fresh timestamped run_dir;
    # override it to continue writing into the ORIGINAL run's directory
    # instead, so checkpoints/model_final.zip land alongside the ones
    # already there rather than starting a new run_<timestamp> folder.
    agent.run_dir = run_dir
    agent.run_dir.mkdir(parents=True, exist_ok=True)

    agent.resume_training(args.checkpoint)


if __name__ == "__main__":
    main()
