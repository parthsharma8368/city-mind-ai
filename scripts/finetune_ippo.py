"""Continue training a TRAINED keep/advance IPPO model under a (slightly) different environment config, e.g. one extra
observation switch whose input size is unchanged. The init model's weights AND optimizer state are kept; training runs for
``total_timesteps`` MORE steps from the ppo config, into a fresh run directory.

  python scripts/finetune_ippo.py --init-model <model_final.zip> --sumo-cfg ... --tls-json ... \
      --ppo-config configs/ippo_finetune.yaml --env-config configs/environment_pending.yaml \
      --output-dir artifacts/experiments/finetune --seed 2042
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune a trained IPPO model")
    p.add_argument("--init-model", required=True)
    p.add_argument("--sumo-cfg", required=True)
    p.add_argument("--tls-json", required=True)
    p.add_argument("--ppo-config", required=True)
    p.add_argument("--env-config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=2042, help="SUMO traffic seeds start here (use one the init run never used)")
    p.add_argument("--demand-id", default="1.0x")
    a = p.parse_args()
    from src.agents.ippo_agent import IPPOAgent

    agent = IPPOAgent(sumo_cfg=a.sumo_cfg, tls_metadata_json=a.tls_json, ppo_config_path=a.ppo_config,
                      env_config_path=a.env_config, seed=a.seed, demand_id=a.demand_id, output_dir=a.output_dir)
    # completed_timesteps=0 -> train total_timesteps MORE steps, keeping the loaded model's own step counter.
    agent.resume_training(a.init_model, completed_timesteps=0)


if __name__ == "__main__":
    main()
