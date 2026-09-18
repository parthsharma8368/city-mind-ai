"""
run_visual_demo.py
==================
Visual demo: Load trained PPO model and run with sumo-gui so you can
watch the RL agent controlling traffic lights in real-time on the Ingolstadt map.

Usage:
  python scripts/run_visual_demo.py [--model path/to/model_final.zip] [--delay 100]

When SUMO-GUI opens:
  1. Press the green ▶ Play button (or Ctrl+A) to start
  2. Use View > Zoom to see individual intersections
  3. Colored circles on roads = traffic signals (Green/Yellow/Red)
  4. The PPO agent sends new actions every 5 simulation seconds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default model path (smoke-test model — replace with fully trained one)
DEFAULT_MODEL = (
    "artifacts/experiments/ppo/run_20260917_174650_seed42/model_final.zip"
)
SUMO_CFG = "data/raw/intas/scenario/InTAS_train.sumocfg"
TLS_JSON = "artifacts/scenario/tls_metadata.json"


def parse_args():
    p = argparse.ArgumentParser(description="Visual SUMO-GUI demo with PPO agent")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to trained PPO model .zip")
    p.add_argument("--sumo-cfg", default=SUMO_CFG, help="Path to .sumocfg")
    p.add_argument("--tls-json", default=TLS_JSON, help="Path to tls_metadata.json")
    p.add_argument("--delay", type=int, default=100,
                   help="Milliseconds per SUMO step (100 = watchable, 0 = fast)")
    p.add_argument("--duration", type=int, default=3600, help="Episode duration (seconds)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("CityMind AI — Live Visual Simulation Demo")
    print("=" * 60)
    print(f"  Model   : {args.model}")
    print(f"  Scenario: {args.sumo_cfg}")
    print(f"  Delay   : {args.delay}ms per step")
    print(f"  Duration: {args.duration}s episode")
    print("=" * 60 + "\n")

    # ── Resolve PPO model path (with auto-discovery) ──────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        # Search for any trained model in artifacts/experiments/
        candidates = list(Path("artifacts/experiments").glob("**/model_final.zip")) + \
                     list(Path("artifacts/experiments").glob("**/*.zip"))
        if candidates:
            # Pick latest modified
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = candidates[0]
            print(f"[Demo] Specified model not found. Auto-discovered latest: {model_path}")
        else:
            raise FileNotFoundError(f"No PPO model zip found at {args.model} or in artifacts/experiments/")

    from stable_baselines3 import PPO
    model = PPO.load(str(model_path), device="cpu")
    print(f"[Demo] PPO model loaded successfully: {model_path}")

    # ── Build environment with sumo-gui ───────────────────────────────────────
    from src.environment.centralized_sumo_env import CentralizedSumoEnv

    sumo_extra = ["--end", str(args.duration)]
    viewsettings_path = Path("configs/viewsettings.xml")
    if viewsettings_path.exists():
        sumo_extra += ["-g", str(viewsettings_path), "--start"]
    if args.delay > 0:
        sumo_extra += ["--delay", str(args.delay)]

    env = CentralizedSumoEnv(
        sumo_cfg=args.sumo_cfg,
        tls_metadata_json=args.tls_json,
        sumo_binary="sumo-gui",      # <── KEY: opens visual window
        decision_interval=5,
        controller="rl",
        seed=args.seed,
        demand_id="1.0x",
        record_trajectories=False,
        sumo_extra_args=sumo_extra,
    )

    print("\n[Demo] SUMO-GUI is launching...")
    print("[Demo] → When the window opens, press ▶ Play (or Ctrl+A) to start\n")

    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    while not done:
        # PPO policy inference
        action, _ = model.predict(obs, deterministic=True)

        # Step the environment (SUMO advances 5 seconds)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step += 1
        total_reward += reward

        if step % 20 == 0:
            print(
                f"  [t={info['sim_time']:>6.0f}s]  step={step:>4d}  "
                f"reward={reward:>8.2f}  cumulative={total_reward:>10.2f}  "
                f"switches={info.get('switch_count', 0)}"
            )

    env.close()

    print(f"\n[Demo] Episode complete after {step} steps")
    print(f"[Demo] Total cumulative reward: {total_reward:.2f}")
    print("[Demo] SUMO-GUI window can be closed now.")


if __name__ == "__main__":
    main()
