"""
run_visual_demo.py
==================
Visual demo: launch sumo-gui so you can watch a chosen controller driving
traffic lights in real time on the Ingolstadt map.

Usage:
  python scripts/run_visual_demo.py --controller fixed_time
  python scripts/run_visual_demo.py --controller max_pressure
  python scripts/run_visual_demo.py --controller ppo   [--model path/to/model_final.zip]
  python scripts/run_visual_demo.py --controller ippo  [--model path/to/model_final.zip]

Run two of these side by side (different --controller, same --seed) to
compare a specific intersection live between two SUMO-GUI windows.

When SUMO-GUI opens:
  1. Press the green ▶ Play button (or Ctrl+A) to start
  2. Use View > Zoom to see individual intersections
  3. Colored circles on roads = traffic signals (Green/Yellow/Red)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

SUMO_CFG = "data/raw/intas/scenario/InTAS_train.sumocfg"
TLS_JSON = "artifacts/scenario/tls_metadata.json"


def parse_args():
    p = argparse.ArgumentParser(description="Visual SUMO-GUI demo")
    p.add_argument(
        "--controller", default="ppo",
        choices=["fixed_time", "max_pressure", "ppo", "ippo"],
        help="Which controller drives the signals in the GUI",
    )
    p.add_argument("--model", default=None, help="Model .zip (ppo/ippo only; auto-discovers latest if omitted)")
    p.add_argument("--sumo-cfg", default=SUMO_CFG, help="Path to .sumocfg")
    p.add_argument("--tls-json", default=TLS_JSON, help="Path to tls_metadata.json")
    p.add_argument("--delay", type=int, default=100,
                   help="Milliseconds per SUMO step (100 = watchable, 0 = fast)")
    p.add_argument("--duration", type=int, default=3600, help="Episode duration (seconds)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_model(model_arg: str | None, experiments_subdir: str) -> Path:
    if model_arg:
        model_path = Path(model_arg)
        if model_path.exists():
            return model_path
        print(f"[Demo] Specified model not found at {model_path}, auto-discovering...")

    candidates = list(Path(f"artifacts/experiments/{experiments_subdir}").glob("**/model_final.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No {experiments_subdir} model_final.zip found. Pass --model explicitly."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"[Demo] Auto-discovered latest {experiments_subdir} model: {candidates[0]}")
    return candidates[0]


def _build_sumo_extra_args(args) -> list[str]:
    sumo_extra = ["--end", str(args.duration)]
    viewsettings_path = Path("configs/viewsettings.xml")
    if viewsettings_path.exists():
        sumo_extra += ["-g", str(viewsettings_path), "--start"]
    if args.delay > 0:
        sumo_extra += ["--delay", str(args.delay)]
    return sumo_extra


def run_fixed_time(args) -> None:
    from src.controllers.fixed_time import FixedTimeController

    controller = FixedTimeController(
        episode_duration=args.duration,
        sumo_binary="sumo-gui",
        sumo_extra_args=_build_sumo_extra_args(args),
    )
    print("\n[Demo] Fixed-Time baseline — SUMO's own static signal programs, no RL.\n")
    result = controller.run_episode(args.sumo_cfg, args.tls_json, seed=args.seed, demand_id="1.0x")
    print(f"\n[Demo] Episode complete. mean_wait={result.mean_waiting_time:.2f}s "
          f"mean_speed={result.mean_speed:.2f} m/s switches={result.total_switches}")


def run_max_pressure(args) -> None:
    from src.controllers.max_pressure import MaxPressureController

    controller = MaxPressureController(
        episode_duration=args.duration,
        sumo_binary="sumo-gui",
        sumo_extra_args=_build_sumo_extra_args(args),
    )
    print("\n[Demo] Max-Pressure baseline — decentralized greedy-queue signal control.\n")
    result = controller.run_episode(args.sumo_cfg, args.tls_json, seed=args.seed, demand_id="1.0x")
    print(f"\n[Demo] Episode complete. mean_wait={result.mean_waiting_time:.2f}s "
          f"mean_speed={result.mean_speed:.2f} m/s switches={result.total_switches}")


def run_ppo(args) -> None:
    from stable_baselines3 import PPO
    from src.environment.centralized_sumo_env import CentralizedSumoEnv

    model_path = _resolve_model(args.model, "ppo")
    model = PPO.load(str(model_path), device="cpu")
    print(f"[Demo] Centralized PPO model loaded: {model_path}")

    env = CentralizedSumoEnv(
        sumo_cfg=args.sumo_cfg,
        tls_metadata_json=args.tls_json,
        sumo_binary="sumo-gui",
        decision_interval=5,
        controller="rl",
        seed=args.seed,
        demand_id="1.0x",
        record_trajectories=False,
        sumo_extra_args=_build_sumo_extra_args(args),
    )

    print("\n[Demo] Centralized PPO — one policy jointly deciding all 98 intersections.\n")
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        total_reward += reward
        if step % 20 == 0:
            print(f"  [t={info['sim_time']:>6.0f}s] step={step:>4d} reward={reward:>8.2f} "
                  f"cumulative={total_reward:>10.2f} switches={info.get('switch_count', 0)}")
    env.close()
    print(f"\n[Demo] Episode complete after {step} steps. Total reward: {total_reward:.2f}")


def run_ippo(args) -> None:
    from stable_baselines3 import PPO
    from src.environment.decentralized_sumo_env import DecentralizedSumoEnv

    model_path = _resolve_model(args.model, "ippo")
    model = PPO.load(str(model_path), device="cpu")
    print(f"[Demo] Parameter-shared IPPO model loaded: {model_path}")

    env = DecentralizedSumoEnv(
        sumo_cfg=args.sumo_cfg,
        tls_metadata_json=args.tls_json,
        sumo_binary="sumo-gui",
        decision_interval=5,
        seed=args.seed,
        demand_id="1.0x",
        episode_duration=args.duration,
        sumo_extra_args=_build_sumo_extra_args(args),
    )

    print("\n[Demo] Parameter-shared IPPO — one shared policy, 98 independent local decisions.\n")
    observations, _ = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    while not done:
        actions = {
            tid: int(model.predict(observations[tid], deterministic=True)[0])
            for tid in env.agents
        }
        observations, rewards, terminations, truncations, _ = env.step(actions)
        step_reward = sum(rewards.values())
        total_reward += step_reward
        step += 1
        done = not env.agents
        if step % 20 == 0:
            print(f"  [step={step:>4d}] step_reward={step_reward:>10.2f} cumulative={total_reward:>12.2f}")
    env.close()
    print(f"\n[Demo] Episode complete after {step} steps. Total reward: {total_reward:.2f}")


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("CityMind AI — Live Visual Simulation Demo")
    print("=" * 60)
    print(f"  Controller: {args.controller}")
    print(f"  Scenario  : {args.sumo_cfg}")
    print(f"  Delay     : {args.delay}ms per step")
    print(f"  Duration  : {args.duration}s episode")
    print("=" * 60)

    {
        "fixed_time": run_fixed_time,
        "max_pressure": run_max_pressure,
        "ppo": run_ppo,
        "ippo": run_ippo,
    }[args.controller](args)

    print("[Demo] SUMO-GUI window can be closed now.")


if __name__ == "__main__":
    main()
