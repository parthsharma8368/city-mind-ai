"""
visualize_simulation.py
=======================
Launch interactive Eclipse SUMO GUI (sumo-gui) with real-time controller control.

Allows you to visually watch:
  1. The trained PPO Reinforcement Learning agent controlling the city lights.
  2. The Adaptive Max-Pressure controller dynamically clearing queues.
  3. The Fixed-Time baseline running pre-timed cycles.

Usage:
  python scripts/visualize_simulation.py --controller ppo
  python scripts/visualize_simulation.py --controller max_pressure
  python scripts/visualize_simulation.py --controller fixed_time
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.environment.centralized_sumo_env import CentralizedSumoEnv
from src.scenario.tls_metadata import load_tls_specs


def parse_args():
    parser = argparse.ArgumentParser(description="Run live simulation in Eclipse SUMO GUI.")
    parser.add_argument(
        "--controller",
        choices=["ppo", "max_pressure", "fixed_time", "random"],
        default="ppo",
        help="Controller to execute: ppo | max_pressure | fixed_time | random"
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/experiments/ppo_verification/run_20260916_235728_seed42/model_final.zip",
        help="Path to trained PPO model checkpoint (.zip)"
    )
    parser.add_argument(
        "--sumo-cfg",
        default="data/raw/intas/scenario/InTAS_buildings.sumocfg",
        help="Path to SUMO configuration file"
    )
    parser.add_argument(
        "--tls-json",
        default="artifacts/scenario/tls_metadata.json",
        help="Path to TLS metadata JSON"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=50,
        help="Delay between simulation steps in milliseconds (50ms gives smooth visual playback)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Maximum decision steps to simulate (default: 1000 = 5000s sim time)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Simulation random seed"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sumo_cfg = str(Path(args.sumo_cfg).resolve())
    tls_json = str(Path(args.tls_json).resolve())

    print("=" * 70)
    print(" CityMind AI — Interactive Eclipse SUMO GUI Visualizer")
    print("=" * 70)
    print(f" Controller : {args.controller.upper()}")
    print(f" SUMO Config: {sumo_cfg}")
    print(f" Visual Delay: {args.delay} ms per step (adjustable in GUI toolbar)")
    print("=" * 70)

    # Initialize environment with GUI binary and visual startup flags
    viewsettings_path = str(root / "configs/viewsettings.xml")
    sumo_extra_args = [
        "--start",                            # Auto-start simulation on launch
        "--delay", str(args.delay),           # Visual step delay
        "--quit-on-end", "false",             # Keep GUI open if simulation ends
        "--window-size", "1400,900",          # Window dimension
        "--gui-settings-file", viewsettings_path, # Perfect center & zoom
    ]

    env = CentralizedSumoEnv(
        sumo_cfg=sumo_cfg,
        tls_metadata_json=tls_json,
        sumo_binary="sumo-gui",     # Eclipse SUMO GUI
        decision_interval=5,
        step_length=1.0,
        controller=args.controller,
        seed=args.seed,
        sumo_extra_args=sumo_extra_args,
    )

    agent = None
    if args.controller == "ppo":
        model_file = Path(args.model_path)
        if not model_file.exists():
            # Fallback check other checkpoints
            alt = Path("artifacts/experiments/ppo_cli_test/run_20260917_000931_seed42/model_final.zip")
            if alt.exists():
                model_file = alt

        if model_file.exists():
            print(f" Loading PPO model weights from: {model_file}")
            from src.agents.ppo_agent import PPOAgent
            agent = PPOAgent(sumo_cfg=sumo_cfg, tls_metadata_json=tls_json, seed=args.seed)
            agent.load(str(model_file))
        else:
            print(f" [Warning] Checkpoint {model_file} not found. Running with policy initialization.")

    elif args.controller == "max_pressure":
        print(" Using Adaptive Max-Pressure queue differential controller.")
        specs = load_tls_specs(Path(tls_json))
        specs_by_id = {s.tls_id: s for s in specs}

    print("\n Launching Eclipse SUMO GUI window...")
    print(" [Tip] Use your mouse scroll-wheel to zoom into any intersection.")
    print(" [Tip] Click on any vehicle or traffic light to inspect live speeds & phases.")
    print(" Press Ctrl+C in this terminal to stop at any time.\n")

    try:
        obs, info = env.reset()
        sim_time = 0.0

        for step in range(args.steps):
            if args.controller == "ppo" and agent is not None:
                action, _ = agent.predict(obs, deterministic=True)
            elif args.controller == "max_pressure":
                # Compute max pressure actions
                from src.environment.sumo_connection import traci
                action = []
                for spec in env.tls_specs:
                    best_phase = 0
                    max_p = -1e9
                    for p_idx, p in enumerate(spec.phases):
                        if "G" not in p.state:
                            continue
                        press = 0.0
                        for lane_id in spec.controlled_lanes:
                            try:
                                press += traci.lane.getLastStepHaltingNumber(lane_id)
                            except Exception:
                                pass
                        if press > max_p:
                            max_p = press
                            best_phase = p_idx
                    curr_phase = env._tls_states[spec.tls_id].current_phase
                    action.append(1 if best_phase != curr_phase else 0)
                import numpy as np
                action = np.array(action, dtype=np.int8)
            elif args.controller == "fixed_time":
                # Fixed time relies on internal static timing (keep action = 0)
                import numpy as np
                action = np.zeros(env.num_tls, dtype=np.int8)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            sim_time = info.get("sim_time", step * 5.0)

            # Print telemetry summary every 10 steps (50s simulation time)
            if step % 10 == 0:
                from src.environment.sumo_connection import traci
                try:
                    active_v = traci.vehicle.getIDCount()
                    mean_spd = sum(traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()[:50]) / max(min(active_v, 50), 1)
                except Exception:
                    active_v, mean_spd = 0, 0.0

                print(
                    f" Step {step:4d} | Sim Time: {sim_time:6.1f}s | "
                    f"Active Cars: {active_v:3d} | Mean Speed: {mean_spd:5.2f} m/s | "
                    f"Step Reward: {reward:+8.2f}"
                )

            if terminated or truncated:
                print(f"\n Episode completed at simulation time {sim_time:.1f}s.")
                break

    except KeyboardInterrupt:
        print("\n User requested stop (Ctrl+C).")
    except Exception as e:
        print(f"\n GUI session ended: {e}")
    finally:
        env.close()
        print(" SUMO GUI session closed cleanly.\n")


if __name__ == "__main__":
    main()
