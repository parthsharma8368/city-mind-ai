"""Watch one controller drive the InTAS network live in SUMO-GUI (the dashboard's "Watch in SUMO" buttons start this).

  python scripts/run_visual_demo.py --controller sumo_actuated|fixed_time|ippo_A|ippo_B [--seed 42] [--delay 80]
                                    [--begin 25200] [--duration 3600] [--window-pos X,Y] [--window-size W,H]

* ``sumo_actuated`` and ``fixed_time`` run the baseline controllers unchanged.
* ``ippo_A`` / ``ippo_B`` are the two final models (artifacts/models/ippo_final_seed42 / _seed1042): the shared policy is
  called for every junction every 5 s, with the observation options it was trained with (its run's config.yaml).
* Every controller gets the same simulation (07:00-08:00, seed 42) the stored results were measured on, so what you watch is
  what the dashboard numbers describe.

The GUI needs the socket ``traci`` binding (libsumo runs SUMO inside Python and cannot drive a window), which this script
selects through CITYMIND_USE_TRACI before any project module is imported.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CITYMIND_USE_TRACI"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUMO_CFG = "data/raw/intas/scenario/InTAS_train.sumocfg"
TLS_JSON = "artifacts/scenario/tls_metadata.json"
VIEW_SETTINGS = "configs/viewsettings.xml"
# The only models the demo will load (the dashboard sends a controller id, never a path).
MODELS = {
    "ippo_A": "artifacts/models/ippo_final_seed42/model_final.zip",
    "ippo_B": "artifacts/models/ippo_final_seed1042/model_final.zip",
}
CONTROLLERS = ("sumo_actuated", "fixed_time", *MODELS)
LABELS = {
    "sumo_actuated": "SUMO actuated (baseline)", "fixed_time": "Fixed time",
    "ippo_A": "Model A (final)", "ippo_B": "Model B (final)",
}


def gui_binary() -> str:
    """Absolute path of sumo-gui: the API server (and a double-clicked terminal) often has no SUMO bin directory on PATH."""
    import shutil

    from src.environment.sumo_connection import resolve_sumo_home

    found = shutil.which("sumo-gui")
    if found:
        return found
    home = resolve_sumo_home()
    candidate = Path(home) / "bin" / "sumo-gui" if home else None
    if candidate and candidate.is_file():
        return str(candidate)
    raise FileNotFoundError("sumo-gui was not found: install SUMO (with the GUI) and export SUMO_HOME, see README.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live SUMO-GUI demo of one controller")
    p.add_argument("--controller", required=True, choices=CONTROLLERS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--begin", type=int, default=25200, help="window start, seconds since midnight (default 07:00)")
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--delay", type=int, default=80, help="milliseconds per simulated second in the window (0 = as fast as possible)")
    p.add_argument("--window-pos", default=None, help="X,Y of the SUMO window (to tile several demos)")
    p.add_argument("--window-size", default=None, help="W,H of the SUMO window")
    p.add_argument("--sumo-cfg", default=SUMO_CFG)
    p.add_argument("--tls-json", default=TLS_JSON)
    return p.parse_args(argv)


def gui_extra_args(args: argparse.Namespace) -> list[str]:
    """Window options appended to the shared SUMO command (--begin/--end identical to the evaluation)."""
    extra = ["--begin", str(args.begin), "--end", str(args.begin + args.duration), "--start", "--quit-on-end"]
    if Path(VIEW_SETTINGS).is_file():
        extra += ["-g", VIEW_SETTINGS]
    if args.delay > 0:
        extra += ["--delay", str(args.delay)]
    if args.window_pos:
        extra += ["--window-pos", args.window_pos]
    if args.window_size:
        extra += ["--window-size", args.window_size]
    return extra


def run_baseline(args: argparse.Namespace) -> None:
    from src.controllers.fixed_time import FixedTimeController
    from src.controllers.sumo_actuated import SumoActuatedController

    cls = SumoActuatedController if args.controller == "sumo_actuated" else FixedTimeController
    controller = cls(episode_duration=args.begin + args.duration, sumo_binary=gui_binary(), sumo_extra_args=gui_extra_args(args))
    result = controller.run_episode(args.sumo_cfg, args.tls_json, seed=args.seed, demand_id="1.0x")
    print(f"[demo] {LABELS[args.controller]} finished: mean wait {result.mean_waiting_time:.2f} s, "
          f"{result.total_arrived} cars arrived")


def run_model(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO

    from src.agents.model_contract import run_env_settings
    from src.environment.decentralized_sumo_env import DecentralizedSumoEnv
    from src.environment.local_observation import LocalObservationConfig

    model_path = ROOT / MODELS[args.controller]
    if not model_path.is_file():
        raise FileNotFoundError(f"{MODELS[args.controller]} not found (see docs/MODELS.md)")
    model = PPO.load(str(model_path), device="cpu")
    settings = run_env_settings(model_path)
    env = DecentralizedSumoEnv(
        sumo_cfg=args.sumo_cfg, tls_metadata_json=args.tls_json, sumo_binary=gui_binary(), seed=args.seed, demand_id="1.0x",
        obs_config=LocalObservationConfig(
            queue_reach_m=settings["queue_reach_m"], timing_source=settings["timing_source"],
            pending_in_demand=settings["pending_in_demand"],
        ),
        episode_duration=args.begin + args.duration, sumo_extra_args=gui_extra_args(args),
    )
    try:
        observations, _ = env.reset()
        step = 0
        while env.agents:
            actions = {tid: int(model.predict(observations[tid], deterministic=True)[0]) for tid in env.agents}
            observations, _rewards, _terminated, _truncated, _info = env.step(actions)
            step += 1
            if step % 60 == 0:
                print(f"[demo] {LABELS[args.controller]}: {step * 5 // 60} min simulated")
    finally:
        env.close()
    print(f"[demo] {LABELS[args.controller]} finished after {step} decisions")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"[demo] {LABELS[args.controller]} - seed {args.seed}, {args.begin}s to {args.begin + args.duration}s")
    (run_baseline if args.controller in ("sumo_actuated", "fixed_time") else run_model)(args)


if __name__ == "__main__":
    main()
