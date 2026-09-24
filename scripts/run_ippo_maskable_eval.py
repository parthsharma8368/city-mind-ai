"""
run_ippo_maskable_eval.py
==========================
Evaluates a trained MaskablePPO (phase-selection) IPPO model on the same
3-seed peak-window protocol as scripts/run_evaluation.py's other
controllers, so results are directly comparable to the canonical
artifacts/metrics/ files. Not folded into EvaluationRunner itself since
that class assumes a plain PPO.predict() call; this one needs
action_masks() every step.

Usage:
  python scripts/run_ippo_maskable_eval.py \
    --sumo-cfg data/raw/intas/scenario/InTAS_train.sumocfg \
    --tls-json artifacts/scenario/tls_metadata.json \
    --model artifacts/experiments/ippo_phase_select/run_.../model_final.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a phase-selection MaskablePPO IPPO model")
    p.add_argument("--sumo-cfg", required=True)
    p.add_argument("--tls-json", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    p.add_argument("--begin-time", type=int, default=25200)
    p.add_argument("--episode-duration", type=int, default=3600)
    p.add_argument("--output-dir", default="artifacts/metrics")
    p.add_argument("--tripinfo-dir", default="artifacts/metrics/tripinfo")
    return p.parse_args()


def run_one_episode(model, sumo_cfg, tls_json, seed, begin_time, absolute_end, tripinfo_path):
    from src.environment.decentralized_sumo_env import DecentralizedSumoEnv

    env = DecentralizedSumoEnv(
        sumo_cfg=sumo_cfg,
        tls_metadata_json=tls_json,
        seed=seed,
        demand_id="1.0x",
        episode_duration=absolute_end,
        sumo_extra_args=[
            "--begin", str(begin_time), "--end", str(absolute_end),
            "--tripinfo-output", tripinfo_path,
        ],
    )

    observations, infos = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        masks = env.action_masks()
        actions = {
            tid: int(model.predict(observations[tid], action_masks=masks[tid], deterministic=True)[0])
            for tid in env.agents
        }
        observations, rewards, terminations, truncations, _ = env.step(actions)
        total_reward += sum(rewards.values())
        steps += 1
        done = not env.agents

    env.metrics.record_backlog()
    summary = env.metrics.summarize()
    env.close()
    return summary, total_reward, steps


def main():
    args = parse_args()
    from sb3_contrib import MaskablePPO
    from src.evaluation.tripinfo import parse_tripinfo
    from src.controllers.base import ControllerResult
    import json
    from dataclasses import asdict

    model = MaskablePPO.load(args.model, device="cpu")
    print(f"[Eval] Loaded phase-selection model: {args.model}")

    absolute_end = args.begin_time + args.episode_duration
    output_dir = Path(args.output_dir)
    tripinfo_dir = Path(args.tripinfo_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tripinfo_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        print(f"\n[Eval] Seed {seed} ...")
        trip_path = str(tripinfo_dir / f"ippo_phase_select_seed{seed}_1.0x.xml")
        summary, total_reward, steps = run_one_episode(
            model, args.sumo_cfg, args.tls_json, seed, args.begin_time, absolute_end, trip_path
        )
        trip_summary = parse_tripinfo(trip_path)

        result = ControllerResult(
            controller="ippo_phase_select",
            seed=seed,
            demand_id="1.0x",
            total_steps=steps,
            episode_duration=summary.episode_duration,
            mean_waiting_time=summary.mean_waiting_time,
            total_waiting_time=summary.total_waiting_time_sum,
            mean_queue=summary.mean_queue,
            max_queue=summary.max_queue,
            throughput=summary.throughput,
            total_arrived=summary.total_arrived,
            total_departed=summary.total_departed,
            mean_speed=summary.mean_speed,
            total_switches=summary.total_switches,
            cumulative_reward=total_reward,
            source_run_id=Path(args.model).parent.name,
            total_teleports=summary.total_teleports,
            backlog=summary.backlog,
            trips_completed=trip_summary.trips_completed,
            mean_trip_duration=trip_summary.mean_trip_duration,
            mean_trip_timeloss=trip_summary.mean_trip_timeloss,
        )
        results.append(result)

        fname = output_dir / f"ippo_phase_select_seed{seed}_1.0x.json"
        with open(fname, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        print(f"  -> trips={result.trips_completed} timeloss={result.mean_trip_timeloss:.2f} "
              f"backlog={result.backlog} teleports={result.total_teleports} "
              f"mean_wait={result.mean_waiting_time:.2f} switches={result.total_switches}")

    print("\n" + "=" * 70)
    print("PHASE-SELECTION IPPO — 3-SEED SUMMARY")
    print("=" * 70)
    import numpy as np
    trips = [r.trips_completed for r in results]
    timeloss = [r.mean_trip_timeloss for r in results]
    backlog = [r.backlog for r in results]
    wait = [r.mean_waiting_time for r in results]
    print(f"trips_completed: {np.mean(trips):.0f} +/- {np.std(trips):.1f}")
    print(f"mean_trip_timeloss: {np.mean(timeloss):.2f} +/- {np.std(timeloss):.2f}")
    print(f"backlog: {np.mean(backlog):.1f} +/- {np.std(backlog):.1f}")
    print(f"mean_waiting_time: {np.mean(wait):.2f} +/- {np.std(wait):.2f}")


if __name__ == "__main__":
    main()
