"""
evaluate.py
===========
Paired evaluation protocol.

For each (controller, seed) combination:
  - Use identical scenario, demand, simulation duration, seed
  - Only the controller changes
  - Saves per-step Parquet + episode summary JSON

Usage:
  python scripts/run_evaluation.py \
    --sumo-cfg data/raw/intas/ingolstadt.sumocfg \
    --tls-json artifacts/scenario/tls_metadata.json \
    --ppo-model artifacts/experiments/ppo/run_.../model_final.zip
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.controllers.base import ControllerResult
from src.controllers.fixed_time import FixedTimeController
from src.controllers.sumo_actuated import SumoActuatedController
from src.controllers.max_pressure import MaxPressureController
from src.evaluation.tripinfo import parse_tripinfo


def _extract_run_id(model_path: str) -> str:
    """model_final.zip -> its run dir name; checkpoints/xxx.zip -> the run dir two levels up."""
    p = Path(model_path).resolve()
    return p.parent.parent.name if p.parent.name == "checkpoints" else p.parent.name


class EvaluationRunner:
    """
    Runs paired evaluation for all configured controllers and seeds.
    Ensures identical conditions — only controller differs.
    """

    def __init__(
        self,
        sumo_cfg: str,
        tls_metadata_json: str,
        eval_config_path: str = "configs/evaluation.yaml",
        ppo_model_path: str | None = None,
        ippo_model_path: str | None = None,
        output_dir: str = "artifacts/metrics",
        trajectory_dir: str = "artifacts/trajectories",
    ) -> None:
        self.sumo_cfg = sumo_cfg
        self.tls_metadata_json = tls_metadata_json
        self.ppo_model_path = ppo_model_path
        self.ippo_model_path = ippo_model_path
        self.output_dir = Path(output_dir)
        self.trajectory_dir = Path(trajectory_dir)
        self.tripinfo_dir = self.output_dir / "tripinfo"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.tripinfo_dir.mkdir(parents=True, exist_ok=True)

        with open(eval_config_path) as f:
            self.cfg: dict = yaml.safe_load(f)

        self.seeds: list[int] = self.cfg.get("seeds", [42])
        self.demand_scales: list[float] = self.cfg.get("demand_scales", [1.0])
        # episode_duration is a DURATION in seconds (e.g. 3600 = 1 hour);
        # begin_time is the absolute sim-time the episode starts at (e.g.
        # 25200 = 07:00:00). self.absolute_end (begin_time + episode_duration)
        # is what every controller's own loop-termination check compares
        # against, and begin_time is passed to SUMO via --begin.
        self.episode_duration: int = int(self.cfg.get("episode_duration", 3600))
        self.begin_time: int = int(self.cfg.get("begin_time", 0))
        self.absolute_end: int = self.begin_time + self.episode_duration

    def run_all(self) -> list[ControllerResult]:
        """Run all (controller, seed, demand) combinations."""
        results: list[ControllerResult] = []

        for demand_scale in self.demand_scales:
            demand_id = f"{demand_scale}x"
            print(f"\n{'='*60}")
            print(f"Demand: {demand_id}")
            print(f"{'='*60}")

            for seed in self.seeds:
                print(f"\nSeed: {seed}")
                print("-" * 40)

                begin_args = ["--begin", str(self.begin_time)] if self.begin_time else []

                # SUMO-actuated baseline (InTAS's own programs, unmodified —
                # this is what this project previously, inaccurately, called
                # "fixed_time"; see sumo_actuated.py)
                print("[1/4] SUMO-actuated baseline ...")
                sa = SumoActuatedController(
                    episode_duration=self.absolute_end, sumo_extra_args=begin_args
                )
                sa_traj = str(self.trajectory_dir / f"sumo_actuated_seed{seed}_{demand_id}.parquet")
                sa_trip = str(self.tripinfo_dir / f"sumo_actuated_seed{seed}_{demand_id}.xml")
                sa_result = sa.run_episode(
                    self.sumo_cfg, self.tls_metadata_json, seed, demand_id,
                    trajectory_path=sa_traj, tripinfo_path=sa_trip,
                )
                results.append(sa_result)
                self._save_result(sa_result)
                print(f"  → mean_waiting={sa_result.mean_waiting_time:.2f}s, "
                      f"throughput={sa_result.throughput:.3f}")

                # Fixed-time baseline (genuinely static programs)
                print("[2/4] Fixed-time (static) baseline ...")
                ft = FixedTimeController(
                    episode_duration=self.absolute_end, sumo_extra_args=begin_args
                )
                ft_traj = str(self.trajectory_dir / f"fixed_time_seed{seed}_{demand_id}.parquet")
                ft_trip = str(self.tripinfo_dir / f"fixed_time_seed{seed}_{demand_id}.xml")
                ft_result = ft.run_episode(
                    self.sumo_cfg, self.tls_metadata_json, seed, demand_id,
                    trajectory_path=ft_traj, tripinfo_path=ft_trip,
                )
                results.append(ft_result)
                self._save_result(ft_result)
                print(f"  → mean_waiting={ft_result.mean_waiting_time:.2f}s, "
                      f"throughput={ft_result.throughput:.3f}")

                # Max-Pressure baseline
                print("[3/4] Max-Pressure baseline ...")
                mp = MaxPressureController(
                    episode_duration=self.absolute_end, sumo_extra_args=begin_args
                )
                mp_traj = str(self.trajectory_dir / f"max_pressure_seed{seed}_{demand_id}.parquet")
                mp_trip = str(self.tripinfo_dir / f"max_pressure_seed{seed}_{demand_id}.xml")
                mp_result = mp.run_episode(
                    self.sumo_cfg, self.tls_metadata_json, seed, demand_id,
                    trajectory_path=mp_traj, tripinfo_path=mp_trip,
                )
                results.append(mp_result)
                self._save_result(mp_result)
                print(f"  → mean_waiting={mp_result.mean_waiting_time:.2f}s, "
                      f"throughput={mp_result.throughput:.3f}")

                # PPO (if model provided)
                if self.ppo_model_path:
                    print("[4/5] PPO evaluation ...")
                    ppo_result = self._run_ppo_episode(seed, demand_id)
                    results.append(ppo_result)
                    self._save_result(ppo_result)
                    print(f"  → mean_waiting={ppo_result.mean_waiting_time:.2f}s, "
                          f"throughput={ppo_result.throughput:.3f}")

                # IPPO (if model provided)
                if self.ippo_model_path:
                    print("[5/5] IPPO evaluation ...")
                    ippo_result = self._run_ippo_episode(seed, demand_id)
                    results.append(ippo_result)
                    self._save_result(ippo_result)
                    print(f"  → mean_waiting={ippo_result.mean_waiting_time:.2f}s, "
                          f"throughput={ippo_result.throughput:.3f}")

        # Save combined results
        self._save_combined(results)
        return results

    def _run_ppo_episode(self, seed: int, demand_id: str) -> ControllerResult:
        """Run one PPO evaluation episode."""
        from src.agents.ppo_agent import PPOAgent, make_env
        from src.environment.centralized_sumo_env import CentralizedSumoEnv
        import numpy as np

        agent = PPOAgent(
            sumo_cfg=self.sumo_cfg,
            tls_metadata_json=self.tls_metadata_json,
            seed=seed,
            demand_id=demand_id,
        )
        agent.load(self.ppo_model_path)

        ppo_trip = str(self.tripinfo_dir / f"ppo_seed{seed}_{demand_id}.xml")
        env = make_env(
            sumo_cfg=self.sumo_cfg,
            tls_metadata_json=self.tls_metadata_json,
            seed=seed,
            demand_id=demand_id,
            episode_duration=self.absolute_end,
            begin_time=self.begin_time,
            record_trajectories=True,
            tripinfo_path=ppo_trip,
        )

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        # Must run before env.close() tears down the TraCI connection —
        # record_backlog() reads live SUMO state.
        env.unwrapped.metrics.record_backlog()
        summary = env.unwrapped.metrics.summarize()

        # Save trajectory
        traj_path = str(
            self.trajectory_dir / f"ppo_seed{seed}_{demand_id}.parquet"
        )
        env.unwrapped.save_trajectory(traj_path)
        env.close()

        trip_summary = parse_tripinfo(ppo_trip)
        return ControllerResult(
            controller="ppo",
            seed=seed,
            demand_id=demand_id,
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
            source_run_id=_extract_run_id(self.ppo_model_path),
            total_teleports=summary.total_teleports,
            backlog=summary.backlog,
            trips_completed=trip_summary.trips_completed,
            mean_trip_duration=trip_summary.mean_trip_duration,
            mean_trip_timeloss=trip_summary.mean_trip_timeloss,
        )

    def _run_ippo_episode(self, seed: int, demand_id: str) -> ControllerResult:
        """Run one parameter-shared IPPO evaluation episode (plain loop, no SuperSuit needed for inference)."""
        from stable_baselines3 import PPO
        from src.environment.decentralized_sumo_env import DecentralizedSumoEnv
        from src.environment.sumo_connection import traci

        model = PPO.load(self.ippo_model_path, device="cpu")

        ippo_trip = str(self.tripinfo_dir / f"ippo_seed{seed}_{demand_id}.xml")
        env = DecentralizedSumoEnv(
            sumo_cfg=self.sumo_cfg,
            tls_metadata_json=self.tls_metadata_json,
            seed=seed,
            demand_id=demand_id,
            episode_duration=self.absolute_end,
            sumo_extra_args=[
                "--begin", str(self.begin_time), "--end", str(self.absolute_end),
                "--tripinfo-output", ippo_trip,
            ],
        )

        observations, _ = env.reset()
        trajectory_rows: list[dict] = []
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            actions = {
                tid: int(model.predict(observations[tid], deterministic=True)[0])
                for tid in env.agents
            }
            observations, rewards, terminations, truncations, _ = env.step(actions)
            total_reward += sum(rewards.values())
            steps += 1
            done = not env.agents  # DecentralizedSumoEnv empties .agents on the terminal step

            sim_time = traci.simulation.getTime()
            for vid in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vid)
                spd = traci.vehicle.getSpeed(vid)
                trajectory_rows.append({
                    "time": sim_time, "vehicle_id": vid, "x": x, "y": y, "speed": spd,
                    "tls_id": None, "phase": None, "tls_state": None,
                })
            for tid, state in env._runtime_states.items():
                spec = env.specs_by_id[tid]
                phase_idx = state.current_phase
                state_str = spec.phases[phase_idx].state if phase_idx < len(spec.phases) else ""
                trajectory_rows.append({
                    "time": sim_time, "vehicle_id": None, "x": None, "y": None, "speed": None,
                    "tls_id": tid, "phase": phase_idx, "tls_state": state_str,
                })

        env.metrics.record_backlog()
        summary = env.metrics.summarize()
        env.close()

        if trajectory_rows:
            traj_path = self.trajectory_dir / f"ippo_seed{seed}_{demand_id}.parquet"
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(trajectory_rows).to_parquet(traj_path, index=False)

        trip_summary = parse_tripinfo(ippo_trip)
        return ControllerResult(
            controller="ippo",
            seed=seed,
            demand_id=demand_id,
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
            source_run_id=_extract_run_id(self.ippo_model_path),
            total_teleports=summary.total_teleports,
            backlog=summary.backlog,
            trips_completed=trip_summary.trips_completed,
            mean_trip_duration=trip_summary.mean_trip_duration,
            mean_trip_timeloss=trip_summary.mean_trip_timeloss,
        )

    def _save_result(self, result: ControllerResult) -> None:
        """Save one result as JSON."""
        fname = f"{result.controller}_seed{result.seed}_{result.demand_id}.json"
        path = self.output_dir / fname
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

    def _save_combined(self, results: list[ControllerResult]) -> None:
        """Save all results as a single Parquet file."""
        rows = [asdict(r) for r in results]
        df = pd.DataFrame(rows)
        path = self.output_dir / "evaluation_results.parquet"
        try:
            df.to_parquet(path, index=False, engine="pyarrow")
        except Exception:
            try:
                df.to_parquet(path, index=False)
            except Exception:
                df.to_csv(str(path).replace(".parquet", ".csv"), index=False)
        print(f"\nAll results saved: {path}")
        print(df[["controller", "seed", "demand_id",
                   "mean_waiting_time", "mean_queue",
                   "throughput", "total_arrived", "backlog", "total_teleports",
                   "trips_completed", "mean_trip_timeloss"]].to_string())
