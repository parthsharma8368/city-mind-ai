"""
evaluate.py
===========
Paired evaluation protocol.

For each (controller, seed) combination:
  - Use identical scenario, demand, simulation duration, seed
  - Only the controller changes
  - Saves per-step Parquet + episode summary JSON (+ drain-aware metrics and
    full provenance, see drain_metrics.py / provenance.py)

Result files and the overwrite guard
------------------------------------
* The canonical 07:00-08:00 peak window keeps the legacy names
  ``{controller}_seed{seed}_{demand}.json`` (the API serves these).
* Any other window writes ``..._b{begin}_d{duration}`` files into
  ``<metrics>/runs/b{begin}_d{duration}/`` so it can never replace a canonical
  artifact.
* Every JSON stores ``provenance.config_hash`` (sumocfg content, window, step
  length, demand, SUMO version, protocol). Writing over an existing result
  whose hash differs - or that has no hash (legacy) - is refused unless
  ``overwrite=True`` (CLI ``--overwrite``). The check runs before any
  simulation starts.
* JSON and parquet are written atomically (temp file + os.replace) and the
  combined parquet is rebuilt from ALL per-run JSONs in the directory, in a
  ``finally`` block, so a crash mid-run never loses finished controllers.

Usage:
  python scripts/run_evaluation.py \
    --sumo-cfg data/raw/intas/scenario/InTAS_train.sumocfg \
    --tls-json artifacts/scenario/tls_metadata.json \
    [--ppo-model ...] [--ippo-model ...] [--phase-select-model ...]
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.controllers.base import ControllerResult, result_from_summary
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController
from src.controllers.sumo_actuated import SumoActuatedController
from src.evaluation import drain_metrics
from src.evaluation.provenance import (
    atomic_write_text,
    build_provenance,
    config_hash,
    is_canonical_window,
    read_sumocfg,
)
from src.evaluation.tripinfo import parse_tripinfo

DEFAULT_METRICS_DIR = "artifacts/metrics"
DEFAULT_TRAJECTORY_DIR = "artifacts/trajectories"

BASELINES = ("sumo_actuated", "fixed_time", "max_pressure")
RL_CONTROLLERS = ("ppo", "ippo", "ippo_phase_select")
ALL_CONTROLLERS = BASELINES + RL_CONTROLLERS

_BASELINE_CLASSES = {
    "sumo_actuated": SumoActuatedController,
    "fixed_time": FixedTimeController,
    "max_pressure": MaxPressureController,
}


class ResultExistsError(FileExistsError):
    """A result for a DIFFERENT (or unknown, legacy) configuration already exists."""


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
        ippo_phase_select_model_path: str | None = None,
        output_dir: str | None = None,
        trajectory_dir: str | None = None,
        controllers: list[str] | None = None,
        overwrite: bool = False,
        compute_drain: bool = True,
    ) -> None:
        self.sumo_cfg = sumo_cfg
        self.tls_metadata_json = tls_metadata_json
        self.ppo_model_path = ppo_model_path
        self.ippo_model_path = ippo_model_path
        self.ippo_phase_select_model_path = ippo_phase_select_model_path
        self.overwrite = overwrite
        self.compute_drain = compute_drain

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

        self.canonical = is_canonical_window(self.begin_time, self.episode_duration)
        self.window_tag = f"b{self.begin_time}_d{self.episode_duration}"
        # canonical window: legacy names; anything else: window-tagged names in runs/<tag>/
        self.name_suffix = "" if self.canonical else f"_{self.window_tag}"

        self.output_dir = Path(self._resolve_dir(output_dir, "output_dir", DEFAULT_METRICS_DIR))
        self.trajectory_dir = Path(self._resolve_dir(trajectory_dir, "trajectory_dir", DEFAULT_TRAJECTORY_DIR))
        self.tripinfo_dir = self.output_dir / "tripinfo"
        self.stats_dir = self.output_dir / "sumo_stats"
        self.cache_dir = Path(DEFAULT_METRICS_DIR) / "_drain_cache"
        for d in (self.output_dir, self.trajectory_dir, self.tripinfo_dir, self.stats_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.controllers = self._select_controllers(controllers)
        self._sumocfg_info = read_sumocfg(sumo_cfg)
        self._demand: drain_metrics.ScheduledDemand | None = None

    # ── configuration ───────────────────────────────────────────────────────
    def _resolve_dir(self, cli_value: str | None, cfg_key: str, default: str) -> str:
        """CLI value > config value > default; a non-canonical window never lands in the default dir."""
        if cli_value:
            return cli_value
        cfg_value = self.cfg.get(cfg_key)
        if cfg_value and cfg_value != default:
            return cfg_value
        return default if self.canonical else str(Path(default) / "runs" / self.window_tag)

    def _select_controllers(self, requested: list[str] | None) -> list[str]:
        models = {
            "ppo": self.ppo_model_path,
            "ippo": self.ippo_model_path,
            "ippo_phase_select": self.ippo_phase_select_model_path,
        }
        if requested is None:
            return list(BASELINES) + [c for c in RL_CONTROLLERS if models[c]]
        unknown = [c for c in requested if c not in ALL_CONTROLLERS]
        if unknown:
            raise ValueError(f"unknown controller(s) {unknown}; choose from {ALL_CONTROLLERS}")
        missing = [c for c in requested if c in models and not models[c]]
        if missing:
            raise ValueError(f"no model path given for {missing}")
        return [c for c in ALL_CONTROLLERS if c in requested]

    def _cfg_hash(self, demand_id: str) -> str:
        return config_hash(
            self.sumo_cfg, self.begin_time, self.episode_duration, demand_id,
            self._sumocfg_info["step_length"], drain_metrics.PROTOCOL_ID,
        )

    def _stem(self, controller: str, seed: int, demand_id: str) -> str:
        return f"{controller}_seed{seed}_{demand_id}{self.name_suffix}"

    def _model_path(self, controller: str) -> str | None:
        return {"ppo": self.ppo_model_path, "ippo": self.ippo_model_path,
                "ippo_phase_select": self.ippo_phase_select_model_path}.get(controller)

    # ── overwrite guard ─────────────────────────────────────────────────────
    def _guard(self, controller: str, seed: int, demand_id: str) -> None:
        path = self.output_dir / f"{self._stem(controller, seed, demand_id)}.json"
        if not path.exists() or self.overwrite:
            return
        try:
            existing = json.loads(path.read_text()).get("provenance", {}).get("config_hash")
        except (OSError, json.JSONDecodeError):
            existing = None
        new = self._cfg_hash(demand_id)
        if existing != new:
            raise ResultExistsError(
                f"{path} already exists with config_hash={existing or 'NONE (legacy result, config unknown)'}; "
                f"this run has config_hash={new}. Refusing to overwrite a result of a different/unknown "
                "configuration. Pass overwrite=True / --overwrite to replace it deliberately, or choose "
                "another --output-dir."
            )

    # ── drain-aware scoring ─────────────────────────────────────────────────
    def _scheduled_demand(self) -> drain_metrics.ScheduledDemand:
        if self._demand is None:
            info = self._sumocfg_info
            self._demand = drain_metrics.build_scheduled_demand(
                info["route_files"], info["net_file"], self.begin_time, self.absolute_end,
                cache_dir=self.cache_dir,
            )
        return self._demand

    def _sumo_extra_args(self, stem: str) -> list[str]:
        """Window + SUMO's own statistic/summary outputs (reliable teleport / discard counters)."""
        return [
            "--begin", str(self.begin_time),
            "--statistic-output", str(self.stats_dir / f"{stem}.statistic.xml"),
            "--summary-output", str(self.stats_dir / f"{stem}.summary.xml"),
        ]

    # ── running ─────────────────────────────────────────────────────────────
    def run_all(self) -> list[ControllerResult]:
        """Run all (controller, seed, demand) combinations."""
        # Fail fast, before any 3-minute simulation, if a different config's result is in the way.
        for demand_scale in self.demand_scales:
            for seed in self.seeds:
                for ctrl in self.controllers:
                    self._guard(ctrl, seed, f"{demand_scale}x")

        results: list[ControllerResult] = []
        failures: list[str] = []
        n = len(self.controllers)
        try:
            for demand_scale in self.demand_scales:
                demand_id = f"{demand_scale}x"
                print(f"\n{'='*60}\nDemand: {demand_id}\n{'='*60}")
                for seed in self.seeds:
                    print(f"\nSeed: {seed}\n{'-'*40}")
                    for i, ctrl in enumerate(self.controllers, 1):
                        print(f"[{i}/{n}] {ctrl} ...")
                        try:
                            result = self._run_one(ctrl, seed, demand_id)
                        except Exception as exc:  # keep finished controllers; report at the end
                            failures.append(f"{ctrl} seed {seed} {demand_id}: {type(exc).__name__}: {exc}")
                            print(f"  !! FAILED: {failures[-1]}")
                            continue
                        results.append(result)
                        print(f"  → trips={result.trips_completed}, backlog={result.backlog}, "
                              f"mean_waiting={result.mean_waiting_time:.2f}s, throughput={result.throughput:.3f}")
        finally:
            self._save_combined()
        if failures:
            raise RuntimeError("evaluation finished with failures:\n  " + "\n  ".join(failures))
        return results

    def _run_one(self, ctrl: str, seed: int, demand_id: str) -> ControllerResult:
        stem = self._stem(ctrl, seed, demand_id)
        tripinfo = str(self.tripinfo_dir / f"{stem}.xml")
        traj = str(self.trajectory_dir / f"{stem}.parquet")
        if ctrl in _BASELINE_CLASSES:
            runner = _BASELINE_CLASSES[ctrl](
                episode_duration=self.absolute_end, sumo_extra_args=self._sumo_extra_args(stem)
            )
            result = runner.run_episode(
                self.sumo_cfg, self.tls_metadata_json, seed, demand_id,
                trajectory_path=traj, tripinfo_path=tripinfo,
            )
        elif ctrl == "ppo":
            result = self._run_ppo_episode(seed, demand_id, stem, tripinfo, traj)
        else:
            result = self._run_ippo_episode(ctrl, seed, demand_id, stem, tripinfo, traj)
        self._save_result(result, stem, tripinfo, traj, seed, demand_id)
        return result

    def _run_ppo_episode(self, seed: int, demand_id: str, stem: str, tripinfo: str, traj: str) -> ControllerResult:
        """Run one centralized-PPO evaluation episode."""
        from src.agents.ppo_agent import PPOAgent, make_env

        agent = PPOAgent(
            sumo_cfg=self.sumo_cfg,
            tls_metadata_json=self.tls_metadata_json,
            seed=seed,
            demand_id=demand_id,
        )
        agent.load(self.ppo_model_path)

        env = make_env(
            sumo_cfg=self.sumo_cfg,
            tls_metadata_json=self.tls_metadata_json,
            seed=seed,
            demand_id=demand_id,
            episode_duration=self.absolute_end,
            begin_time=self.begin_time,
            record_trajectories=True,
            tripinfo_path=tripinfo,
        )
        # make_env cannot take extra SUMO args; the env starts SUMO lazily at reset().
        extra = getattr(env.unwrapped, "sumo_extra_args", None)
        if isinstance(extra, list):
            extra += self._sumo_extra_args(stem)[2:]  # statistic/summary outputs (--begin is already set)

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
        env.unwrapped.save_trajectory(traj)
        env.close()

        return self._rl_result("ppo", summary, seed, demand_id, steps, total_reward, tripinfo)

    def _run_ippo_episode(
        self, controller: str, seed: int, demand_id: str, stem: str, tripinfo: str, traj: str
    ) -> ControllerResult:
        """One parameter-shared IPPO episode (plain loop, no SuperSuit needed for inference).

        controller == "ippo_phase_select" loads a MaskablePPO and passes the
        per-agent action masks to predict(); "ippo" is the plain binary-action PPO.
        """
        from src.environment.decentralized_sumo_env import DecentralizedSumoEnv
        from src.environment.sumo_connection import traci

        maskable = controller == "ippo_phase_select"
        model_path = self._model_path(controller)
        if maskable:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(model_path, device="cpu")
        else:
            from stable_baselines3 import PPO
            model = PPO.load(model_path, device="cpu")

        env = DecentralizedSumoEnv(
            sumo_cfg=self.sumo_cfg,
            tls_metadata_json=self.tls_metadata_json,
            seed=seed,
            demand_id=demand_id,
            episode_duration=self.absolute_end,
            sumo_extra_args=[
                *self._sumo_extra_args(stem), "--end", str(self.absolute_end), "--tripinfo-output", tripinfo,
            ],
        )

        observations, _ = env.reset()
        trajectory_rows: list[dict] = []
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            if maskable:
                masks = env.action_masks()
                actions = {
                    tid: int(model.predict(observations[tid], action_masks=masks[tid], deterministic=True)[0])
                    for tid in env.agents
                }
            else:
                actions = {
                    tid: int(model.predict(observations[tid], deterministic=True)[0]) for tid in env.agents
                }
            observations, rewards, terminations, truncations, _ = env.step(actions)
            total_reward += sum(rewards.values())
            steps += 1
            done = not env.agents  # DecentralizedSumoEnv empties .agents on the terminal step

            sim_time = traci.simulation.getTime()
            for vid in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vid)
                trajectory_rows.append({
                    "time": sim_time, "vehicle_id": vid, "x": x, "y": y,
                    "speed": traci.vehicle.getSpeed(vid),
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
            Path(traj).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(trajectory_rows).to_parquet(traj, index=False)

        return self._rl_result(controller, summary, seed, demand_id, steps, total_reward, tripinfo)

    def _rl_result(
        self, controller: str, summary: Any, seed: int, demand_id: str, steps: int, total_reward: float,
        tripinfo: str,
    ) -> ControllerResult:
        """RL episode -> ControllerResult through the shared core mapping (strict tripinfo parse)."""
        result = result_from_summary(
            controller, summary, seed=seed, demand_id=demand_id,
            total_switches=summary.total_switches, trip_summary=parse_tripinfo(tripinfo),
        )
        return replace(
            result, total_steps=steps, cumulative_reward=total_reward,
            source_run_id=_extract_run_id(self._model_path(controller) or ""),
        )

    # ── persistence ─────────────────────────────────────────────────────────
    def build_payload(
        self, result: ControllerResult, stem: str, tripinfo: str, traj: str, seed: int, demand_id: str
    ) -> dict[str, Any]:
        """Full JSON payload: legacy fields + tripinfo extras + drain metrics + provenance."""
        payload: dict[str, Any] = asdict(result)
        trip = parse_tripinfo(tripinfo)
        payload.update({
            "mean_trip_waiting_time": trip.mean_trip_waiting_time,
            "zero_length_trips": trip.zero_length_trips,
            "trips_completed_nonzero": trip.trips_completed_nonzero,
            "mean_trip_duration_nonzero": trip.mean_trip_duration_nonzero,
            "mean_trip_timeloss_nonzero": trip.mean_trip_timeloss_nonzero,
            "mean_trip_timeloss_plus_departdelay_nonzero": trip.mean_trip_timeloss_plus_departdelay_nonzero,
        })
        if self.compute_drain:
            score = drain_metrics.score_tripinfo_file(
                self._scheduled_demand(), tripinfo,
                statistic_path=self.stats_dir / f"{stem}.statistic.xml",
                summary_path=self.stats_dir / f"{stem}.summary.xml",
                trajectory_path=traj,
                inserted=result.total_departed, json_teleports=result.total_teleports,
                backlog=result.backlog,
            )
            payload.update(drain_metrics.drain_json_fields(score))
        payload["provenance"] = build_provenance(
            sumo_cfg=self.sumo_cfg, seed=seed, begin_time=self.begin_time,
            episode_duration=self.episode_duration, demand_id=demand_id,
            model_path=self._model_path(result.controller), protocol=drain_metrics.PROTOCOL_ID,
            extra={"tripinfo_path": tripinfo, "trajectory_path": traj, "eval_window_tag": self.window_tag},
        )
        return payload

    def _save_result(
        self, result: ControllerResult, stem: str, tripinfo: str, traj: str, seed: int, demand_id: str
    ) -> None:
        """Save one result as JSON (atomic)."""
        payload = self.build_payload(result, stem, tripinfo, traj, seed, demand_id)
        atomic_write_text(self.output_dir / f"{stem}.json", json.dumps(payload, indent=2, default=str))

    def _save_combined(self) -> Path | None:
        """Rebuild evaluation_results.parquet from EVERY per-run JSON in the output dir (atomic).

        Includes results of earlier invocations, so a partial run never shrinks the table.
        """
        rows = []
        for f in sorted(self.output_dir.glob("*_seed*_*.json")):
            if f.name.startswith("drain_scores"):
                continue
            data = json.loads(f.read_text())
            if isinstance(data, dict) and "controller" in data and "seed" in data:
                rows.append({k: v for k, v in data.items() if not isinstance(v, (dict, list))})
        if not rows:
            print("\n[evaluate] no result JSONs found; combined parquet not written")
            return None
        df = pd.DataFrame(rows)
        path = self.output_dir / "evaluation_results.parquet"
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
        df.to_parquet(tmp, index=False, engine="pyarrow")
        os.replace(tmp, path)
        print(f"\nAll results saved: {path} ({len(df)} runs)")
        cols = [c for c in ("controller", "seed", "demand_id", "mean_waiting_time", "throughput", "backlog",
                            "trips_completed", "mean_trip_timeloss", "lag_per_scheduled_car_s",
                            "due_completion_ratio") if c in df.columns]
        print(df[cols].to_string())
        return path
