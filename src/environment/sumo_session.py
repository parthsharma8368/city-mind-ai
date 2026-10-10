"""
sumo_session.py
===============
Lifecycle code shared by CentralizedSumoEnv and DecentralizedSumoEnv, so the two
environments (and, through ``build_sumo_command`` / ``SafetyLayer.take_over``,
the baselines) start SUMO the same way:

  * ONE command builder (sumo_args.build_sumo_command) -- the exact argv is kept
    in ``env.last_sumo_command`` and written to the metrics summary.
  * seed handling: ``seed_mode='fixed'`` (default, what evaluation uses) re-uses
    the constructor seed every episode; ``'per_episode'`` (training) uses
    ``base_seed + episode_index`` so a policy is not trained on one single
    traffic realisation (rl-core-005).
  * optional warm start: SUMO is started ``warm_start_seconds`` before the
    episode's ``--begin`` and run under its NATIVE signals, so the policy's
    first decision happens in a loaded network instead of an empty one.
  * the native signal program is frozen at t0 (rl-core-003).
  * episode end flags: time limit => truncated only (rl-core-006).
"""
from __future__ import annotations

import logging
import time

from src.environment.sumo_args import build_sumo_command, get_option, with_option
from src.environment.sumo_connection import USING_LIBSUMO, traci

log = logging.getLogger(__name__)

SEED_MODES = ("fixed", "per_episode")


def episode_end_flags(sim_time: float, episode_duration: float, min_expected: int | None) -> tuple[bool, bool]:
    """(terminated, truncated). Reaching the time limit is a TRUNCATION: the
    value of the last state is not zero, so GAE must bootstrap from it. A natural
    end (nothing left to simulate) before the limit is a true termination."""
    truncated = sim_time >= episode_duration
    terminated = (not truncated) and (min_expected == 0)
    return terminated, truncated


def mark_time_limit(info: dict, terminated: bool, truncated: bool) -> dict:
    """Set the SB3 convention key. SB3's on-policy loop bootstraps the value of
    ``infos[i]["terminal_observation"]`` only when this key is True; SuperSuit's
    SB3 wrapper ORs terminated|truncated into ``done`` and forwards infos."""
    info["TimeLimit.truncated"] = bool(truncated and not terminated)
    return info


class SumoSessionMixin:
    """Expects: sumo_binary, sumo_cfg, step_length, sumo_extra_args, _seed,
    episode_duration, safety, metrics, _sim_running, and the options set by
    ``_init_session`` below."""

    def _init_session(
        self,
        seed_mode: str = "fixed",
        warm_start_seconds: float = 0.0,
        freeze_native_program: bool = True,
        route_steps: int | None = None,
        teleport_time: float | None = None,
        label_prefix: str = "citymind",
    ) -> None:
        if seed_mode not in SEED_MODES:
            raise ValueError(f"seed_mode must be one of {SEED_MODES}, got {seed_mode!r}")
        self.seed_mode = seed_mode
        self.warm_start_seconds = float(warm_start_seconds)
        self.freeze_native_program = freeze_native_program
        self.route_steps = route_steps
        self.teleport_time = teleport_time
        self._label_prefix = label_prefix
        self._base_seed = self._seed
        self._episode_index = 0
        self.last_sumo_command: list[str] = []

    # -- seeds ---------------------------------------------------------------
    @staticmethod
    def _worker_seed_offset() -> int:
        """SuperSuit clones the env into N worker processes with the SAME base
        seed; without an offset every parallel worker would replay identical
        traffic. Worker k (spawn order, 1-based) gets +100000*k; the main
        process gets 0."""
        import multiprocessing

        ident = getattr(multiprocessing.current_process(), "_identity", ())
        return 100_000 * ident[0] if ident else 0

    def _resolve_episode_seed(self, seed: int | None) -> int:
        """Seed to use for the episode about to start."""
        if seed is not None:
            self._base_seed = int(seed)
            self._episode_index = 0
            self._seed = int(seed)
        elif self.seed_mode == "per_episode":
            self._seed = self._base_seed + self._episode_index + self._worker_seed_offset()
        self._episode_index += 1
        return self._seed

    # -- SUMO lifecycle ------------------------------------------------------
    def build_command(self) -> list[str]:
        cmd = build_sumo_command(
            self.sumo_binary, self.sumo_cfg, step_length=self.step_length, seed=self._seed,
            route_steps=self.route_steps, teleport_time=self.teleport_time,
            extra_args=self.sumo_extra_args,
        )
        if self.warm_start_seconds > 0:
            begin = get_option(cmd, "--begin")
            begin_f = float(begin) if begin is not None else 0.0
            cmd = with_option(cmd, "--begin", str(max(0, int(begin_f - self.warm_start_seconds))))
        return cmd

    def _start_sumo(self) -> None:
        """Launch SUMO, run the (optional) native warm start and freeze the
        native signal program. On ANY failure SUMO is closed again."""
        cmd = self.build_command()
        self.last_sumo_command = cmd
        if USING_LIBSUMO:
            traci.start(cmd)
        else:
            traci.start(cmd, label=f"{self._label_prefix}_{self._seed}_{int(time.time() * 1000)}")
        self._sim_running = True
        try:
            if self.warm_start_seconds > 0:
                for _ in range(round(self.warm_start_seconds / self.step_length)):
                    traci.simulationStep()
            if self.freeze_native_program:
                self.safety.take_over()
        except BaseException:
            self._close_sumo()
            raise

    def _close_sumo(self) -> None:
        """Close the connection. Never raises; always clears _sim_running."""
        try:
            self.metrics.record_backlog()
        except Exception as exc:  # noqa: BLE001 - close path must not mask the real error
            log.debug("record_backlog failed during close: %r", exc)
        try:
            traci.close()
        except Exception as exc:  # noqa: BLE001
            log.debug("traci.close failed: %r", exc)
        self._sim_running = False

    def close(self) -> None:
        if self._sim_running:
            self._close_sumo()

    # -- episode end ---------------------------------------------------------
    def _episode_end_flags(self, sim_time: float) -> tuple[bool, bool]:
        if sim_time >= self.episode_duration:
            return episode_end_flags(sim_time, self.episode_duration, None)
        try:
            min_expected = traci.simulation.getMinExpectedNumber()
        except traci.exceptions.TraCIException as exc:
            # A FatalTraCIError (dead SUMO) is NOT caught and propagates.
            log.warning("getMinExpectedNumber failed (%r); ending the episode as terminated", exc)
            return True, False
        return episode_end_flags(sim_time, self.episode_duration, min_expected)

    def _check_terminated(self, sim_time: float) -> bool:
        """Natural-end flag only (kept for callers of the old name; the time
        limit is now reported by ``truncated``, never by ``terminated``)."""
        return self._episode_end_flags(sim_time)[0]
