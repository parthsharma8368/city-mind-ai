"""
local_reward.py
===============
Per-intersection reward for parameter-shared IPPO: a LEVEL of delay, summed over decision steps.

  r_i(t) = - scale * clip( w_q * Q_i + w_d * D_i + w_s * S_i + w_b * P_i + w_g * P / N ,  c )

  Q_i  cars halted on this signal's own incoming lanes (+ the unsignalised feeder roads behind them)   [stopped cars]
  D_i  mean cars halted per lane this signal feeds into                                              [spillback]
  S_i  1 if the signal changed phase during the step, else 0                                          [lost clearance time]
  P_i  cars waiting OUTSIDE the network whose route's first signal is this one                         [entry backlog]
  P    all cars waiting outside the network, N = number of signals (shared by every agent)
  c    clip bound (applied before the scale)

Every term counts delayed cars, so the return over an episode is (minus) total vehicle-delay - the quantity the evaluation
scores. Weights w_b + w_g = 1 make a car held outside the network cost as much as a stopped one: blocking entry can't pay.
Unlike the change-of-waiting-time reward (a see-saw that telescopes to final-minus-initial waiting and rewards every switch),
a level cannot be gamed by timing a queue release.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.environment.sumo_connection import traci


@dataclass
class LocalRewardConfig:
    queue_level_weight: float = 1.0          # w_q
    downstream_level_weight: float = 2.0     # w_d
    switching_penalty: float = 1.0           # w_s
    backlog_weight: float = 0.5              # w_b
    global_backlog_weight: float = 0.5       # w_g
    reward_scale: float = 0.1                # scale: keeps returns O(10-100) so the value loss stays small
    reward_clip: float | None = 200.0        # c


class LocalRewardCalculator:
    def __init__(self, config: LocalRewardConfig | None = None) -> None:
        self.cfg = config or LocalRewardConfig()
        self.reset()

    def reset(self) -> None:
        self._pending_by_tls: dict[str, int] = {}
        self._pending_by_edge: dict[str, int] = {}
        self._pending_total = 0
        self._pending_cache: dict[str, tuple[str | None, str | None]] = {}   # vehicle -> (tls, first signalised edge)

    # ── cars waiting outside the network ────────────────────────────────────
    def pending_for(self, tls_id: str) -> int:
        return self._pending_by_tls.get(tls_id, 0)

    def pending_by_edge(self) -> dict[str, int]:
        """Waiting-outside cars keyed by the first signalised edge on their route (where they will queue once inserted)."""
        return self._pending_by_edge

    def refresh_pending(self, edge_to_tls: dict[str, str]) -> None:
        """One cheap TraCI call per decision step; a vehicle's route is looked up once."""
        by_tls: dict[str, int] = {}
        by_edge: dict[str, int] = {}
        try:
            pending = list(traci.simulation.getPendingVehicles())
        except traci.exceptions.TraCIException:
            pending = []
        for vid in pending:
            if vid not in self._pending_cache:
                tls = first_edge = None
                try:
                    for edge in traci.vehicle.getRoute(vid):
                        if edge in edge_to_tls:
                            tls, first_edge = edge_to_tls[edge], edge
                            break
                except traci.exceptions.TraCIException:
                    pass
                self._pending_cache[vid] = (tls, first_edge)
            tls, first_edge = self._pending_cache[vid]
            if tls is not None:
                by_tls[tls] = by_tls.get(tls, 0) + 1
                by_edge[first_edge] = by_edge.get(first_edge, 0) + 1
        alive = set(pending)
        self._pending_cache = {v: x for v, x in self._pending_cache.items() if v in alive}
        self._pending_by_tls, self._pending_by_edge, self._pending_total = by_tls, by_edge, len(pending)

    # ── reward ──────────────────────────────────────────────────────────────
    def compute(
        self, tls_id: str, switched: bool, queue_level: float, downstream_level: float, n_agents: int = 1
    ) -> tuple[float, dict]:
        c = self.cfg
        terms = {
            "queue": c.queue_level_weight * queue_level,
            "downstream": c.downstream_level_weight * downstream_level,
            "switch": c.switching_penalty * (1.0 if switched else 0.0),
            "backlog": c.backlog_weight * self.pending_for(tls_id),
            "global_backlog": c.global_backlog_weight * (self._pending_total / max(1, n_agents)),
        }
        cost = sum(terms.values())
        if c.reward_clip is not None:
            cost = min(cost, abs(c.reward_clip))
        info = {f"reward/{k}_component": -v for k, v in terms.items()}
        info["metrics/pending_local"] = self.pending_for(tls_id)
        return float(-c.reward_scale * cost), info
