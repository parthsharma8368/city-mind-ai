"""The level-based delay reward and the count of cars waiting outside the network."""
from __future__ import annotations

import pytest

from src.environment import local_reward as lr
from src.environment.local_reward import LocalRewardCalculator, LocalRewardConfig


def _calc(**kw):
    base = {"queue_level_weight": 0.0, "downstream_level_weight": 0.0, "switching_penalty": 0.0, "backlog_weight": 0.0,
            "global_backlog_weight": 0.0, "reward_scale": 1.0, "reward_clip": None}
    return LocalRewardCalculator(LocalRewardConfig(**{**base, **kw}))


def test_reward_is_minus_the_weighted_cars_delayed_and_has_no_see_saw():
    """Same queue, same reward: no 'release bonus' from a change-of-waiting-time term."""
    calc = _calc(queue_level_weight=1.0, downstream_level_weight=2.0, switching_penalty=1.0)
    r1, _ = calc.compute("T", False, queue_level=10.0, downstream_level=1.5)
    r2, _ = calc.compute("T", False, queue_level=10.0, downstream_level=1.5)
    assert r1 == r2 == pytest.approx(-(10.0 + 2.0 * 1.5))
    rs, info = calc.compute("T", True, queue_level=10.0, downstream_level=1.5)
    assert rs == pytest.approx(-(10.0 + 3.0 + 1.0)) and info["reward/switch_component"] == -1.0


def test_clip_bounds_the_cost_and_the_scale_is_applied_last():
    calc = _calc(queue_level_weight=1.0, reward_scale=0.1, reward_clip=50.0)
    r, _ = calc.compute("T", False, queue_level=500.0, downstream_level=0.0)
    assert r == pytest.approx(-5.0)                                    # cost 500 -> clipped to 50 -> x0.1


def test_defaults_match_the_trained_recipe():
    c = LocalRewardConfig()
    assert (c.queue_level_weight, c.downstream_level_weight, c.switching_penalty) == (1.0, 2.0, 1.0)
    assert (c.backlog_weight, c.global_backlog_weight, c.reward_scale, c.reward_clip) == (0.5, 0.5, 0.1, 200.0)


class _Traci:
    class exceptions:                      # noqa: N801
        TraCIException = Exception

    class simulation:                      # noqa: N801
        pending = ["a", "b", "c"]

        @classmethod
        def getPendingVehicles(cls):
            return cls.pending

    class vehicle:                         # noqa: N801
        calls = 0

        @classmethod
        def getRoute(cls, v):
            cls.calls += 1
            return {"a": ["E0", "x"], "b": ["E0"], "c": ["u", "G"], "d": ["x"]}[v]


def test_pending_cars_are_charged_to_the_first_signalised_edge_on_their_route(monkeypatch):
    monkeypatch.setattr(lr, "traci", _Traci)
    _Traci.vehicle.calls = 0
    calc = _calc(backlog_weight=0.5, global_backlog_weight=0.5)
    calc.refresh_pending({"E0": "T0", "G": "T1"})
    assert calc.pending_by_edge() == {"E0": 2, "G": 1} and calc.pending_for("T0") == 2 and calc.pending_for("T1") == 1
    r, _ = calc.compute("T0", False, 0.0, 0.0, n_agents=3)
    assert r == pytest.approx(-(0.5 * 2 + 0.5 * 3 / 3))                # own share + shared share
    calc.refresh_pending({"E0": "T0", "G": "T1"})
    assert _Traci.vehicle.calls == 3                                   # a route is looked up once per vehicle
    _Traci.simulation.pending = ["d"]                                  # a car that never meets a signal counts only globally
    calc.refresh_pending({"E0": "T0", "G": "T1"})
    assert calc.pending_by_edge() == {} and calc.pending_for("T0") == 0
    _Traci.simulation.pending = ["a", "b", "c"]
    calc.reset()
    assert calc.pending_by_edge() == {}
