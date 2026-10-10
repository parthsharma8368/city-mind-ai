"""Regression tests for MetricsCollector's departure/arrival/teleport counting.

Guards against the undercounting bug fixed alongside the peak-hour mechanics
work: every controller calls traci.simulationStep() once per raw simulation
second but only sampled metrics once per decision interval (every 5th
step). getDepartedNumber()/getArrivedNumber()/getStartingTeleportNumber()
report counts SINCE THE LAST CALL, not cumulative totals, so without an
accumulator in between, 4 out of every 5 steps' departures and arrivals
were silently dropped.
"""
from __future__ import annotations

from unittest.mock import patch

from src.environment.metrics import MetricsCollector


def test_tick_accumulates_across_multiple_steps_before_record_step():
    """5 raw steps (5, 3, 0, 2, 1 departures) must sum to 11 at record_step,
    not just report the last step's count (1)."""
    departed_per_step = [5, 3, 0, 2, 1]
    arrived_per_step = [1, 1, 4, 0, 2]

    collector = MetricsCollector()
    with patch("src.environment.metrics.traci") as mock_traci:
        mock_traci.simulation.getDepartedNumber.return_value = 0
        collector.reset("test_controller", seed=42, demand_id="1.0x")

        for departed, arrived in zip(departed_per_step, arrived_per_step):
            mock_traci.simulation.getDepartedNumber.return_value = departed
            mock_traci.simulation.getArrivedNumber.return_value = arrived
            mock_traci.simulation.getStartingTeleportNumber.return_value = 0
            collector.tick()

        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.trafficlight.getIDList.return_value = []
        mock_traci.lane.getIDList.return_value = []
        rec = collector.record_step(step=5, sim_time=5.0, reward=0.0, switch_count=0)

    assert rec.departed_vehicles == sum(departed_per_step)
    assert rec.arrived_vehicles == sum(arrived_per_step)


def test_record_step_without_any_tick_reports_zero_not_a_fresh_traci_read():
    """record_step() must never fall back to a fresh getDepartedNumber() call —
    that reintroduces the exact bug (counting only "since the last raw
    simulationStep()", i.e. whatever happened after the last tick())."""
    collector = MetricsCollector()
    with patch("src.environment.metrics.traci") as mock_traci:
        mock_traci.simulation.getDepartedNumber.return_value = 0
        collector.reset("test_controller", seed=42, demand_id="1.0x")

        # Simulate a fresh, un-ticked traci state that would look like a large
        # undercounted-but-nonzero read if record_step ever queried it directly.
        mock_traci.simulation.getDepartedNumber.return_value = 999
        mock_traci.simulation.getArrivedNumber.return_value = 999
        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.trafficlight.getIDList.return_value = []
        mock_traci.lane.getIDList.return_value = []
        rec = collector.record_step(step=0, sim_time=0.0, reward=0.0, switch_count=0)

    assert rec.departed_vehicles == 0
    assert rec.arrived_vehicles == 0


def test_teleports_accumulate_over_full_episode():
    collector = MetricsCollector()
    with patch("src.environment.metrics.traci") as mock_traci:
        mock_traci.simulation.getDepartedNumber.return_value = 0
        mock_traci.simulation.getArrivedNumber.return_value = 0
        collector.reset("test_controller", seed=42, demand_id="1.0x")

        for teleports in (0, 1, 0, 2):
            mock_traci.simulation.getStartingTeleportNumber.return_value = teleports
            collector.tick()

        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.trafficlight.getIDList.return_value = []
        mock_traci.lane.getIDList.return_value = []
        collector.record_step(step=4, sim_time=4.0, reward=0.0, switch_count=0)
        # backlog (metrics schema v2) = in-network + pending, NOT the
        # route-steps-dependent getMinExpectedNumber.
        mock_traci.vehicle.getIDList.return_value = ["a", "b", "c", "d", "e"]
        mock_traci.simulation.getPendingVehicles.return_value = ["p", "q"]
        mock_traci.simulation.getMinExpectedNumber.return_value = 999
        collector.record_backlog()
        summary = collector.summarize()

    assert summary.total_teleports == 3
    assert summary.backlog == 7


# ──────────────────────────────────────────────────────────────────────────────
# Unified metric definition + summarize() with known inputs (tests-reliability-002)
# ──────────────────────────────────────────────────────────────────────────────
import pytest  # noqa: E402
import traci.exceptions as texc  # noqa: E402

from src.environment.metrics import (  # noqa: E402
    GreenSwitchCounter,
    compute_network_queue_speed,
    unique_controlled_lanes,
)
from src.environment.schema import METRICS_SCHEMA_VERSION  # noqa: E402
from src.scenario.tls_metadata import TLSRuntimeState  # noqa: E402


def _wire(mock_traci):
    mock_traci.exceptions.TraCIException = texc.TraCIException
    mock_traci.exceptions.FatalTraCIError = texc.FatalTraCIError


@patch("src.environment.metrics.traci")
def test_queue_speed_definition_is_per_vehicle_speed_over_unique_lanes(mock_traci):
    _wire(mock_traci)
    halting = {"a": 2, "b": 4, "c": 0}
    mock_traci.lane.getLastStepHaltingNumber.side_effect = lambda lid: halting[lid]
    speeds = {"v1": 10.0, "v2": 0.0, "v3": 5.0}
    mock_traci.vehicle.getSpeed.side_effect = lambda v: speeds[v]
    total_q, mean_q, max_q, mean_spd = compute_network_queue_speed(["a", "b", "c"], ["v1", "v2", "v3"])
    assert (total_q, mean_q, max_q) == (6.0, 2.0, 4.0)
    assert mean_spd == pytest.approx(5.0)               # per vehicle, NOT a mean of lane speeds
    assert compute_network_queue_speed([], []) == (0.0, 0.0, 0.0, 0.0)


@patch("src.environment.metrics.traci")
def test_unique_controlled_lanes_dedupes_and_sorts(mock_traci):
    mock_traci.trafficlight.getIDList.return_value = ["t1", "t2"]
    mock_traci.trafficlight.getControlledLanes.side_effect = lambda t: {
        "t1": ["b", "a", "a", "b"], "t2": ["c", "a"]}[t]
    assert unique_controlled_lanes() == ["a", "b", "c"]


@patch("src.environment.metrics.traci")
def test_record_step_ignores_states_so_rl_and_baselines_share_one_definition(mock_traci):
    """eval-scripts-2: RL envs used to pass states= and got a mean of per-lane
    speeds (empty lanes at free-flow 13.89) over per-link duplicated lanes."""
    _wire(mock_traci)
    mock_traci.trafficlight.getIDList.return_value = ["t"]
    mock_traci.trafficlight.getControlledLanes.return_value = ["a", "a", "b"]   # per-link duplicates
    mock_traci.lane.getLastStepHaltingNumber.side_effect = lambda lid: {"a": 3, "b": 1}[lid]
    mock_traci.vehicle.getIDList.return_value = ["v"]
    mock_traci.vehicle.getSpeed.return_value = 2.0
    mock_traci.vehicle.getWaitingTime.return_value = 7.0
    mock_traci.simulation.getDepartedNumber.return_value = 0
    states = {"t": TLSRuntimeState(tls_id="t")}
    states["t"].lane_queue = [9.0, 9.0, 9.0]
    states["t"].lane_speed = [13.89, 13.89, 13.89]

    c1 = MetricsCollector()
    c1.reset("a", 1, "x")
    with_states = c1.record_step(1, 5.0, 0.0, 0, states=states)
    c2 = MetricsCollector()
    c2.reset("a", 1, "x")
    without = c2.record_step(1, 5.0, 0.0, 0)
    assert with_states == without
    assert without.average_speed == 2.0 and without.total_queue == 4.0
    assert without.average_queue == 2.0 and without.max_queue == 3.0


@patch("src.environment.metrics.traci")
def test_summarize_with_known_records(mock_traci):
    _wire(mock_traci)
    mock_traci.simulation.getDepartedNumber.return_value = 0
    c = MetricsCollector()
    c.reset("ctrl", 7, "1.0x", sumo_command=["sumo", "--seed", "7"])
    from src.environment.metrics import StepRecord

    def rec(step, t, dep, arr, wait, q, mq, spd, sw, rew):
        return StepRecord(step, t, 10, dep, arr, wait * 10, wait, q, q / 2, mq, spd, sw, rew)

    c._records = [
        rec(1, 5.0, 10, 4, 2.0, 6.0, 4.0, 8.0, 1, -1.0),
        rec(2, 10.0, 6, 5, 4.0, 10.0, 9.0, 4.0, 2, -3.0),
    ]
    c._pending_teleports = 3
    c._backlog = 11
    s = c.summarize()
    assert s.metrics_schema_version == METRICS_SCHEMA_VERSION == 2
    assert s.sumo_command == ["sumo", "--seed", "7"]
    assert (s.total_departed, s.total_arrived) == (16, 9)
    assert s.throughput == pytest.approx(9 / 16)         # arrived / departed (not inverted)
    assert s.mean_waiting_time == pytest.approx(3.0)
    assert s.total_waiting_time_sum == pytest.approx(60.0)
    assert s.mean_queue == pytest.approx((3.0 + 5.0) / 2)
    assert s.max_queue == 9.0
    assert s.mean_speed == pytest.approx(6.0)
    assert s.total_switches == 3
    assert s.cumulative_reward == pytest.approx(-4.0)
    assert s.episode_duration == 5.0 and s.episode_steps == 2
    assert (s.total_teleports, s.backlog) == (3, 11)


@patch("src.environment.metrics.traci")
def test_tick_traci_error_is_logged_not_fatal_but_fatal_error_propagates(mock_traci, caplog):
    _wire(mock_traci)
    c = MetricsCollector()
    mock_traci.simulation.getDepartedNumber.side_effect = texc.TraCIException("boom")
    with caplog.at_level("WARNING"):
        c.tick()
    assert "tick()" in caplog.text and c._pending_departed == 0
    mock_traci.simulation.getDepartedNumber.side_effect = texc.FatalTraCIError("connection closed")
    with pytest.raises(texc.FatalTraCIError):
        c.tick()


@patch("src.environment.metrics.traci")
def test_failed_record_step_does_not_zero_the_pending_counts(mock_traci):
    _wire(mock_traci)
    mock_traci.simulation.getDepartedNumber.return_value = 3
    mock_traci.simulation.getArrivedNumber.return_value = 2
    mock_traci.simulation.getStartingTeleportNumber.return_value = 0
    c = MetricsCollector()
    c.reset("c", 1, "x")
    c.tick()
    mock_traci.trafficlight.getIDList.return_value = []
    mock_traci.lane.getIDList.return_value = []
    mock_traci.vehicle.getIDList.side_effect = texc.TraCIException("read failed")
    c.record_step(1, 5.0, 0.0, 0)                       # swallowed, logged
    assert (c._pending_departed, c._pending_arrived) == (3, 2)   # kept for the next record
    mock_traci.vehicle.getIDList.side_effect = None
    mock_traci.vehicle.getIDList.return_value = []
    rec = c.record_step(2, 10.0, 0.0, 0)
    assert (rec.departed_vehicles, rec.arrived_vehicles) == (3, 2)


@patch("src.environment.metrics.traci")
def test_flush_and_summarize_fold_trailing_counts_into_last_record(mock_traci):
    _wire(mock_traci)
    mock_traci.simulation.getDepartedNumber.return_value = 2
    mock_traci.simulation.getArrivedNumber.return_value = 1
    mock_traci.simulation.getStartingTeleportNumber.return_value = 0
    mock_traci.trafficlight.getIDList.return_value = []
    mock_traci.lane.getIDList.return_value = []
    mock_traci.vehicle.getIDList.return_value = []
    c = MetricsCollector()
    c.reset("c", 1, "x")
    c.tick()
    c.record_step(1, 1.0, 0.0, 0)
    c.tick()                                            # steps after the last record
    c.tick()
    s = c.summarize()
    assert (s.total_departed, s.total_arrived) == (6, 3)


def test_green_switch_counter_counts_green_to_different_green_only():
    g = GreenSwitchCounter()
    seq = ["GGrr", "GGrr", "yyrr", "rrrr", "rrGG", "rrGG", "rryy", "GGrr"]
    assert [g.update("t", s) for s in seq] == [False, False, False, False, True, False, False, True]
    assert g.total == 2
    assert g.update("other", "GGrr") is False             # first sighting is a baseline, not a switch


@patch("src.environment.metrics.traci")
def test_backlog_is_in_network_plus_pending_independent_of_route_steps(mock_traci):
    _wire(mock_traci)
    mock_traci.vehicle.getIDList.return_value = ["a", "b"]
    mock_traci.simulation.getPendingVehicles.return_value = ["p"]
    mock_traci.simulation.getMinExpectedNumber.return_value = 5000   # route-steps-dependent, must be unused
    c = MetricsCollector()
    c.record_backlog()
    assert c._backlog == 3
