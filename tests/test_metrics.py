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
        mock_traci.simulation.getMinExpectedNumber.return_value = 7
        collector.record_backlog()
        summary = collector.summarize()

    assert summary.total_teleports == 3
    assert summary.backlog == 7
