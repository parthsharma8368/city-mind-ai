"""
tripinfo.py
===========
Parses SUMO's --tripinfo-output XML into trip-level summary statistics.

In-network per-step averages (mean_waiting_time, mean_queue) are biased by
how many vehicles are in the network at each sampling instant — a
controller that lets vehicles depart faster clears them out of the average
sooner, which can make faster controllers look worse on those metrics.
Tripinfo is per-completed-trip: each row is one vehicle's whole journey,
so trips_completed and mean_timeloss are stable under exactly that effect.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TripinfoSummary:
    trips_completed: int = 0
    mean_trip_duration: float = 0.0
    mean_trip_timeloss: float = 0.0
    mean_trip_waiting_time: float = 0.0


def parse_tripinfo(path: str | Path) -> TripinfoSummary:
    """Parse a --tripinfo-output file. Returns zeros if the file is missing
    or empty (e.g. the run crashed before SUMO flushed it)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return TripinfoSummary()

    try:
        root = ET.parse(p).getroot()
    except ET.ParseError:
        return TripinfoSummary()

    durations, timelosses, waits = [], [], []
    for trip in root.findall("tripinfo"):
        durations.append(float(trip.get("duration", 0.0)))
        timelosses.append(float(trip.get("timeLoss", 0.0)))
        waits.append(float(trip.get("waitingTime", 0.0)))

    n = len(durations)
    if n == 0:
        return TripinfoSummary()

    return TripinfoSummary(
        trips_completed=n,
        mean_trip_duration=sum(durations) / n,
        mean_trip_timeloss=sum(timelosses) / n,
        mean_trip_waiting_time=sum(waits) / n,
    )
