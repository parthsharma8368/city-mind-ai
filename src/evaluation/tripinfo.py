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

Two caveats that every consumer must keep in mind (see
src/evaluation/drain_metrics.py for the drain-aware protocol that fixes them):

1. ~16% of tripinfo rows are zero-length artifact trips (routeLength == 0,
   duration ~1 s, 'h'-prefixed ids: origin edge == destination edge). They
   are counted in the legacy ``trips_completed`` / ``mean_trip_*`` fields for
   backwards compatibility, but the ``*_nonzero`` fields exclude them and
   ``zero_length_trips`` reports how many there were.
2. Only vehicles that FINISH inside the window appear here: a controller that
   leaves long trips unfinished looks better on any per-completed-trip mean.

Strictness: a missing, empty, truncated or attribute-incomplete file raises
:class:`TripinfoError` — it is never silently turned into "0 trips, 0.0 s of
time loss" (which would rank best on a lower-is-better metric). Pass
``allow_partial=True`` to salvage the complete rows of a truncated file; the
result then carries ``partial=True``.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# Attributes every <tripinfo> row must carry for the summary statistics.
REQUIRED_ATTRS = ("id", "depart", "departDelay", "arrival", "duration",
                  "routeLength", "waitingTime", "timeLoss")
# A route shorter than this is a zero-length artifact trip (origin == destination).
MIN_ROUTE_LENGTH_M = 1.0


class TripinfoError(ValueError):
    """The tripinfo file is missing, empty, truncated or malformed."""


@dataclass
class TripinfoSummary:
    # --- legacy fields (average over EVERY row, zero-length artifacts included) ---
    trips_completed: int = 0
    mean_trip_duration: float = 0.0
    mean_trip_timeloss: float = 0.0
    mean_trip_waiting_time: float = 0.0
    # --- new fields: zero-length artifact trips excluded ---
    zero_length_trips: int = 0
    trips_completed_nonzero: int = 0
    mean_trip_duration_nonzero: float = 0.0
    mean_trip_timeloss_nonzero: float = 0.0
    mean_trip_waiting_time_nonzero: float = 0.0
    mean_trip_timeloss_plus_departdelay_nonzero: float = 0.0
    partial: bool = False


def read_tripinfo_rows(path: str | Path, *, allow_partial: bool = False) -> tuple[list[dict[str, str]], bool]:
    """Return ``(rows, partial)``; each row is the raw attribute dict of one <tripinfo>.

    Raises TripinfoError on a missing/empty file, an unparseable file (unless
    ``allow_partial``) or a row missing a required attribute.
    """
    p = Path(path)
    if not p.exists():
        raise TripinfoError(f"tripinfo file not found: {p}")
    if p.stat().st_size == 0:
        raise TripinfoError(f"tripinfo file is empty: {p}")

    rows: list[dict[str, str]] = []
    partial = False
    try:
        for _event, el in ET.iterparse(str(p), events=("end",)):
            if el.tag != "tripinfo":
                continue
            attrs = dict(el.attrib)
            missing = [a for a in REQUIRED_ATTRS if a not in attrs]
            if missing:
                raise TripinfoError(
                    f"{p}: <tripinfo id={attrs.get('id')!r}> is missing attribute(s) {missing}"
                )
            rows.append(attrs)
            el.clear()
    except ET.ParseError as exc:
        if not allow_partial:
            raise TripinfoError(
                f"{p}: truncated or malformed tripinfo XML ({exc}); "
                f"{len(rows)} complete rows were readable. Re-run the episode."
            ) from exc
        partial = True
    return rows, partial


def parse_tripinfo(path: str | Path, *, allow_partial: bool = False) -> TripinfoSummary:
    """Parse a --tripinfo-output file into a :class:`TripinfoSummary`.

    A valid file that contains zero ``<tripinfo>`` rows yields an all-zero
    summary (a legitimately empty run); anything else wrong raises.
    """
    rows, partial = read_tripinfo_rows(path, allow_partial=allow_partial)

    dur = [float(r["duration"]) for r in rows]
    tl = [float(r["timeLoss"]) for r in rows]
    wt = [float(r["waitingTime"]) for r in rows]
    dd = [float(r["departDelay"]) for r in rows]
    nz = [float(r["routeLength"]) >= MIN_ROUTE_LENGTH_M for r in rows]

    n = len(rows)
    if n == 0:
        return TripinfoSummary(partial=partial)

    def _mean(vals: list[float], mask: list[bool] | None = None) -> float:
        sel = vals if mask is None else [v for v, m in zip(vals, mask) if m]
        return sum(sel) / len(sel) if sel else 0.0

    n_nz = sum(nz)
    return TripinfoSummary(
        trips_completed=n,
        mean_trip_duration=_mean(dur),
        mean_trip_timeloss=_mean(tl),
        mean_trip_waiting_time=_mean(wt),
        zero_length_trips=n - n_nz,
        trips_completed_nonzero=n_nz,
        mean_trip_duration_nonzero=_mean(dur, nz),
        mean_trip_timeloss_nonzero=_mean(tl, nz),
        mean_trip_waiting_time_nonzero=_mean(wt, nz),
        mean_trip_timeloss_plus_departdelay_nonzero=_mean([a + b for a, b in zip(tl, dd)], nz),
        partial=partial,
    )
