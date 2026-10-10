"""
drain_metrics.py
================
Drain-aware evaluation protocol (ported from the demand/capacity EDA,
artifacts/eda/demand_capacity_protocol/scripts/{04,06,09,10,11}_*.py).

Why it exists
-------------
Per-completed-trip means (mean_trip_timeloss, mean waiting time) and
``trips_completed`` are survivor-biased: a controller that never lets a long
trip finish, or never lets a car in at a congested entry edge, looks *better*
on them. On the 07:00-08:00 window ~16% of tripinfo rows are zero-length
artifact trips, and vehicles blocked at insertion for more than
max-depart-delay (300 s) are discarded by SUMO and appear in no metric at all.

The protocol scores every run against the SCHEDULED demand instead:

* **Free-flow estimate** per scheduled car: route length / lane speed limits
  plus junction (via-lane) crossing time, from the route file's probability-
  weighted route distribution and the net file, then linearly calibrated
  (a * tff_exp + c) against SUMO's own ``duration - timeLoss`` of
  single-route, never-rerouted completed trips.
* ``ideal = depart + ff_cal``: when the car would be done with zero delay.
* **Lag per scheduled car** = mean over ALL scheduled cars of
  ``min(actual, END) - min(ideal, END)`` clipped at the horizon, where
  ``actual`` is the real arrival, or END (08:00) for a car that is unfinished
  or never inserted. Zero-length trips are included (they always finish
  immediately) so the denominator is the full scheduled-car count.
* **Due-completion ratio** = share of the zero-delay-feasible (ideal <= END),
  non-zero-length cars that actually completed by END.
* Per-entry-edge completion table, completion share per 10-minute departure
  cohort, teleports from SUMO's own statistic output (tripinfo undercounts).
* Per-trip means exclude zero-length trips.

Everything here is pure post-processing of SUMO output files plus the static
route/net files: the offline CLI (scripts/rescore_drain_metrics.py) re-scores
existing artifacts with zero simulation.
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.evaluation.tripinfo import MIN_ROUTE_LENGTH_M, read_tripinfo_rows

# Bump when the *definitions* below change; stored in every result JSON as
# ``drain_protocol``. (``metrics_schema_version`` is a different thing: the core
# version of the per-step metric definitions, src/environment/schema.py.)
PROTOCOL_ID = "drain-v1"

# Entry edges singled out by the EDA (heavy carIn gates; 380271165#0 is a TLS-metered gate).
DEFAULT_ENTRY_EDGES: tuple[str, ...] = ("380271165#0", "450072140", "266507542", "-233673422#10")
# Linear calibration fitted by the EDA on the 18 stored runs (n=8658, R^2=0.948).
# Used by the live evaluator; the offline rescorer re-fits it from the tripinfo it is given.
DEFAULT_FF_CALIBRATION: dict[str, float] = {"a": 0.9877174633263874, "c": 13.038266572183339}
COHORT_MINUTES = 10

_ATTR = re.compile(r'([\w:.-]+)="([^"]*)"')
_DEFAULT_SPEED = 13.9  # m/s, fallback for edges/lanes missing from the net


# ──────────────────────────────────────────────────────────────────────────────
# Static scenario → scheduled-demand table
# ──────────────────────────────────────────────────────────────────────────────
def parse_net(net_file: str | Path) -> tuple[dict[str, tuple[float, float]], dict[tuple[str, str], tuple[float, float]]]:
    """Return ``(edges, transitions)``.

    edges: edge id -> (length of lane 0, speed limit of lane 0) for non-internal edges.
    transitions: (from_edge, to_edge) -> (mean via-lane length, mean via-lane speed)
    averaged over the lane-level connections of that edge pair.
    """
    edges: dict[str, tuple[float, float]] = {}
    ilane: dict[str, tuple[float, float]] = {}
    conns: list[tuple[str, str, str | None]] = []
    for _ev, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag == "edge":
            lanes = el.findall("lane")
            if el.get("function") == "internal":
                for ln in lanes:
                    ilane[ln.get("id")] = (float(ln.get("length")), float(ln.get("speed")))
            elif lanes:
                l0 = lanes[0]
                edges[el.get("id")] = (float(l0.get("length")), float(l0.get("speed")))
            el.clear()
        elif el.tag == "connection":
            conns.append((el.get("from"), el.get("to"), el.get("via")))
            el.clear()
    acc: dict[tuple[str, str], list[float]] = {}
    for fr, to, via in conns:
        ln, sp = ilane.get(via, (0.0, _DEFAULT_SPEED)) if via else (0.0, _DEFAULT_SPEED)
        a = acc.setdefault((fr, to), [0.0, 0.0, 0.0])
        a[0] += ln
        a[1] += sp
        a[2] += 1
    transitions = {k: (v[0] / v[2], v[1] / v[2]) for k, v in acc.items()}
    return edges, transitions


def scan_route_vehicles(route_files: Iterable[str | Path], begin: float, end: float) -> list[dict[str, Any]]:
    """Stream *.rou.xml files; keep vehicles with begin <= depart <= end (route distributions included).

    Line-oriented on purpose (the InTAS route files are ~1.8 GB in total).
    """
    out: list[dict[str, Any]] = []
    for f in route_files:
        cur: dict[str, Any] | None = None
        with open(f, buffering=1 << 20) as fh:
            for line in fh:
                if "<vehicle " in line:
                    a = dict(_ATTR.findall(line))
                    cur = None
                    try:
                        dep = float(a["depart"])
                    except (KeyError, ValueError):
                        continue
                    if begin <= dep <= end:
                        cur = {
                            "id": a["id"], "vType": a.get("type", ""), "depart": dep,
                            "departPos": float(a.get("departPos", 0.0) or 0.0),
                            "arrivalPos": float(a.get("arrivalPos", 0.0) or 0.0),
                            "routes": [],
                        }
                        out.append(cur)
                elif cur is not None and "<route " in line:
                    a = dict(_ATTR.findall(line))
                    if "edges" in a:
                        cur["routes"].append((float(a.get("probability", 1.0)), a["edges"].split()))
    return out


def scheduled_bus_count(route_files: Iterable[str | Path], begin: float, end: float) -> int:
    """Buses scheduled in [begin, end) from ``<flow ... begin end period>`` definitions."""
    n = 0
    for f in route_files:
        if not str(f).endswith(".flow.xml"):
            continue
        for fl in ET.parse(f).getroot().findall("flow"):
            try:
                b, e, p = float(fl.get("begin")), float(fl.get("end")), float(fl.get("period"))
            except (TypeError, ValueError):
                continue
            k = 0
            while b + k * p < e:
                if begin <= b + k * p < end:
                    n += 1
                k += 1
    return n


def build_scheduled_cars(
    route_files: Iterable[str | Path],
    net_file: str | Path,
    begin: float,
    end: float,
) -> pd.DataFrame:
    """One row per scheduled car departing in [begin, end), with free-flow features.

    Columns: id, depart, departPos, arrivalPos, nRoutes, nEdgesMain, firstEdge,
    lastEdge, len_exp, tff_exp, tff_min, prefix, zero_len. (Calibrated ff/ideal
    columns are added by :func:`attach_free_flow`.)
    """
    edges, trans = parse_net(net_file)
    rows = []
    for v in scan_route_vehicles(route_files, begin, end):
        routes = sorted(v["routes"], key=lambda r: -r[0])  # stable: ties keep file order
        if not routes:
            continue
        ptot = sum(p for p, _ in routes)
        Ls, Ts = [], []
        for _p, ed in routes:
            lens = np.array([edges[x][0] if x in edges else 0.0 for x in ed])
            sp = np.array([edges[x][1] if x in edges else _DEFAULT_SPEED for x in ed])
            eff = lens.copy()
            if len(ed) == 1:
                eff[0] = max(v["arrivalPos"] - v["departPos"], 0.0)
            else:
                eff[0] = max(lens[0] - v["departPos"], 0.0)
                eff[-1] = min(v["arrivalPos"], lens[-1])
            vl = np.zeros(max(len(ed) - 1, 0))
            vt = np.zeros_like(vl)
            for i in range(len(ed) - 1):
                ln, s = trans.get((ed[i], ed[i + 1]), (0.0, _DEFAULT_SPEED))
                vl[i] = ln
                vt[i] = ln / max(s, 1.0)
            Ls.append(eff.sum() + vl.sum())
            Ts.append((eff / sp).sum() + vt.sum())
        pw = np.array([p for p, _ in routes]) / ptot
        main = routes[0][1]
        rows.append((
            v["id"], v["depart"], v["departPos"], v["arrivalPos"], len(routes), len(main), main[0], main[-1],
            float(np.dot(pw, Ls)), float(np.dot(pw, Ts)), float(min(Ts)),
        ))
    cars = pd.DataFrame(rows, columns=[
        "id", "depart", "departPos", "arrivalPos", "nRoutes", "nEdgesMain", "firstEdge", "lastEdge",
        "len_exp", "tff_exp", "tff_min",
    ])
    cars["prefix"] = cars["id"].str.extract(r"^([A-Za-z]+)")[0]
    cars["zero_len"] = (cars.nEdgesMain == 1) & (cars.departPos == cars.arrivalPos)
    return cars


def trips_frame(path: str | Path, *, allow_partial: bool = False) -> pd.DataFrame:
    """Strictly-parsed tripinfo rows as a typed DataFrame (see tripinfo.read_tripinfo_rows)."""
    rows, _partial = read_tripinfo_rows(path, allow_partial=allow_partial)
    cols = ["id", "depart", "departDelay", "arrival", "duration", "routeLength", "waitingTime", "timeLoss"]
    df = pd.DataFrame(rows, columns=None if rows else cols)
    for col in ("depart", "departDelay", "arrival", "duration", "routeLength", "waitingTime", "timeLoss"):
        df[col] = df[col].astype(float)
    df["rerouteNo"] = pd.to_numeric(df["rerouteNo"], errors="coerce").fillna(0).astype(int) if "rerouteNo" in df else 0
    if "vType" not in df:
        df["vType"] = ""
    df["is_bus"] = df["vType"].astype(str).str.contains("bus", case=False)
    return df


def calibrate_free_flow(cars: pd.DataFrame, trips: pd.DataFrame) -> dict[str, float]:
    """Fit ff_sumo = a * tff_exp + c on single-route, never-rerouted, non-zero-length completed trips.

    ``trips`` may be the concatenation of many runs (the EDA used all 18).
    """
    d = trips.merge(cars[["id", "nRoutes", "tff_exp"]], on="id")
    d = d[(d.nRoutes == 1) & (d.rerouteNo == 0) & (d.routeLength > 0)]
    if len(d) < 50:
        raise ValueError(f"only {len(d)} usable trips to calibrate free-flow time (need >= 50)")
    ff = (d.duration - d.timeLoss).to_numpy()
    A = np.c_[d.tff_exp.to_numpy(), np.ones(len(d))]
    coef = np.linalg.lstsq(A, ff, rcond=None)[0]
    pred = A @ coef
    r2 = 1 - ((ff - pred) ** 2).sum() / ((ff - ff.mean()) ** 2).sum()
    return {"a": float(coef[0]), "c": float(coef[1]), "n": len(d), "r2": float(r2),
            "mae": float(np.abs(ff - pred).mean())}


def attach_free_flow(cars: pd.DataFrame, calibration: dict[str, float]) -> pd.DataFrame:
    """Add ``ff_cal`` (zero-length trips: 1.0 s) and ``ideal = depart + ff_cal``."""
    out = cars.copy()
    out["ff_cal"] = np.where(out.zero_len, 1.0, calibration["a"] * out.tff_exp + calibration["c"])
    out["ideal"] = out.depart + out.ff_cal
    return out


@dataclass
class ScheduledDemand:
    """The scheduled-car table for one (scenario, window) plus calibration metadata."""
    cars: pd.DataFrame            # incl. ff_cal and ideal
    begin: float
    end: float
    scheduled_buses: int
    calibration: dict[str, float]
    calibration_source: str

    @property
    def n_cars(self) -> int:
        return len(self.cars)

    @property
    def n_scheduled_total(self) -> int:
        return self.n_cars + self.scheduled_buses


def _fingerprint(files: Iterable[str | Path], begin: float, end: float) -> str:
    h = hashlib.sha256(f"{PROTOCOL_ID}|{begin}|{end}".encode())
    for f in files:
        p = Path(f)
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


def load_scheduled_cars(
    route_files: list[str | Path],
    net_file: str | Path,
    begin: float,
    end: float,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """build_scheduled_cars with an on-disk parquet cache (route files are huge)."""
    cache_path = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / f"scheduled_cars_{_fingerprint([*route_files, net_file], begin, end)}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
    cars = build_scheduled_cars(route_files, net_file, begin, end)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cars.to_parquet(cache_path, index=False)
    return cars


def build_scheduled_demand(
    route_files: list[str | Path],
    net_file: str | Path,
    begin: float,
    end: float,
    *,
    calibration_trips: pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
) -> ScheduledDemand:
    """Scheduled cars + calibrated free-flow times for [begin, end).

    With ``calibration_trips`` (tripinfo of existing runs) the free-flow
    calibration is re-fitted; otherwise DEFAULT_FF_CALIBRATION is used.
    """
    all_cars = load_scheduled_cars(route_files, net_file, begin, end, cache_dir)
    if calibration_trips is not None:
        calib = calibrate_free_flow(all_cars, calibration_trips)
        source = f"fitted on {calib['n']} stored trips"
    else:
        calib = dict(DEFAULT_FF_CALIBRATION)
        source = "default constants from the demand/capacity EDA"
    cars = attach_free_flow(all_cars[all_cars.depart < end].reset_index(drop=True), calib)
    return ScheduledDemand(
        cars=cars, begin=begin, end=end,
        scheduled_buses=scheduled_bus_count(route_files, begin, end),
        calibration=calib, calibration_source=source,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SUMO statistic / summary output (reliable teleport and insertion counters)
# ──────────────────────────────────────────────────────────────────────────────
def parse_statistic_output(path: str | Path) -> dict[str, int]:
    """Counters from SUMO ``--statistic-output``: loaded/inserted/waiting, teleports, collisions.

    Raises ValueError if the file is missing or not a SUMO statistics file.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise ValueError(f"statistic-output missing or empty: {p}")
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"statistic-output unparsable: {p}: {exc}") from exc
    veh, tel, saf = root.find("vehicles"), root.find("teleports"), root.find("safety")
    if veh is None or tel is None:
        raise ValueError(f"{p} has no <vehicles>/<teleports> element")
    out = {
        "loaded": int(veh.get("loaded")), "inserted": int(veh.get("inserted")),
        "running": int(veh.get("running")), "waiting": int(veh.get("waiting")),
        "teleports_total": int(tel.get("total")), "teleports_jam": int(tel.get("jam", 0)),
        "teleports_yield": int(tel.get("yield", 0)), "teleports_wrong_lane": int(tel.get("wrongLane", 0)),
    }
    if saf is not None:
        out["collisions"] = int(saf.get("collisions", 0))
    return out


def parse_summary_discarded(path: str | Path) -> int | None:
    """``discarded`` (vehicles dropped by max-depart-delay) from the last <step> of --summary-output."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    with p.open("rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 8192))
        tail = fh.read().decode("utf-8", "ignore")
    m = re.findall(r'<step [^>]*?discarded="(\d+)"', tail)
    return int(m[-1]) if m else None


# ──────────────────────────────────────────────────────────────────────────────
# Scoring one run
# ──────────────────────────────────────────────────────────────────────────────
def _safe_ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def top_origin_edges(cars: pd.DataFrame, k: int = 10) -> list[str]:
    nz = cars[~cars.zero_len]
    return list(nz.firstEdge.value_counts().head(k).index)


def score_run(
    demand: ScheduledDemand,
    trips: pd.DataFrame,
    *,
    entry_edges: Iterable[str] = DEFAULT_ENTRY_EDGES,
    n_top_origin_edges: int = 10,
    seen_ids: set[str] | None = None,
    inserted: int | None = None,
    statistic: dict[str, int] | None = None,
    json_teleports: int | None = None,
    discarded: int | None = None,
    backlog: int | None = None,
) -> dict[str, Any]:
    """Drain-aware metrics for one run from its tripinfo rows (``trips_frame``).

    seen_ids: optional ids of vehicles ever observed in the network (from a
    trajectory) so inserted-but-unfinished cars can be told from never-inserted ones.
    inserted: SUMO's inserted count (JSON ``total_departed`` or statistic-output).
    """
    cars = demand.cars
    W1 = float(demand.end)
    ids = cars["id"].to_numpy()
    ideal = cars["ideal"].to_numpy()
    zero = cars["zero_len"].to_numpy()
    nonzero = ~zero

    g = trips[~trips["is_bus"]].drop_duplicates("id")
    arr = g.set_index("id")["arrival"].reindex(ids)
    done = arr.notna().to_numpy()

    due = ideal <= W1
    due_nz = due & nonzero
    # lag = (W1 - ideal)^+ - (W1 - arrival)^+   [arrival := W1 for unfinished / never-inserted]
    lag = np.clip(W1 - ideal, 0, None) - np.where(done, np.clip(W1 - arr.fillna(W1).to_numpy(), 0, None), 0.0)
    n_cars = len(cars)
    n_due_nz = int(due_nz.sum())

    gn = g[g.routeLength >= MIN_ROUTE_LENGTH_M]
    out: dict[str, Any] = {
        "drain_protocol": PROTOCOL_ID,
        "scheduled_cars": n_cars,
        "scheduled_zero_length_cars": int(zero.sum()),
        "scheduled_buses": demand.scheduled_buses,
        "cars_completed": int(done.sum()),
        "cars_completed_nonzero": int((done & nonzero).sum()),
        "zero_len_completed": int((done & zero).sum()),
        "due_nz_n": n_due_nz,
        "due_nz_completed": int((done & due_nz).sum()),
        "due_completion_ratio": _safe_ratio((done & due_nz).sum(), n_due_nz),
        "lag_per_scheduled_car_s": float(lag.sum() / n_cars) if n_cars else None,
        "lag_per_due_nz_car_s": float(lag[due_nz].sum() / n_due_nz) if n_due_nz else None,
        "lag_area_veh_h": float(lag.sum() / 3600.0),
        "overdue_cars": int((due & ~done).sum()),
        "mean_timeloss_cars_nonzero": float(gn.timeLoss.mean()) if len(gn) else None,
        "mean_timeloss_plus_departdelay_cars_nonzero": float((gn.timeLoss + gn.departDelay).mean()) if len(gn) else None,
        "mean_depart_delay_cars": float(g.departDelay.mean()) if len(g) else None,
        "completed_veh_km": float(g.routeLength.sum() / 1000.0),
        "tripinfo_rows": len(trips),
        "tripinfo_zero_length_rows": int((trips.routeLength < MIN_ROUTE_LENGTH_M).sum()),
        "ff_calibration": {k: demand.calibration[k] for k in ("a", "c")},
    }

    # --- insertion accounting ---------------------------------------------------------
    if statistic is not None:
        inserted = statistic["inserted"] if inserted is None else inserted
    if inserted is not None:
        out["inserted"] = int(inserted)
        out["never_inserted"] = int(demand.n_scheduled_total - inserted)
    if discarded is not None:
        out["discarded_max_depart_delay"] = int(discarded)
    if backlog is not None:
        out["backlog"] = int(backlog)

    # --- teleports: SUMO's own statistic > metrics-collector count; tripinfo is NOT a source ---
    if statistic is not None:
        out["teleports"] = int(statistic["teleports_total"])
        out["teleports_source"] = "sumo --statistic-output"
        out["sumo_loaded"] = statistic["loaded"]
        out["sumo_waiting_for_insertion"] = statistic["waiting"]
        if "collisions" in statistic:
            out["collisions"] = statistic["collisions"]
    elif json_teleports is not None:
        out["teleports"] = int(json_teleports)
        out["teleports_source"] = "MetricsCollector (getStartingTeleportNumber)"

    # --- per-entry-edge table -----------------------------------------------------------
    edges = list(dict.fromkeys([*entry_edges, *top_origin_edges(cars, n_top_origin_edges)]))
    first = cars["firstEdge"].to_numpy()
    dd = g.set_index("id")["departDelay"].reindex(ids).to_numpy()
    seen = np.array([i in seen_ids for i in ids]) if seen_ids is not None else None
    table = []
    for e in edges:
        m = (first == e) & nonzero
        sched = int(m.sum())
        row: dict[str, Any] = {
            "edge": e, "role": "entry" if e in set(entry_edges) else "top_origin",
            "scheduled_nonzero": sched, "completed": int((m & done).sum()),
            "completion_ratio": _safe_ratio((m & done).sum(), sched),
            "due_nz": int((m & due).sum()), "due_completed": int((m & due & done).sum()),
            "mean_depart_delay_completed_s": float(np.nanmean(dd[m & done])) if (m & done).any() else None,
            "completed_at_cap": int(((dd >= 299) & m & done).sum()),
        }
        if seen is not None:
            ins = m & (done | seen)
            row["inserted"] = int(ins.sum())
            row["never_inserted"] = sched - int(ins.sum())
        table.append(row)
    out["entry_edges"] = table

    # --- completion share per departure cohort (non-zero cars, planned departure) --------
    depart = cars["depart"].to_numpy()
    cohorts = []
    span = round((demand.end - demand.begin) / 60)
    for a in range(0, span, COHORT_MINUTES):
        b = min(a + COHORT_MINUTES, span)
        m = nonzero & (depart >= demand.begin + a * 60) & (depart < demand.begin + b * 60)
        sched = int(m.sum())
        cohorts.append({
            "cohort": f"{a}-{b}", "scheduled_nonzero": sched, "completed": int((m & done).sum()),
            "completion_share": _safe_ratio((m & done).sum(), sched),
        })
    out["cohorts"] = cohorts
    return out


def score_tripinfo_file(
    demand: ScheduledDemand,
    tripinfo_path: str | Path,
    *,
    statistic_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    inserted: int | None = None,
    json_teleports: int | None = None,
    backlog: int | None = None,
) -> dict[str, Any]:
    """Convenience wrapper used by the live evaluator and the offline CLI."""
    trips = trips_frame(tripinfo_path)
    statistic = None
    if statistic_path and Path(statistic_path).exists():
        statistic = parse_statistic_output(statistic_path)
    discarded = parse_summary_discarded(summary_path) if summary_path else None
    seen = None
    if trajectory_path and Path(trajectory_path).exists():
        t = pd.read_parquet(trajectory_path, columns=["vehicle_id"])
        seen = set(t["vehicle_id"].dropna().unique())
    return score_run(
        demand, trips, seen_ids=seen, inserted=inserted, statistic=statistic,
        json_teleports=json_teleports, discarded=discarded, backlog=backlog,
    )


FLAT_KEYS = (
    "drain_protocol", "lag_per_scheduled_car_s", "lag_per_due_nz_car_s", "due_completion_ratio",
    "due_nz_n", "due_nz_completed", "cars_completed_nonzero", "zero_len_completed", "overdue_cars",
    "mean_timeloss_cars_nonzero", "mean_timeloss_plus_departdelay_cars_nonzero", "mean_depart_delay_cars",
    "completed_veh_km", "lag_area_veh_h", "never_inserted", "discarded_max_depart_delay", "teleports",
    "teleports_source",
)


def flat_fields(score: dict[str, Any]) -> dict[str, Any]:
    """The scalar subset of a score that is merged into result JSONs / parquet rows."""
    return {k: score[k] for k in FLAT_KEYS if k in score}


def drain_json_fields(score: dict[str, Any]) -> dict[str, Any]:
    """New fields for the evaluation JSON: flat scalars + the nested tables under ``drain_metrics``."""
    return {
        **flat_fields(score),
        "drain_metrics": {k: score[k] for k in ("entry_edges", "cohorts", "ff_calibration", "scheduled_cars",
                                                "scheduled_zero_length_cars", "scheduled_buses",
                                                "tripinfo_rows", "tripinfo_zero_length_rows") if k in score},
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=float)
