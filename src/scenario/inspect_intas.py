"""
inspect_intas.py
================
Programmatically inspect the InTAS SUMO scenario and emit machine-readable
metadata artifacts:

  artifacts/scenario/tls_metadata.json
  artifacts/scenario/network_metadata.json
  artifacts/scenario/demand_metadata.json

Run:
  python -m src.scenario.inspect_intas --scenario-dir data/raw/intas
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── SUMO_HOME guard (import-safe: resolves the macOS default / PATH sumo) ───
from src.environment.sumo_connection import resolve_sumo_home  # noqa: E402

SUMO_HOME = resolve_sumo_home()
if SUMO_HOME is None:
    raise EnvironmentError(
        "SUMO_HOME is not set and no `sumo` binary was found on PATH. "
        "Export SUMO_HOME to your SUMO installation's share/sumo directory."
    )
if os.path.join(SUMO_HOME, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))

import traci  # noqa: E402  (after sys.path patch)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseInfo:
    index: int
    state: str           # e.g. "GGrr"
    duration: float      # seconds
    min_dur: float = 0.0
    max_dur: float = 0.0

@dataclass
class TLSInfo:
    tls_id: str
    program_id: str
    type: str            # "static" | "actuated" | "delay_based" | etc.
    phases: list[PhaseInfo] = field(default_factory=list)
    controlled_lanes: list[str] = field(default_factory=list)
    controlled_links: list[list] = field(default_factory=list)
    # rl-core-007: signal_lanes[i] = incoming lane(s) of signal index i
    signal_lanes: list[list[str]] = field(default_factory=list)
    junction_id: str = ""

@dataclass
class NetworkMetadata:
    net_file: str
    junction_count: int = 0
    edge_count: int = 0
    lane_count: int = 0
    tls_count: int = 0
    boundary: list[float] = field(default_factory=list)  # [xmin, ymin, xmax, ymax]
    junctions: list[dict] = field(default_factory=list)

@dataclass
class DemandMetadata:
    route_files: list[str] = field(default_factory=list)
    additional_files: list[str] = field(default_factory=list)
    vehicle_types: list[str] = field(default_factory=list)
    total_route_count: int = 0
    simulation_begin: float = 0.0
    simulation_end: float = 0.0
    step_length: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: find SUMO configuration file
# ─────────────────────────────────────────────────────────────────────────────

def find_sumocfg(scenario_dir: Path) -> Path:
    """Return the first .sumocfg found in scenario_dir (recursively)."""
    candidates = list(scenario_dir.rglob("*.sumocfg"))
    if not candidates:
        raise FileNotFoundError(
            f"No .sumocfg found under {scenario_dir}. "
            "Make sure InTAS is downloaded to data/raw/intas."
        )
    # Prefer top-level configs
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def parse_sumocfg(cfg_path: Path) -> dict[str, Any]:
    """Parse a .sumocfg XML and return a dict with resolved paths."""
    tree = ET.parse(cfg_path)
    root = tree.getroot()
    base = cfg_path.parent

    def _resolve(raw: str) -> list[Path]:
        """Split comma-separated file refs and resolve against base."""
        return [base / p.strip() for p in raw.split(",") if p.strip()]

    result: dict[str, Any] = {
        "cfg_path": str(cfg_path),
        "net_file": None,
        "route_files": [],
        "additional_files": [],
        "begin": 0.0,
        "end": 86400.0,
        "step_length": 1.0,
    }

    input_el = root.find("input")
    if input_el is not None:
        nf = input_el.find("net-file")
        if nf is not None:
            result["net_file"] = base / nf.get("value", "")
        rf = input_el.find("route-files")
        if rf is not None:
            result["route_files"] = _resolve(rf.get("value", ""))
        af = input_el.find("additional-files")
        if af is not None:
            result["additional_files"] = _resolve(af.get("value", ""))

    time_el = root.find("time")
    if time_el is not None:
        b = time_el.find("begin")
        e = time_el.find("end")
        s = time_el.find("step-length")
        if b is not None:
            result["begin"] = float(b.get("value", 0))
        if e is not None:
            result["end"] = float(e.get("value", 86400))
        if s is not None:
            result["step_length"] = float(s.get("value", 1.0))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Network inspection via TraCI (short simulation)
# ─────────────────────────────────────────────────────────────────────────────

def signal_lanes_from_links(controlled_links) -> list[list[str]]:
    """traci.trafficlight.getControlledLinks(tid) -> per-signal list of the
    distinct INCOMING lanes (link tuples are (inLane, outLane, viaLane))."""
    out: list[list[str]] = []
    for signal_links in controlled_links:
        lanes: list[str] = []
        for link in signal_links:
            if link and link[0] and link[0] not in lanes:
                lanes.append(link[0])
        out.append(lanes)
    return out


def add_signal_lanes_to_metadata(json_path: Path, cfg_path: Path, begin: float = 0.0) -> int:
    """Additively patch an existing tls_metadata.json with ``signal_lanes``
    (no other field is touched). Returns the number of TLS patched."""
    data = json.loads(Path(json_path).read_text())
    cmd = ["sumo", "--configuration-file", str(cfg_path), "--begin", str(begin),
           "--end", str(begin + 2), "--no-warnings", "--no-step-log"]
    traci.start(cmd, label="signal_lanes")
    try:
        patched = 0
        for entry in data:
            links = traci.trafficlight.getControlledLinks(entry["tls_id"])
            entry["signal_lanes"] = signal_lanes_from_links(links)
            patched += 1
    finally:
        traci.close()
    Path(json_path).write_text(json.dumps(data, indent=2, default=str))
    return patched


def inspect_via_traci(cfg: dict[str, Any]) -> tuple[list[TLSInfo], NetworkMetadata]:
    """
    Start SUMO headlessly for one step to interrogate TraCI.
    Returns (tls_info_list, network_metadata).
    """
    cmd = [
        "sumo",
        "--configuration-file", cfg["cfg_path"],
        "--begin", str(cfg["begin"]),
        "--end", str(cfg["begin"] + 2),   # run only 2 seconds
        "--no-warnings",
        "--no-step-log",
    ]

    traci.start(cmd, label="inspect")
    try:
        tls_ids = traci.trafficlight.getIDList()
        print(f"  Found {len(tls_ids)} traffic-light systems: {tls_ids}")

        tls_list: list[TLSInfo] = []
        for tid in tls_ids:
            logic = traci.trafficlight.getAllProgramLogics(tid)
            # Pick the active program
            current_prog = traci.trafficlight.getProgram(tid)
            active_logic = next(
                (l for l in logic if l.programID == current_prog), logic[0]
            )
            phases = [
                PhaseInfo(
                    index=i,
                    state=ph.state,
                    duration=ph.duration,
                    min_dur=getattr(ph, "minDur", ph.duration),
                    max_dur=getattr(ph, "maxDur", ph.duration),
                )
                for i, ph in enumerate(active_logic.phases)
            ]
            controlled_lanes = list(traci.trafficlight.getControlledLanes(tid))
            signal_lanes = signal_lanes_from_links(traci.trafficlight.getControlledLinks(tid))
            junction_id = ""
            # In SUMO the TLS id often equals the junction id
            all_junctions = traci.junction.getIDList()
            if tid in all_junctions:
                junction_id = tid

            tls_list.append(TLSInfo(
                tls_id=tid,
                program_id=active_logic.programID,
                type=str(active_logic.type),
                phases=phases,
                controlled_lanes=controlled_lanes,
                signal_lanes=signal_lanes,
                junction_id=junction_id,
            ))

        # Network-level
        all_edges = traci.edge.getIDList()
        all_lanes = traci.lane.getIDList()
        all_junctions = traci.junction.getIDList()

        # Bounding box from junctions
        positions = [traci.junction.getPosition(j) for j in all_junctions]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        boundary = [min(xs), min(ys), max(xs), max(ys)] if xs else []

        junctions_meta = []
        for jid in all_junctions:
            pos = traci.junction.getPosition(jid)
            junctions_meta.append({"id": jid, "x": pos[0], "y": pos[1]})

        net_meta = NetworkMetadata(
            net_file=str(cfg.get("net_file", "")),
            junction_count=len(all_junctions),
            edge_count=len(all_edges),
            lane_count=len(all_lanes),
            tls_count=len(tls_ids),
            boundary=boundary,
            junctions=junctions_meta,
        )

    finally:
        traci.close()

    return tls_list, net_meta


# ─────────────────────────────────────────────────────────────────────────────
# Demand inspection (static XML parse, no simulation needed)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_demand(cfg: dict[str, Any]) -> DemandMetadata:
    """Parse route files to collect demand metadata."""
    vtypes: set[str] = set()
    total_routes = 0

    for rf in cfg.get("route_files", []):
        rf_path = Path(rf)
        if not rf_path.exists():
            print(f"  Warning: route file not found: {rf_path}")
            continue
        try:
            tree = ET.parse(rf_path)
            root = tree.getroot()
            for vtype in root.iter("vType"):
                vtypes.add(vtype.get("id", ""))
            total_routes += sum(1 for _ in root.iter("route"))
            total_routes += sum(1 for _ in root.iter("flow"))
            total_routes += sum(1 for _ in root.iter("vehicle"))
        except ET.ParseError as e:
            print(f"  Warning: could not parse {rf_path}: {e}")

    return DemandMetadata(
        route_files=[str(rf) for rf in cfg.get("route_files", [])],
        additional_files=[str(af) for af in cfg.get("additional_files", [])],
        vehicle_types=sorted(vtypes),
        total_route_count=total_routes,
        simulation_begin=cfg["begin"],
        simulation_end=cfg["end"],
        step_length=cfg["step_length"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved → {path}")


def tls_to_dict(t: TLSInfo) -> dict:
    return {
        "tls_id": t.tls_id,
        "program_id": t.program_id,
        "type": t.type,
        "junction_id": t.junction_id,
        "phase_count": len(t.phases),
        "controlled_lane_count": len(t.controlled_lanes),
        "controlled_lanes": t.controlled_lanes,
        "signal_lanes": t.signal_lanes,
        "phases": [asdict(ph) for ph in t.phases],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect InTAS SUMO scenario and emit metadata JSON artifacts."
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=Path("data/raw/intas"),
        help="Path to InTAS scenario directory (default: data/raw/intas)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/scenario"),
        help="Directory for metadata JSON outputs",
    )
    parser.add_argument(
        "--patch-signal-lanes", type=Path, default=None, metavar="SUMOCFG",
        help="Only add the per-signal `signal_lanes` field to an existing "
             "<output-dir>/tls_metadata.json using this .sumocfg, then exit.",
    )
    args = parser.parse_args()

    if args.patch_signal_lanes is not None:
        n = add_signal_lanes_to_metadata(
            args.output_dir.resolve() / "tls_metadata.json", args.patch_signal_lanes.resolve()
        )
        print(f"Patched signal_lanes for {n} TLS.")
        return

    scenario_dir = args.scenario_dir.resolve()
    output_dir = args.output_dir.resolve()

    print(f"\n{'='*60}")
    print("CityMind AI — InTAS Scenario Inspector")
    print(f"{'='*60}")
    print(f"Scenario dir : {scenario_dir}")
    print(f"Output dir   : {output_dir}\n")

    if not scenario_dir.exists():
        print(f"ERROR: scenario_dir does not exist: {scenario_dir}")
        print("Clone InTAS first:\n  git clone https://github.com/silaslobo/InTAS data/raw/intas")
        sys.exit(1)

    # 1. Find and parse .sumocfg
    print("[1/4] Locating .sumocfg ...")
    cfg_path = find_sumocfg(scenario_dir)
    print(f"  Found: {cfg_path}")
    cfg = parse_sumocfg(cfg_path)
    cfg["cfg_path"] = str(cfg_path)

    print(f"  net_file      : {cfg['net_file']}")
    print(f"  route_files   : {[str(r) for r in cfg['route_files']]}")
    print(f"  additional    : {[str(a) for a in cfg['additional_files']]}")
    print(f"  begin-end     : {cfg['begin']} – {cfg['end']} s")

    # 2. Inspect demand (static)
    print("\n[2/4] Inspecting demand files ...")
    demand_meta = inspect_demand(cfg)
    print(f"  Total route/flow/vehicle entries: {demand_meta.total_route_count}")
    print(f"  Vehicle types: {demand_meta.vehicle_types}")

    # 3. Inspect TLS + network via TraCI
    print("\n[3/4] Starting SUMO (2-step headless) to inspect TLS + network ...")
    tls_list, net_meta = inspect_via_traci(cfg)
    print(f"\n  Junctions   : {net_meta.junction_count}")
    print(f"  Edges       : {net_meta.edge_count}")
    print(f"  Lanes       : {net_meta.lane_count}")
    print(f"  TLS systems : {net_meta.tls_count}")
    print(f"  Bounding box: {net_meta.boundary}")

    for t in tls_list:
        print(f"\n  TLS '{t.tls_id}'  program='{t.program_id}'  type={t.type}")
        print(f"    phases={len(t.phases)}  lanes={len(t.controlled_lanes)}")
        for ph in t.phases:
            print(f"    Phase {ph.index}: state='{ph.state}'  dur={ph.duration}s")

    # 4. Save JSON artifacts
    print(f"\n[4/4] Saving metadata artifacts to {output_dir} ...")
    save_json(
        [tls_to_dict(t) for t in tls_list],
        output_dir / "tls_metadata.json",
    )
    save_json(asdict(net_meta), output_dir / "network_metadata.json")
    save_json(asdict(demand_meta), output_dir / "demand_metadata.json")
    save_json(
        {
            "controlled_tls_ids": [t.tls_id for t in tls_list],
            "total_tls": len(tls_list),
            "cfg_path": str(cfg_path),
            "net_file": str(cfg.get("net_file", "")),
        },
        output_dir / "scenario_summary.json",
    )

    print(f"\n{'='*60}")
    print("Inspection complete.")
    print(f"  {len(tls_list)} TLS systems catalogued.")
    print(f"  Artifacts: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
