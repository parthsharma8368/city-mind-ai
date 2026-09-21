"""
generate_static_programs.py
============================
Builds a real static (fixed-time) signal-program additional file for every
TLS in the InTAS network, for use as an honest `fixed_time` baseline.

Why this exists: the project's `FixedTimeController` lets SUMO's own InTAS
programs run unmodified, but 76 of the 98 `tlLogic` entries in
ingolstadt.net.xml are `type="actuated"` (vehicle-actuated, gap-based green
extension between minDur/maxDur) — measured directly on 2026-09-20/21 by
comparing observed phase durations against programmed ones (e.g. TLS
1863241632's 35s/6s/34s greens ran for 7-8s/7s/10s at the 07:00 peak).
Forcing `traci.trafficlight.Logic.type = 0` at runtime does NOT change this
— verified directly: phase durations were byte-identical between a
"forced-static" run and an unmodified run. SUMO only honors program type
when it's part of a loaded program definition, so this script writes a
genuinely separate, distinct-programID static program per TLS with only the
`duration` attribute kept (no minDur/maxDur), which run_static_fixed_time
selects via traci.trafficlight.setProgram() before the episode starts.

Usage:
  python scripts/generate_static_programs.py \
    --net-file data/raw/intas/scenario/ingolstadt.net.xml \
    --output artifacts/scenario/static_programs.add.xml
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

STATIC_PROGRAM_ID = "static_fixed"


def build_static_additional_file(net_file: Path, output: Path) -> int:
    tree = ET.parse(net_file)
    root = tree.getroot()

    out_root = ET.Element("additional")
    n_tls = 0
    for tl in root.findall("tlLogic"):
        tls_id = tl.get("id")
        # A joined TLS (multiple programs sharing one id) has one tlLogic
        # entry per program in the source net file; only convert the
        # program actually running (programID "0", InTAS's convention —
        # verified: every tlLogic in ingolstadt.net.xml uses programID "0").
        if tl.get("programID") != "0":
            continue
        new_tl = ET.SubElement(out_root, "tlLogic")
        new_tl.set("id", tls_id)
        new_tl.set("type", "static")
        new_tl.set("programID", STATIC_PROGRAM_ID)
        new_tl.set("offset", tl.get("offset", "0"))
        for phase in tl.findall("phase"):
            new_phase = ET.SubElement(new_tl, "phase")
            new_phase.set("duration", phase.get("duration"))
            new_phase.set("state", phase.get("state"))
            # Deliberately drop minDur/maxDur — that's what makes this
            # program static instead of actuated.
        n_tls += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(out_root, space="    ")
    ET.ElementTree(out_root).write(output, encoding="UTF-8", xml_declaration=True)
    return n_tls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net-file", default="data/raw/intas/scenario/ingolstadt.net.xml")
    p.add_argument("--output", default="artifacts/scenario/static_programs.add.xml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = build_static_additional_file(Path(args.net_file), Path(args.output))
    print(f"Wrote {n} static tlLogic programs (programID={STATIC_PROGRAM_ID!r}) to {args.output}")


if __name__ == "__main__":
    main()
