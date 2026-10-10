"""Tiny synthetic SUMO scenario (net + routes + tripinfo) for the evaluation-pipeline tests."""
from __future__ import annotations

from pathlib import Path

NET = """<net>
  <edge id=":j" function="internal"><lane id=":j_0" index="0" speed="10.00" length="10.00" shape="0,0 1,1"/></edge>
  <edge id="E0" from="a" to="b"><lane id="E0_0" index="0" speed="10.00" length="100.00" shape="0,0 100,0"/></edge>
  <edge id="E1" from="b" to="c"><lane id="E1_0" index="0" speed="10.00" length="200.00" shape="100,0 300,0"/></edge>
  <edge id="E2" from="c" to="d"><lane id="E2_0" index="0" speed="20.00" length="100.00" shape="300,0 400,0"/></edge>
  <connection from="E0" to="E1" fromLane="0" toLane="0" via=":j_0"/>
  <connection from="E1" to="E2" fromLane="0" toLane="0"/>
</net>
"""

ROUTES = """<routes>
  <vType id="car"/>
  <vehicle id="carIn1" type="car" depart="0.00" departPos="0.00" arrivalPos="100.00">
    <routeDistribution last="0"><route cost="1" probability="1.0" edges="E0 E1"/></routeDistribution>
  </vehicle>
  <vehicle id="carIn2" type="car" depart="100.00" departPos="0.00" arrivalPos="50.00">
    <routeDistribution last="0"><route cost="1" probability="1.0" edges="E0 E1 E2"/></routeDistribution>
  </vehicle>
  <vehicle id="hz1" type="car" depart="50.00" departPos="5.00" arrivalPos="5.00">
    <routeDistribution last="0"><route cost="1" probability="1.0" edges="E1"/></routeDistribution>
  </vehicle>
  <vehicle id="late" type="car" depart="900.00" departPos="0.00" arrivalPos="50.00">
    <routeDistribution last="0"><route cost="1" probability="1.0" edges="E0 E1"/></routeDistribution>
  </vehicle>
</routes>
"""

FLOWS = """<routes>
  <flow id="bus1" type="bus" begin="0" end="300" period="100" line="1"><route edges="E0 E1"/></flow>
</routes>
"""

SUMOCFG = """<configuration>
  <input>
    <net-file value="net.net.xml"/>
    <route-files value="cars.rou.xml,buses.flow.xml"/>
  </input>
  <time><begin value="0"/><end value="200"/><step-length value="1"/></time>
  <processing><max-depart-delay value="300"/></processing>
</configuration>
"""

BEGIN, END = 0.0, 200.0


def tripinfo_row(vid, depart, arrival, length, delay=0.0, loss=20.0, wait=5.0, vtype="car"):
    return (f'    <tripinfo id="{vid}" depart="{depart:.2f}" departDelay="{delay:.2f}" arrival="{arrival:.2f}" '
            f'duration="{arrival - depart:.2f}" routeLength="{length:.2f}" waitingTime="{wait:.2f}" '
            f'timeLoss="{loss:.2f}" rerouteNo="0" vType="{vtype}"/>\n')


def write_tripinfo(path: Path, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<?xml version="1.0"?>\n<tripinfos>\n' + "".join(rows) + "</tripinfos>\n")
    return path


def write_scenario(root: Path) -> Path:
    """Write net/routes/flows/sumocfg under root; return the sumocfg path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "net.net.xml").write_text(NET)
    (root / "cars.rou.xml").write_text(ROUTES)
    (root / "buses.flow.xml").write_text(FLOWS)
    cfg = root / "scn.sumocfg"
    cfg.write_text(SUMOCFG)
    return cfg
