"""A small scriptable in-memory traci used by env/controller tests (no SUMO).

It simulates only what the project code touches: time, a couple of lanes/vehicles
and traffic lights whose displayed state is whatever was last written (starting
from a native 'program' that the tests can advance), plus call recording.
"""
from __future__ import annotations

from types import SimpleNamespace

import traci.exceptions as _texc  # pure-python package, importable without SUMO

TraCIException = _texc.TraCIException
FatalTraCIError = _texc.FatalTraCIError


class FakeTraci:
    def __init__(self, tls: dict[str, list[str]], *, end_time: float = 1e9, begin: float = 0.0):
        """tls: {tls_id: [native phase state strings]} (native program cycles every 2 s/phase)."""
        self.tls_native = tls
        self.time = begin
        self.begin = begin
        self.end_time = end_time
        self.started: list[list[str]] = []
        self.closed = 0
        self.steps = 0
        self.written: list[tuple[float, str, str]] = []     # (time, tls, state)
        self.programs_set: list[tuple[str, str]] = []
        self.owned: set[str] = set()
        self.held_state: dict[str, str] = {}
        self.min_expected = 5
        self.departed_per_step = 1
        self.fail_on: dict[str, BaseException] = {}
        self.vehicles = {"v1": 4.0, "v2": 0.0}                # id -> speed
        self.halting = {"l0": 2, "l1": 0}
        self.exceptions = SimpleNamespace(TraCIException=TraCIException, FatalTraCIError=FatalTraCIError)
        outer = self

        class _Sim:
            @staticmethod
            def getTime():
                return outer.time

            @staticmethod
            def getMinExpectedNumber():
                if "getMinExpectedNumber" in outer.fail_on:
                    raise outer.fail_on["getMinExpectedNumber"]
                return outer.min_expected

            @staticmethod
            def getDepartedNumber():
                return outer.departed_per_step

            @staticmethod
            def getArrivedNumber():
                return 0

            @staticmethod
            def getStartingTeleportNumber():
                return 0

            @staticmethod
            def getPendingVehicles():
                return []

        class _TL:
            @staticmethod
            def getIDList():
                return list(outer.tls_native)

            @staticmethod
            def getPhase(tid):
                return int((outer.time - outer.begin) // 2) % len(outer.tls_native[tid])

            @staticmethod
            def getRedYellowGreenState(tid):
                if tid in outer.owned:
                    return outer.held_state[tid]
                return outer.tls_native[tid][_TL.getPhase(tid)]

            @staticmethod
            def setRedYellowGreenState(tid, state):
                if "setRedYellowGreenState" in outer.fail_on:
                    raise outer.fail_on["setRedYellowGreenState"]
                outer.owned.add(tid)
                outer.held_state[tid] = state
                outer.written.append((outer.time, tid, state))

            @staticmethod
            def setProgram(tid, prog):
                if tid in outer.fail_on.get("setProgram_missing", ()):
                    raise TraCIException("no program")
                outer.programs_set.append((tid, prog))

            @staticmethod
            def getControlledLanes(tid):
                return ["l0", "l1"]

        class _Lane:
            @staticmethod
            def getIDList():
                return list(outer.halting)

            @staticmethod
            def getLastStepHaltingNumber(lid):
                return outer.halting[lid]

            @staticmethod
            def getLastStepVehicleNumber(lid):
                return 1

            @staticmethod
            def getLastStepMeanSpeed(lid):
                return 13.89          # empty lanes report free-flow

            @staticmethod
            def getLastStepOccupancy(lid):
                return 10.0

            @staticmethod
            def getLinks(lid):
                return []

            @staticmethod
            def getShape(lid):
                return [(0.0, 0.0), (1.0, 1.0)]

        class _Veh:
            @staticmethod
            def getIDList():
                return list(outer.vehicles)

            @staticmethod
            def getSpeed(vid):
                return outer.vehicles[vid]

            @staticmethod
            def getWaitingTime(vid):
                return 1.0

            @staticmethod
            def getLaneID(vid):
                return "l0"

            @staticmethod
            def getPosition(vid):
                return (1.0, 2.0)

        class _Junction:
            @staticmethod
            def getPosition(jid):
                return (float(len(jid)), 0.0)

        self.simulation, self.trafficlight = _Sim, _TL
        self.lane, self.vehicle, self.junction = _Lane, _Veh, _Junction

    # -- top level API ------------------------------------------------------
    def start(self, cmd, label=None):
        self.started.append(list(cmd))
        self.time = float(cmd[cmd.index("--begin") + 1]) if "--begin" in cmd else self.begin
        self.closed_flag = False
        self.owned.clear()

    def close(self):
        self.closed += 1

    def simulationStep(self):
        if "simulationStep" in self.fail_on:
            raise self.fail_on["simulationStep"]
        self.time += 1.0
        self.steps += 1
        if self.time >= self.end_time:
            self.min_expected = 0


def install(monkeypatch, fake: FakeTraci) -> FakeTraci:
    """Patch ``traci`` in every project module that imported it."""
    import importlib

    for mod in (
        "src.environment.sumo_session", "src.environment.safety", "src.environment.metrics",
        "src.environment.observation", "src.environment.reward", "src.environment.local_observation",
        "src.environment.local_reward", "src.environment.centralized_sumo_env",
        "src.environment.decentralized_sumo_env", "src.controllers.base",
        "src.controllers.sumo_actuated", "src.controllers.fixed_time", "src.controllers.max_pressure",
    ):
        m = importlib.import_module(mod)
        if hasattr(m, "traci"):
            monkeypatch.setattr(m, "traci", fake)
        if hasattr(m, "USING_LIBSUMO"):
            monkeypatch.setattr(m, "USING_LIBSUMO", False)
    return fake
