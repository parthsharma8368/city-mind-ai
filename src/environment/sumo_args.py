"""
sumo_args.py
============
ONE function builds the SUMO command line for every controller and every
environment (rl-core-004), so baselines and RL can be run on a byte-identical
simulation and the exact command can be recorded next to the results.

    from src.environment.sumo_args import build_sumo_command, command_signature
"""
from __future__ import annotations

from collections.abc import Sequence

# Options that never influence the simulated traffic (output sinks, GUI,
# bookkeeping) and are therefore ignored when comparing two commands.
_NON_SIMULATION_OPTIONS = frozenset({"--tripinfo-output", "--no-step-log", "--no-warnings"})


def build_sumo_command(
    sumo_binary: str,
    sumo_cfg: str,
    *,
    step_length: float = 1.0,
    seed: int = 42,
    begin: float | None = None,
    end: float | None = None,
    teleport_time: float | None = None,
    route_steps: int | None = None,
    additional_files: Sequence[str] | None = None,
    tripinfo_output: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    """Build the sumo argv. Optional arguments are emitted only when given, so
    the default output equals the historical per-class command lines.

    ``extra_args`` (e.g. a pre-built ``["--begin", "25200", "--end", "28800"]``)
    are appended last, exactly like the old ``sumo_extra_args`` behaviour.
    """
    cmd = [
        sumo_binary,
        "--configuration-file", str(sumo_cfg),
        "--step-length", str(step_length),
        "--no-warnings",
        "--no-step-log",
        "--seed", str(seed),
    ]
    if begin is not None:
        cmd += ["--begin", str(int(begin) if float(begin).is_integer() else begin)]
    if end is not None:
        cmd += ["--end", str(int(end) if float(end).is_integer() else end)]
    if teleport_time is not None:
        cmd += ["--time-to-teleport", str(teleport_time)]
    if route_steps is not None:
        cmd += ["--route-steps", str(route_steps)]
    if additional_files:
        cmd += ["--additional-files", ",".join(str(f) for f in additional_files)]
    if tripinfo_output:
        cmd += ["--tripinfo-output", str(tripinfo_output)]
    if extra_args:
        cmd += [str(a) for a in extra_args]
    return cmd


def get_option(cmd: Sequence[str], name: str) -> str | None:
    """Value of the LAST occurrence of ``name`` in ``cmd`` (SUMO: last wins)."""
    value = None
    for i, token in enumerate(cmd[:-1]):
        if token == name:
            value = cmd[i + 1]
    return value


def with_option(cmd: Sequence[str], name: str, value: str) -> list[str]:
    """Copy of ``cmd`` with ``name`` set to ``value`` (replaces existing)."""
    out: list[str] = []
    skip = False
    for i, token in enumerate(cmd):
        if skip:
            skip = False
            continue
        if token == name and i + 1 < len(cmd):
            skip = True
            continue
        out.append(token)
    return out + [name, str(value)]


def command_signature(cmd: Sequence[str]) -> dict[str, str | None]:
    """Order-independent map of simulation-relevant options -> value, for
    comparing two commands (``command_signature(a) == command_signature(b)``)."""
    sig: dict[str, str | None] = {}
    tokens = list(cmd)
    i = 1  # tokens[0] is the binary
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith("--") else None
            if tok not in _NON_SIMULATION_OPTIONS:
                sig[tok] = nxt
            i += 2 if nxt is not None else 1
        else:
            i += 1
    return sig
