"""
provenance.py
=============
Run provenance for evaluation results, and the config fingerprint that keeps
different evaluation configurations from silently overwriting each other.

Every result JSON records: git commit + dirty flag, sumocfg path + sha256,
SUMO version, begin/end/step length, seed, model path + sha256, the command
line, a UTC timestamp. ``config_hash`` is a short hash of everything that
changes the *meaning* of a result (sumocfg content, window, step length,
demand scale, protocol versions, SUMO version); it is part of the overwrite
guard in :mod:`src.evaluation.evaluate`.

The helpers are read-only: they run ``git rev-parse`` / ``git status`` /
``sumo --version`` and never modify the repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The one window the published/API-served artifacts use (07:00-08:00 peak).
CANONICAL_BEGIN = 25200
CANONICAL_DURATION = 3600


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str | None:
    """sha256 of a file's bytes; None if the path does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@lru_cache(maxsize=1)
def git_info() -> dict[str, Any]:
    """Commit SHA and whether tracked files differ from it (read-only git calls)."""
    commit = _run(["git", "rev-parse", "HEAD"], PROJECT_ROOT)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], PROJECT_ROOT)
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


@lru_cache(maxsize=1)
def sumo_version(binary: str = "sumo") -> str | None:
    out = _run([binary, "--version"])
    return out.splitlines()[0].strip() if out else None


def read_sumocfg(path: str | Path) -> dict[str, Any]:
    """Key facts of a .sumocfg: input files (absolute), max-depart-delay, step-length, begin/end."""
    p = Path(path)
    root = ET.parse(p).getroot()

    def _val(xpath: str) -> str | None:
        el = root.find(xpath)
        return el.get("value") if el is not None else None

    def _files(xpath: str) -> list[Path]:
        raw = _val(xpath)
        if not raw:
            return []
        return [(p.parent / f.strip()).resolve() for f in raw.split(",") if f.strip()]

    net = _files(".//net-file")
    return {
        "path": str(p),
        "sha256": sha256_file(p),
        "net_file": net[0] if net else None,
        "route_files": _files(".//route-files"),
        "additional_files": _files(".//additional-files"),
        "max_depart_delay": _val(".//max-depart-delay"),
        "step_length": float(_val(".//step-length") or 1.0),
        "begin": _val(".//begin"),
        "end": _val(".//end"),
    }


def is_canonical_window(begin_time: int, episode_duration: int) -> bool:
    return int(begin_time) == CANONICAL_BEGIN and int(episode_duration) == CANONICAL_DURATION


def config_hash(
    sumo_cfg: str | Path,
    begin_time: int,
    episode_duration: int,
    demand_id: str,
    step_length: float = 1.0,
    protocol: str | None = None,
) -> str:
    """Short hash of every setting that changes what a result means."""
    payload = {
        "sumocfg_sha256": sha256_file(sumo_cfg),
        "begin": int(begin_time),
        "duration": int(episode_duration),
        "demand_id": demand_id,
        "step_length": float(step_length),
        "sumo_version": sumo_version(),
        "protocol": protocol,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def run_tag(begin_time: int, episode_duration: int, chash: str) -> str:
    """Directory/file tag for a configuration, e.g. ``b25200_d3600_1a2b3c4d5e6f``."""
    return f"b{int(begin_time)}_d{int(episode_duration)}_{chash}"


def build_provenance(
    *,
    sumo_cfg: str | Path,
    seed: int,
    begin_time: int,
    episode_duration: int,
    demand_id: str,
    model_path: str | Path | None = None,
    protocol: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The provenance block stored in every result JSON."""
    cfg_info = read_sumocfg(sumo_cfg)
    step = cfg_info["step_length"]
    block: dict[str, Any] = {
        **git_info(),
        "sumocfg_path": str(sumo_cfg),
        "sumocfg_sha256": cfg_info["sha256"],
        "max_depart_delay": cfg_info["max_depart_delay"],
        "sumo_version": sumo_version(),
        "begin_time": int(begin_time),
        "end_time": int(begin_time) + int(episode_duration),
        "episode_duration": int(episode_duration),
        "step_length": step,
        "seed": int(seed),
        "demand_id": demand_id,
        "model_path": str(model_path) if model_path else None,
        "model_sha256": sha256_file(model_path) if model_path else None,
        "command_line": " ".join([os.path.basename(sys.executable)] + sys.argv),
        "python": re.sub(r"\s.*", "", sys.version),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config_hash(sumo_cfg, begin_time, episode_duration, demand_id, step, protocol),
    }
    if extra:
        block.update(extra)
    return block


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write via a temp file in the same directory + os.replace (readers never see half a file)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, p)
