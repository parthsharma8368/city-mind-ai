"""util.py — Small shared helpers for the API routers."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("citymind.api.util")


def sanitize_nan(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/inf) with None so responses stay valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_nan(v) for v in obj]
    return obj


def read_json_file(path: Path, retries: int = 3, delay: float = 0.05) -> Any | None:
    """Read a JSON file's bytes once per attempt; retry on a decode error (file may be mid-write).

    Returns None (and logs) if the file never parses or cannot be read.
    """
    for attempt in range(retries):
        try:
            return json.loads(path.read_bytes())
        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(delay)
        except OSError as exc:
            logger.warning("could not read %s: %s", path.name, exc)
            return None
    logger.warning("skipping unparseable JSON (possibly mid-write): %s", path.name)
    return None
