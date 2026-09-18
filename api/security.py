"""
security.py — Optional shared-secret auth for state-changing endpoints.

Local research use (uvicorn bound to 127.0.0.1) needs no configuration: with
CITYMIND_API_KEY unset, `require_api_key` is a no-op. The moment the API is
exposed beyond localhost (e.g. `--host 0.0.0.0`), set CITYMIND_API_KEY and
every process-spawning endpoint starts requiring it via the `X-API-Key`
header — this is what stands between an unauthenticated network client and
`subprocess.Popen`.
"""
from __future__ import annotations

import logging
import os

from fastapi import Header, HTTPException

logger = logging.getLogger("citymind.security")
_warned = False


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("CITYMIND_API_KEY")
    if not expected:
        global _warned
        if not _warned:
            logger.warning(
                "CITYMIND_API_KEY is not set — simulation-control endpoints are "
                "unauthenticated. Fine for 127.0.0.1-only use; set this env var "
                "before binding uvicorn to any non-localhost host."
            )
            _warned = True
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
