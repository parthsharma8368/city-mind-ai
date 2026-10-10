"""
security.py — Access control for state-changing (process-spawning) endpoints.

Policy (fails closed):
  * CITYMIND_API_KEY set (non-blank): every request must send a matching
    ``X-API-Key`` header (constant-time compare). A blank / whitespace value is
    treated as *unset*, never as "configured but open".
  * CITYMIND_API_KEY unset: only loopback clients (127.0.0.1 / ::1) are allowed,
    so ``--host 0.0.0.0`` does not expose ``subprocess.Popen`` to the network.
    ``CITYMIND_ALLOW_INSECURE=1`` is the explicit opt-out.
  * CSRF: a request carrying an ``Origin`` outside the CORS allow-list, or
    ``Sec-Fetch-Site: cross-site`` with no Origin, or a "simple" form/text
    content type, is rejected (browsers always attach Origin to cross-origin POSTs).
  * Behind a reverse proxy every client looks local; set a key in that case.
"""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import Header, HTTPException, Request, WebSocket

logger = logging.getLogger("citymind.security")

DEFAULT_ORIGINS = (
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"
)
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_SIMPLE_CONTENT_TYPES = ("application/x-www-form-urlencoded", "multipart/form-data", "text/plain")


def parse_origins(raw: str | None) -> list[str]:
    """Parse a comma-separated origin list; a wildcard is never accepted."""
    origins: list[str] = []
    for item in (raw if raw is not None else DEFAULT_ORIGINS).split(","):
        item = item.strip().rstrip("/")
        if not item:
            continue
        if "*" in item or item == "null":
            logger.error("Ignoring unsafe CITYMIND_CORS_ORIGINS entry %r", item)
            continue
        origins.append(item)
    return origins


def allowed_origins() -> list[str]:
    return parse_origins(os.environ.get("CITYMIND_CORS_ORIGINS"))


def configured_key() -> str | None:
    """The API key, or None when unset/blank (blank is NOT 'open by configuration')."""
    key = os.environ.get("CITYMIND_API_KEY")
    if key is not None and not key.strip():
        logger.error("CITYMIND_API_KEY is set but blank - treating as unset (loopback-only access).")
        return None
    return key or None


def _insecure_allowed() -> bool:
    return os.environ.get("CITYMIND_ALLOW_INSECURE") == "1"


def is_loopback(host: str | None) -> bool:
    return host in _LOOPBACK_HOSTS


def key_matches(provided: str | None) -> bool:
    expected = configured_key()
    if expected is None:
        return False
    return hmac.compare_digest((provided or "").encode(), expected.encode())


def origin_ok(origin: str | None) -> bool:
    """A missing Origin (curl, scripts, same-origin GET) is fine; a present one must be allow-listed."""
    return origin is None or origin.rstrip("/") in allowed_origins()


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    origin = request.headers.get("origin")
    if not origin_ok(origin):
        raise HTTPException(status_code=403, detail="Origin not allowed.")
    # The dev UI (http://localhost:5173) calls the API at http://127.0.0.1:8000: browsers label that
    # "cross-site" even though the Origin is allow-listed, so only reject cross-site requests that
    # carry no allow-listed Origin to vouch for them.
    if origin is None and request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(status_code=403, detail="Cross-site requests are not allowed.")
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type in _SIMPLE_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Use application/json or no body.")

    if configured_key() is not None:
        if not key_matches(x_api_key):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
        return
    host = request.client.host if request.client else None
    if not (is_loopback(host) or _insecure_allowed()):
        raise HTTPException(
            status_code=403,
            detail="CITYMIND_API_KEY is not configured; this endpoint is loopback-only.",
        )


def websocket_allowed(websocket: WebSocket) -> bool:
    """Handshake check: Origin allow-list, then key (header or subprotocol) or loopback."""
    if not origin_ok(websocket.headers.get("origin")):
        return False
    if configured_key() is not None:
        provided = websocket.headers.get("x-api-key")
        if provided is None:
            for proto in (websocket.headers.get("sec-websocket-protocol") or "").split(","):
                proto = proto.strip()
                if proto.startswith("citymind-key."):
                    provided = proto[len("citymind-key."):]
        return key_matches(provided)
    host = websocket.client.host if websocket.client else None
    return is_loopback(host) or _insecure_allowed()
