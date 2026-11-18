"""Remote OpenAI-compatible client helpers (standard library only).

No torch/qwen_asr/vllm imports anywhere in this module. The full transcription
client arrives in N2; N1 only needs the unauthenticated health probe used by
``qasr doctor --server``.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .errors import UsageError

DEFAULT_TIMEOUT = 5.0
MAX_BODY_BYTES = 64 * 1024
MAX_REDIRECTS = 3
_SENSITIVE_QUERY = re.compile(r"token|key|secret|password|passwd|apikey|api_key", re.IGNORECASE)


def serve_token() -> Optional[str]:
    """Bearer token, read only from the environment (never persisted)."""
    token = os.environ.get("QASR_SERVE_TOKEN")
    return token or None


def redact_url(url: str) -> str:
    """Remove userinfo and sensitive query values before display/logging."""
    text = str(url or "")
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        return "<invalid-url>"
    host = parsed.hostname or ""
    if parsed.username is not None or parsed.password is not None:
        if parsed.port:
            host += f":{parsed.port}"
    else:
        host = parsed.netloc
    query = parsed.query
    if query:
        pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
        if any(_SENSITIVE_QUERY.search(key) for key, _ in pairs):
            query = urllib.parse.urlencode(
                [(key, "***" if _SENSITIVE_QUERY.search(key) else value)
                 for key, value in pairs]
            )
    return urllib.parse.urlunsplit(
        (parsed.scheme, host, parsed.path, query, parsed.fragment)
    )


def normalize_base_url(url: str) -> str:
    """Validate an http(s) base URL and strip the trailing slash."""
    text = str(url or "").strip()
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise UsageError(
            f"invalid server URL: {redact_url(text)!r}",
            hint="expected http://host:port or https://host",
        )
    return text.rstrip("/")


def _origin(url: str) -> tuple[str, str, Optional[int]]:
    parsed = urllib.parse.urlsplit(url)
    default_port = 443 if parsed.scheme == "https" else 80
    return (parsed.scheme.lower(), (parsed.hostname or "").lower(),
            parsed.port or default_port)


class SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow at most 3 redirects, never sending credentials cross-origin."""

    max_repeats = MAX_REDIRECTS
    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _origin(req.full_url) != _origin(newurl):
            raise urllib.error.HTTPError(
                newurl,
                code,
                "refusing cross-origin redirect to protect credentials",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_request(request: urllib.request.Request, *, timeout: float):
    """Open a request with the credential-safe redirect policy."""
    opener = urllib.request.build_opener(SameOriginRedirectHandler())
    return opener.open(request, timeout=timeout)


def _headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "qasr-cli"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _model_ids(payload: Any) -> List[str]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids: List[str] = []
    for entry in data:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            ids.append(entry["id"])
    return ids


def probe_server(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """GET ``/v1/models`` and classify the outcome for ``doctor``.

    The return value never contains the Authorization header or token.
    """
    base = normalize_base_url(url)
    token = serve_token()
    started = time.perf_counter()
    result: Dict[str, Any] = {
        "url": redact_url(base),
        "token_set": token is not None,
        "reachable": False,
        "http_status": None,
        "latency_ms": None,
        "models": [],
        "auth_rejected": False,
        "timed_out": False,
        "error": None,
    }
    request = urllib.request.Request(base + "/v1/models", headers=_headers(token))
    try:
        with open_request(request, timeout=timeout) as response:
            raw = response.read(MAX_BODY_BYTES)
            status = int(getattr(response, "status", 200) or 200)
            result.update(reachable=True, http_status=status,
                          latency_ms=int((time.perf_counter() - started) * 1000))
            try:
                result["models"] = _model_ids(json.loads(raw.decode("utf-8")))
            except (ValueError, UnicodeDecodeError):
                result["error"] = "response was not valid JSON"
    except urllib.error.HTTPError as exc:
        reason = str(exc.reason or "").strip()
        detail = f"HTTP {exc.code}" + (f": {reason}" if reason else "")
        result.update(
            reachable=True,
            http_status=int(exc.code),
            latency_ms=int((time.perf_counter() - started) * 1000),
            auth_rejected=int(exc.code) in (401, 403),
            error=detail,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        timed_out = isinstance(exc, TimeoutError) or "timed out" in str(reason).casefold()
        result.update(
            latency_ms=int((time.perf_counter() - started) * 1000),
            timed_out=timed_out,
            error="request timed out" if timed_out else f"connection failed: {reason}",
        )
    return result
