"""Remote OpenAI-compatible client helpers (standard library only).

No torch/qwen_asr/vllm imports anywhere in this module. The full transcription
client arrives in N2; N1 only needs the unauthenticated health probe used by
``qasr doctor --server``.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .backends import TranscribeResult
from .catalog import spec_for
from .errors import CapabilityError, InputError, QasrError, RuntimeFailure, UsageError
from .languages import iso_for

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
        host = parsed.hostname or ""
        port = parsed.port  # raises ValueError for an invalid port
        if parsed.username is not None or parsed.password is not None:
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            if port is not None:
                host += f":{port}"
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
    except ValueError:
        return "<invalid-url>"


def normalize_base_url(url: str) -> str:
    """Validate an http(s) base URL, strip userinfo and the trailing slash.

    Userinfo cannot be sent by ``urllib`` ``Request`` (it would end up in the
    Host header); credentials travel only via ``QASR_SERVE_TOKEN``.
    """
    text = str(url or "").strip()
    try:
        parsed = urllib.parse.urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise UsageError(
            f"invalid server URL: {redact_url(text)!r}",
            hint="expected http://host:port or https://host",
        ) from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UsageError(
            f"invalid server URL: {redact_url(text)!r}",
            hint="expected http://host:port or https://host",
        )
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit((parsed.scheme, netloc, path, "", ""))


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


# ---------------------------------------------------------------------------
# Remote transcription client (N2)
# ---------------------------------------------------------------------------

RETRY_BASE_DELAY = 0.5
RETRY_MAX_DELAY = 5.0
ERROR_BODY_LIMIT = 2048
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_LANG_PREFIX = re.compile(
    r"^\s*language\s+([^<\r\n]+?)\s*(?:\r?\n|$)", re.IGNORECASE
)
_LANGUAGE_ERROR = re.compile(r"language|lang|unsupported", re.IGNORECASE)

# The transcription endpoint is missing on some servers; remember the probe
# result for the life of this process (per base URL).
_CHAT_ONLY_SERVERS: set[str] = set()


def _clean_language(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    return None if not text or text.casefold() == "none" else text


def parse_asr_output(raw: Any) -> Tuple[Optional[str], str]:
    """Split a Qwen ASR output into ``(language, text)``.

    Handles both ``language X<asr_text>text`` and the fuller form where
    metadata lines follow the language before ``<asr_text>``. The
    ``<asr_text>`` marker itself never leaks into the transcript.
    """
    if raw is None:
        return None, ""
    text = str(raw)
    language: Optional[str] = None
    if "<asr_text>" in text:
        prefix, text = text.split("<asr_text>", 1)
        match = re.search(r"language\s+([^\r\n<]+)", prefix, re.IGNORECASE)
        if match:
            language = _clean_language(match.group(1))
    else:
        match = _LANG_PREFIX.match(text)
        if match:
            language = _clean_language(match.group(1))
            text = text[match.end():]
    text = text.replace("<asr_text>", "")
    return language, text.strip()


def encode_multipart(
    fields: Dict[str, Any],
    *,
    file_bytes: bytes,
    filename: str,
    mime: str,
) -> Tuple[bytes, str]:
    """Encode ``fields`` plus one file part without third-party libraries."""
    boundary = "----qasr" + uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        if value is None:
            continue
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    chunks.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode()
        + file_bytes
        + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class _HTTPFailure(Exception):
    """Internal transport failure before it is mapped to a public error."""

    def __init__(
        self,
        status: Optional[int],
        body: str = "",
        *,
        retryable: bool = False,
        retry_after: Optional[float] = None,
        reason: str = "",
    ) -> None:
        super().__init__(reason or f"HTTP {status}")
        self.status = status
        self.body = body
        self.retryable = retryable
        self.retry_after = retry_after
        self.reason = reason


def _retry_after(headers: Any) -> Optional[float]:
    if not headers:
        return None
    try:
        value = float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= 10 else None


def _body_snippet(raw: bytes, limit: int = ERROR_BODY_LIMIT) -> str:
    text = raw[:limit].decode("utf-8", errors="replace")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _public_error(failure: _HTTPFailure) -> QasrError:
    status = failure.status
    details: Dict[str, Any] = {}
    if status is not None:
        details["http_status"] = status
    if failure.retry_after is not None:
        details["retry_after"] = failure.retry_after
    suffix = f": {failure.body}" if failure.body else ""
    if status in (401, 403):
        return InputError(
            f"server rejected credentials (HTTP {status})",
            hint="check QASR_SERVE_TOKEN",
            details=details,
        )
    if status is not None and 300 <= status < 400:
        return RuntimeFailure(
            f"server returned an unsafe redirect (HTTP {status})",
            details=details,
        )
    if status in (408, 429):
        error = RuntimeFailure(
            f"remote request failed (HTTP {status}){suffix}", details=details
        )
        error.retryable = True
        return error
    if status is not None and 400 <= status < 500:
        return InputError(
            f"server rejected the request (HTTP {status}){suffix}",
            details=details,
        )
    error = RuntimeFailure(
        f"remote request failed ({failure.reason or f'HTTP {status}'}){suffix}",
        details=details,
    )
    error.retryable = True
    return error


@dataclass
class RemoteOptions:
    base_url: str
    model: Optional[str] = None
    timeout: float = 300.0
    retries: int = 2
    log: Optional[Callable[[str], None]] = None


class RemoteBackend:
    """OpenAI-compatible transcription client implementing the Backend protocol."""

    name = "remote"

    def __init__(self, options: RemoteOptions) -> None:
        self.base_url = normalize_base_url(options.base_url)
        self.model = options.model
        self.timeout = float(options.timeout)
        self.retries = max(0, int(options.retries))
        self.log = options.log
        self.token = serve_token()
        self._model_id: Optional[str] = None
        self.notes: List[str] = []

    def prepare(self) -> None:
        """Resolve the served model once, before concurrent workers start."""
        try:
            self._resolve_model_id()
        except _HTTPFailure as failure:
            raise _public_error(failure) from None

    # -- transport -------------------------------------------------------
    def _warn(self, message: str) -> None:
        if self.log is not None:
            self.log(f"warning: {message}")

    def _sleep(self, attempt: int, retry_after: Optional[float]) -> None:
        delay = retry_after if retry_after is not None else min(
            RETRY_MAX_DELAY, RETRY_BASE_DELAY * (2 ** attempt)
        )
        time.sleep(delay)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        data: Optional[bytes] = None,
        content_type: Optional[str] = None,
    ) -> Any:
        headers = _headers(self.token)
        if content_type:
            headers["Content-Type"] = content_type
        attempt = 0
        while True:
            request = urllib.request.Request(
                self.base_url + path, data=data, headers=headers, method=method
            )
            try:
                with open_request(request, timeout=self.timeout) as response:
                    raw = response.read()
                try:
                    return json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    raise _HTTPFailure(
                        None, _body_snippet(raw), reason="response was not valid JSON"
                    ) from None
            except urllib.error.HTTPError as exc:
                body = _body_snippet(exc.read() or b"")
                status = int(exc.code)
                retryable = status in (408, 429) or 500 <= status <= 599
                retry_after = _retry_after(exc.headers)
                if retryable and attempt < self.retries:
                    self._sleep(attempt, retry_after)
                    attempt += 1
                    continue
                raise _HTTPFailure(
                    status, body, retryable=retryable, retry_after=retry_after
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                if attempt < self.retries:
                    self._sleep(attempt, None)
                    attempt += 1
                    continue
                raise _HTTPFailure(
                    None, "", retryable=True, reason=str(reason)
                ) from None

    def _resolve_model_id(self) -> str:
        """Map the requested qasr model onto the server's served model id.

        Servers commonly rename models (``--served-model-name``), so the
        requested alias/repo is only a hint; ``/v1/models`` is authoritative.
        """
        if self._model_id:
            return self._model_id
        payload = self._request_json("GET", "/v1/models")
        models = _model_ids(payload)
        if not models:
            raise InputError(
                "server listed no models",
                hint="pass --model with the model id served by the server",
            )
        lookup = {model.casefold(): model for model in models}
        requested = str(self.model or "").strip()
        if not requested:
            self._model_id = models[0]
            if len(models) > 1:
                note = f"server serves {len(models)} models; using {models[0]!r}"
                self.notes.append(note)
                self._warn(note)
            return self._model_id

        spec = spec_for(requested)
        candidates = [requested, Path(requested).name]
        if spec is not None:
            candidates.extend([spec.repo, spec.alias, spec.repo.rsplit("/", 1)[-1]])
        for candidate in candidates:
            matched = lookup.get(candidate.casefold())
            if matched:
                self._model_id = matched
                return self._model_id
        if len(models) == 1 and spec is not None:
            self._model_id = models[0]
            note = (
                f"server serves {models[0]!r}; requested {requested!r} was not "
                "listed, using the only served model"
            )
            self.notes.append(note)
            self._warn(note)
            return self._model_id
        raise InputError(
            f"server does not serve model {requested!r}",
            hint="served models: " + ", ".join(models) + "; pass --model",
        )

    # -- protocol --------------------------------------------------------
    def capabilities(self) -> Any:
        from .backends import Capabilities

        return Capabilities(name=self.name, batch=False, timestamps=False, align=False)

    def ensure_loaded(self, *, need_align: bool = False) -> None:
        if need_align:
            raise CapabilityError(
                "the remote backend does not return timestamps",
                hint="run timestamps locally with a forced aligner",
            )

    def close(self) -> None:
        return None

    def _audio(self, job: Any) -> Tuple[bytes, str, str]:
        path = job.item.path
        if path is None or not Path(path).is_file():
            raise InputError(f"audio file not found: {path}")
        raw = Path(path).read_bytes()
        if not raw:
            raise InputError(f"audio file is empty: {path}")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise InputError(
                f"audio file exceeds the 25 MB server limit: {path}",
                hint="use --max-chunk-seconds (long-audio chunking arrives in N4)",
            )
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return raw, Path(path).name, mime

    def _transcriptions(
        self, raw: bytes, filename: str, mime: str, language: Optional[str]
    ) -> Any:
        model = self._resolve_model_id()
        body, content_type = encode_multipart(
            {"model": model, "language": language},
            file_bytes=raw,
            filename=filename,
            mime=mime,
        )
        return self._request_json(
            "POST", "/v1/audio/transcriptions", data=body, content_type=content_type
        )

    def _chat(
        self, raw: bytes, mime: str, context: str, language: Optional[str]
    ) -> Any:
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:{mime};base64,"
                                   + base64.b64encode(raw).decode("ascii")
                        },
                    }
                ],
            }
        )
        payload = {"model": self._resolve_model_id(), "messages": messages}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._request_json(
            "POST",
            "/v1/chat/completions",
            data=body,
            content_type="application/json",
        )

    def transcribe(self, jobs: Sequence[Any], opts: Any) -> List[TranscribeResult]:
        if opts.timestamps:
            raise CapabilityError(
                "the remote backend does not return timestamps",
                hint="run timestamps locally with a forced aligner",
            )
        return [self._one(job) for job in jobs]

    def _one(self, job: Any) -> TranscribeResult:
        raw_audio, filename, mime = self._audio(job)
        language = iso_for(job.language) if job.language else None
        warnings: List[str] = list(self.notes)
        used_chat = bool(job.context) or self.base_url in _CHAT_ONLY_SERVERS
        try:
            if job.context:
                payload = self._chat(raw_audio, mime, job.context, language)
            elif used_chat:
                payload = self._chat(raw_audio, mime, "", language)
            else:
                try:
                    payload = self._transcriptions(raw_audio, filename, mime, language)
                except _HTTPFailure as failure:
                    if failure.status in (404, 405):
                        _CHAT_ONLY_SERVERS.add(self.base_url)
                        used_chat = True
                        warnings.append(
                            "transcription endpoint unavailable; used chat completions"
                        )
                        payload = self._chat(raw_audio, mime, "", language)
                    elif (
                        failure.status == 400
                        and language is not None
                        and _LANGUAGE_ERROR.search(failure.body)
                    ):
                        warnings.append(
                            "server rejected the language; retried without it"
                        )
                        payload = self._transcriptions(raw_audio, filename, mime, None)
                    else:
                        raise
        except _HTTPFailure as failure:
            raise _public_error(failure) from None

        if isinstance(payload, dict):
            if "choices" in payload:
                choices = payload.get("choices") or []
                message = (choices[0].get("message") if choices else {}) or {}
                content = message.get("content")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                raw_text = content or ""
            else:
                raw_text = payload.get("text") or ""
        else:
            raw_text = ""

        parsed_language, text = parse_asr_output(raw_text)
        if used_chat and language is not None:
            warnings.append("language_not_enforced_on_chat")
        parsed_iso = iso_for(parsed_language) if parsed_language is not None else None
        if language is not None and parsed_iso is not None and parsed_iso != language:
            warnings.append("server_ignored_language")

        return TranscribeResult(
            text=text,
            language=parsed_language,
            language_source="server",
            segments=[],
            timestamp_source=None,
            warnings=warnings,
        )
