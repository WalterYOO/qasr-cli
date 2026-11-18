"""A small, stdlib-only fake OpenAI / qwen-asr server for black-box tests.

It records requests, parses multipart and JSON bodies, injects delays/errors,
and tracks the maximum number of concurrent requests. Tests must never depend
on implementation internals; they only observe HTTP.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable, Optional
from urllib.parse import parse_qs, urlparse


@dataclass
class FilePart:
    filename: str
    content_type: str
    data: bytes


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, FilePart]]:
    fields: dict[str, str] = {}
    files: dict[str, FilePart] = {}
    if "multipart/form-data" not in content_type.lower():
        return fields, files
    raw = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + body
    )
    message = BytesParser(policy=policy.default).parsebytes(raw)
    if not message.is_multipart():
        return fields, files
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        data = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            files[name] = FilePart(
                filename=filename,
                content_type=part.get_content_type() or "application/octet-stream",
                data=data,
            )
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = data.decode(charset, errors="replace")
    return fields, files


@dataclass
class Request:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")

    @property
    def authorization(self) -> str:
        return self.headers.get("authorization", "")

    @property
    def json(self) -> Any:
        if not self.body:
            return None
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    @property
    def form(self) -> tuple[dict[str, str], dict[str, FilePart]]:
        return parse_multipart(self.body, self.content_type)

    @property
    def query(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(self.path).query)


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0


def json_response(payload: Any, *, status: int = 200,
                  headers: Optional[dict[str, str]] = None,
                  delay: float = 0.0) -> Response:
    merged = {"Content-Type": "application/json"}
    if headers:
        merged.update(headers)
    return Response(
        status=status,
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=merged,
        delay=delay,
    )


def text_response(body: str, *, status: int = 200,
                  headers: Optional[dict[str, str]] = None,
                  delay: float = 0.0) -> Response:
    merged = {"Content-Type": "text/plain; charset=utf-8"}
    if headers:
        merged.update(headers)
    return Response(status=status, body=body.encode("utf-8"), headers=merged, delay=delay)


def error_response(message: str, *, status: int = 400,
                   headers: Optional[dict[str, str]] = None,
                   delay: float = 0.0) -> Response:
    return json_response(
        {"error": {"message": message, "type": "fake_error"}},
        status=status,
        headers=headers,
        delay=delay,
    )


class Sequence:
    """Thread-safe response sequence for a single endpoint."""

    def __init__(self, responses: Iterable[Response]) -> None:
        self._responses = list(responses)
        self._index = 0
        self._lock = threading.Lock()

    def __call__(self, request: Request, server: FakeOpenAIServer) -> Response:
        with self._lock:
            index = self._index
            self._index += 1
        if index < len(self._responses):
            return self._responses[index]
        return self._responses[-1]


class FakeOpenAIServer:
    """Threaded fake server.

    ``handler(request, server)`` may return :class:`Response` or ``None``
    (404). If no handler is supplied, ``transcription_response`` and
    ``chat_response`` are used.
    """

    def __init__(
        self,
        *,
        handler: Optional[Callable[[Request, FakeOpenAIServer], Optional[Response]]] = None,
        models: Optional[Iterable[str]] = None,
        models_response: Optional[Callable[[Request, FakeOpenAIServer], Response]] = None,
        auth_token: Optional[str] = None,
        require_auth: bool = False,
    ) -> None:
        self.handler = handler
        self.models = list(models or ["Qwen/Qwen3-ASR-1.7B"])
        self.models_response = models_response
        self.auth_token = auth_token
        self.require_auth = require_auth
        self.transcription_response: Callable[[Request, FakeOpenAIServer], Response] = (
            lambda request, server: json_response({"text": "remote text"})
        )
        self.chat_response: Callable[[Request, FakeOpenAIServer], Response] = (
            lambda request, server: json_response({
                "choices": [{
                    "message": {"role": "assistant", "content": "language Chinese<asr_text>远程文本"},
                    "finish_reason": "stop",
                }]
            })
        )
        self.requests: list[Request] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                owner._serve(self, "GET")

            def do_POST(self) -> None:  # noqa: N802
                owner._serve(self, "POST")

            def log_message(self, *_args: Any) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.httpd.owner = self  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeOpenAIServer:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.max_active = 0

    def requests_for(self, path: str) -> list[Request]:
        return [request for request in self.requests if request.path == path]

    def post_requests(self) -> list[Request]:
        return [request for request in self.requests if request.method == "POST"]

    # -- dispatch ---------------------------------------------------------
    def _serve(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        length = int(handler.headers.get("Content-Length", "0") or 0)
        body = handler.rfile.read(length) if length else b""
        headers = {key.lower(): value for key, value in handler.headers.items()}
        request = Request(method=method, path=handler.path, headers=headers, body=body)

        with self._lock:
            self.requests.append(request)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            response = self._dispatch(request)
            if response is None:
                response = text_response("not found", status=404)
            if response.delay:
                time.sleep(response.delay)
            handler.send_response(response.status)
            for key, value in response.headers.items():
                handler.send_header(key, value)
            handler.send_header("Content-Length", str(len(response.body)))
            handler.end_headers()
            if response.body:
                handler.wfile.write(response.body)
        finally:
            with self._lock:
                self.active -= 1

    def _dispatch(self, request: Request) -> Optional[Response]:
        if self.require_auth:
            expected = f"Bearer {self.auth_token}" if self.auth_token else None
            if expected is None or request.authorization != expected:
                return error_response("unauthorized", status=401)

        if request.method == "GET" and request.path.split("?", 1)[0] == "/v1/models":
            if self.models_response is not None:
                return self.models_response(request, self)
            return json_response({
                "object": "list",
                "data": [{"id": model, "object": "model"} for model in self.models],
            })

        if self.handler is not None:
            return self.handler(request, self)

        path = request.path.split("?", 1)[0]
        if request.method == "POST" and path == "/v1/audio/transcriptions":
            return self.transcription_response(request, self)
        if request.method == "POST" and path == "/v1/chat/completions":
            return self.chat_response(request, self)
        return None

