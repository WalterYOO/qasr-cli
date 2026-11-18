"""N2: remote OpenAI-compatible transcription client contracts."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jsonschema
import pytest

from qasr_cli import remote as remote_core
from qasr_cli.cli import main
from qasr_cli.remote import parse_asr_output
from tests.conftest import write_wav

SERVED = "qwen3-asr-1.7b"


@pytest.fixture(autouse=True)
def _clear_chat_only_cache():
    remote_core._CHAT_ONLY_SERVERS.clear()
    yield
    remote_core._CHAT_ONLY_SERVERS.clear()


class _Handler(BaseHTTPRequestHandler):
    served_models = [SERVED]
    transcription_status = 200
    transcription_statuses = []
    chat_status = 200
    response_text = "hello"
    response_language = "Chinese"
    chat_text = "language Chinese<asr_text>chat hello"
    delay = 0.0
    fail_filename = None
    redirect_to = None
    language_400 = False
    requests = []
    headers = []
    bodies = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    def _throttle(self):
        cls = type(self)
        with cls.lock:
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
        try:
            if cls.delay:
                time.sleep(cls.delay)
        finally:
            with cls.lock:
                cls.active -= 1

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib naming
        cls = type(self)
        cls.requests.append(("GET", self.path))
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self._send(200, {"object": "list",
                         "data": [{"id": model} for model in cls.served_models]})

    def do_POST(self):  # noqa: N802 - stdlib naming
        cls = type(self)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        cls.requests.append(("POST", self.path))
        cls.headers.append(dict(self.headers))
        cls.bodies.append(body)
        self._throttle()

        if cls.redirect_to:
            self.send_response(302)
            self.send_header("Location", cls.redirect_to)
            self.end_headers()
            return

        if self.path == "/v1/audio/transcriptions":
            if cls.language_400 and b'name="language"' in body:
                self._send(400, {"error": "unsupported language"})
                return
            status = (
                cls.transcription_statuses.pop(0)
                if cls.transcription_statuses
                else cls.transcription_status
            )
            if status != 200:
                self._send(status, {"error": f"status {status}"})
                return
            if cls.fail_filename and f'filename="{cls.fail_filename}"'.encode() in body:
                self._send(400, {"error": "bad audio"})
                return
            text = f"language {cls.response_language}<asr_text>{cls.response_text}"
            self._send(200, {"text": text, "usage": {"seconds": 1}})
            return

        if self.path == "/v1/chat/completions":
            if cls.chat_status != 200:
                self._send(cls.chat_status, {"error": "chat failed"})
                return
            self._send(200, {"choices": [{"message": {"content": cls.chat_text}}]})
            return

        self.send_error(404)

    def log_message(self, *args):  # silence test output
        pass


@contextmanager
def fake_server(**attrs):
    handler = type(
        "Handler",
        (_Handler,),
        {
            "requests": [],
            "headers": [],
            "bodies": [],
            "active": 0,
            "max_active": 0,
            "lock": threading.Lock(),
            **attrs,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run(capsys, argv):
    code = main(argv)
    out = capsys.readouterr()
    return code, out


def _payload(out):
    return json.loads(out.out)


def _post_paths(handler):
    return [path for method, path in handler.requests if method == "POST"]


# ---------------------------------------------------------------------------
# Pure parsing
# ---------------------------------------------------------------------------

def test_parse_asr_output_golden_cases():
    assert parse_asr_output("language Chinese<asr_text>你好") == ("Chinese", "你好")
    assert parse_asr_output("language None<asr_text>") == (None, "")
    assert parse_asr_output("language Chinese\nmeta: 1\n<asr_text>hi") == ("Chinese", "hi")
    assert parse_asr_output("plain text") == (None, "plain text")
    assert parse_asr_output(None) == (None, "")
    assert parse_asr_output("language Chinese<asr_text>a<asr_text>b") == ("Chinese", "ab")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def test_remote_transcription_multipart_and_language(capsys, wav_file, monkeypatch):
    monkeypatch.setenv("QASR_SERVE_TOKEN", "remote-secret")
    with fake_server() as (url, handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--language", "zh", "--json"],
        )
    assert code == 0, out.err
    payload = _payload(out)
    assert payload["result"]["text"] == "hello"
    assert payload["result"]["language"] == "Chinese"
    assert payload["result"]["language_source"] == "server"
    assert payload["result"]["warnings"] == []
    assert handler.requests[0] == ("GET", "/v1/models")
    assert _post_paths(handler) == ["/v1/audio/transcriptions"]
    body = handler.bodies[0]
    assert b'name="model"' in body and SERVED.encode() in body
    assert b'name="language"' in body and b"\r\nzh\r\n" in body
    assert b'filename="clip.wav"' in body
    assert handler.headers[0].get("Authorization") == "Bearer remote-secret"
    assert "remote-secret" not in out.out


def test_remote_chat_context_uses_system_without_prefill(capsys, wav_file):
    with fake_server() as (url, handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--context", "domain terms",
             "--json"],
        )
    assert code == 0, out.err
    assert _post_paths(handler) == ["/v1/chat/completions"]
    request = json.loads(handler.bodies[0].decode("utf-8"))
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert request["messages"][0]["content"] == "domain terms"
    assert "continue_final_message" not in request
    assert all(m["role"] != "assistant" for m in request["messages"])
    assert request["model"] == SERVED
    assert _payload(out)["result"]["text"] == "chat hello"


def test_remote_chat_explicit_language_warns(capsys, wav_file):
    with fake_server() as (url, _handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--context", "x",
             "--language", "zh", "--json"],
        )
    assert code == 0, out.err
    warnings = _payload(out)["result"]["warnings"]
    assert "language_not_enforced_on_chat" in warnings


def test_transcription_404_falls_back_to_chat_and_is_cached(capsys, tmp_path):
    first = write_wav(tmp_path / "a.wav", seconds=0.2)
    second = write_wav(tmp_path / "b.wav", seconds=0.2)
    with fake_server(transcription_status=404) as (url, handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(first), str(second), "--jobs", "1",
             "--language", "zh", "--json"],
        )
    assert code == 0, out.err
    payload = _payload(out)
    assert len(payload["items"]) == 2
    first_warnings = payload["items"][0]["result"]["warnings"]
    assert first_warnings[0] == "transcription endpoint unavailable; used chat completions"
    assert "language_not_enforced_on_chat" in first_warnings
    second_warnings = payload["items"][1]["result"]["warnings"]
    assert second_warnings == ["language_not_enforced_on_chat"]
    # First item probes transcription once, then both use chat only.
    assert _post_paths(handler) == ["/v1/audio/transcriptions", "/v1/chat/completions",
                                    "/v1/chat/completions"]


def test_language_400_retries_once_without_language(capsys, wav_file):
    with fake_server(language_400=True) as (url, handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--language", "zh", "--json"],
        )
    assert code == 0, out.err
    assert len(_post_paths(handler)) == 2
    assert b'name="language"' in handler.bodies[0]
    assert b'name="language"' not in handler.bodies[1]
    assert "server rejected the language" in _payload(out)["result"]["warnings"][0]


def test_remote_server_ignored_language_warns(capsys, wav_file):
    with fake_server(response_language="Chinese") as (url, _handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--language", "en", "--json"],
        )
    assert code == 0, out.err
    assert "server_ignored_language" in _payload(out)["result"]["warnings"]


def test_remote_5xx_retries_three_attempts_then_exit_three(capsys, wav_file):
    with fake_server(transcription_statuses=[500, 503, 500]) as (url, handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--retries", "2", "--json"],
        )
    assert code == 3
    assert len(_post_paths(handler)) == 3
    error = _payload(out)["error"]
    assert error["code"] == 3
    assert error["retryable"] is True


def test_remote_429_maps_to_three(capsys, wav_file):
    with fake_server(transcription_status=429) as (url, handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--retries", "0", "--json"],
        )
    assert code == 3
    assert len(_post_paths(handler)) == 1
    assert _payload(out)["error"]["details"]["http_status"] == 429


def test_remote_408_maps_to_three(capsys, wav_file):
    with fake_server(transcription_status=408) as (url, _handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--retries", "0", "--json"],
        )
    assert code == 3
    assert _payload(out)["error"]["details"]["http_status"] == 408


def test_remote_plain_4xx_maps_to_four_without_retry(capsys, wav_file):
    with fake_server(transcription_status=422) as (url, handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--retries", "2", "--json"],
        )
    assert code == 4
    assert len(_post_paths(handler)) == 1
    assert _payload(out)["error"]["details"]["http_status"] == 422


def test_remote_401_maps_to_four(capsys, wav_file):
    with fake_server(transcription_status=401) as (url, _handler):
        code, out = _run(capsys, ["--server", url, "transcribe", str(wav_file), "--json"])
    assert code == 4
    assert "token" in _payload(out)["error"].get("hint", "").casefold()


def test_remote_timeout_maps_to_three(capsys, wav_file):
    with fake_server(delay=0.6) as (url, _handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--timeout", "0.1",
             "--retries", "0", "--json"],
        )
    assert code == 3
    assert "timed out" in _payload(out)["error"]["message"].casefold() or \
        "time" in _payload(out)["error"]["message"].casefold()


def test_remote_capabilities_rejected_before_any_request(capsys, wav_file):
    with fake_server() as (url, handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--timestamps",
             "--language", "zh", "--json"],
        )
    assert code == 4
    assert "timestamps" in _payload(out)["error"]["message"]
    assert handler.requests == []


def test_remote_align_rejected(capsys, wav_file, tmp_path):
    text = tmp_path / "a.txt"
    text.write_text("hello world", encoding="utf-8")
    with fake_server() as (url, handler):
        code, out = _run(
            capsys,
            ["--server", url, "align", str(wav_file), "--text", str(text), "--json"],
        )
    assert code == 4
    assert "alignment" in _payload(out)["error"]["message"]
    assert handler.requests == []


# ---------------------------------------------------------------------------
# Concurrency / state
# ---------------------------------------------------------------------------

def test_remote_jobs_default_concurrency(capsys, tmp_path):
    files = [write_wav(tmp_path / f"{i}.wav", seconds=0.2) for i in range(3)]
    with fake_server(delay=0.25) as (url, handler):
        started = time.perf_counter()
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", *[str(f) for f in files], "--json"],
        )
        elapsed = time.perf_counter() - started
    assert code == 0, out.err
    assert handler.max_active >= 2
    assert elapsed < 3 * 0.25 + 1.0
    assert [item["input"]["id"] for item in _payload(out)["items"]] == [
        "0", "1", "2",
    ]


def test_remote_jobs_one_is_serial(capsys, tmp_path):
    files = [write_wav(tmp_path / f"{i}.wav", seconds=0.2) for i in range(3)]
    with fake_server(delay=0.1) as (url, handler):
        code, _out = _run(
            capsys,
            ["--server", url, "transcribe", *[str(f) for f in files], "--jobs", "1",
             "--json"],
        )
    assert code == 0
    assert handler.max_active == 1


def test_remote_batch_size_is_ignored_with_warning(capsys, wav_file):
    with fake_server() as (url, _handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--batch-size", "4", "--json"],
        )
    assert code == 0, out.err
    assert "ignores --batch-size" in out.err


def test_remote_state_appends_are_valid_jsonl(capsys, tmp_path):
    files = [write_wav(tmp_path / f"{i}.wav", seconds=0.2) for i in range(4)]
    out_dir = tmp_path / "out"
    with fake_server(delay=0.05) as (url, _handler):
        code, _out = _run(
            capsys,
            ["--server", url, "transcribe", *[str(f) for f in files],
             "--out-dir", str(out_dir), "--jobs", "4", "--json"],
        )
    assert code == 0
    records = [
        json.loads(line)
        for line in (out_dir / ".qasr" / "state.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    statuses = [record["status"] for record in records]
    assert statuses.count("running") == 4
    assert statuses.count("done") == 4


def test_remote_fail_fast_leaves_no_running(capsys, tmp_path):
    bad = write_wav(tmp_path / "bad.wav", seconds=0.2)
    good = write_wav(tmp_path / "good.wav", seconds=0.2)
    out_dir = tmp_path / "out"
    with fake_server(fail_filename="bad.wav", delay=0.3) as (url, _handler):
        code, _out = _run(
            capsys,
            ["--server", url, "transcribe", str(bad), str(good), "--out-dir",
             str(out_dir), "--jobs", "2", "--fail-fast", "--json"],
        )
    assert code == 6  # one failure + one completed/skipped item
    records = [
        json.loads(line)
        for line in (out_dir / ".qasr" / "state.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    final = {}
    for record in records:
        final[record["id"]] = record["status"]
    assert final["bad"] == "failed"
    assert final["good"] in ("skipped", "done")
    assert "running" not in final.values()


def test_remote_model_lookup_honors_served_name(capsys, wav_file):
    with fake_server(served_models=[SERVED]) as (url, handler):
        code, _out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--model", "1.7b", "--json"],
        )
    assert code == 0
    body = handler.bodies[0]
    assert SERVED.encode() in body
    assert b"Qwen/Qwen3-ASR-1.7B" not in body


def test_remote_model_unknown_with_multiple_served_is_input_error(capsys, wav_file):
    with fake_server(served_models=["alpha", "beta"]) as (url, handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--model", "gamma", "--json"],
        )
    assert code == 4
    assert "alpha" in _payload(out)["error"]["hint"]
    assert _post_paths(handler) == []


def test_remote_single_served_model_fallback_is_reported(capsys, wav_file):
    with fake_server(served_models=["custom-asr"]) as (url, _handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(wav_file), "--model", "0.6b", "--json"],
        )
    assert code == 0, out.err
    warnings = _payload(out)["result"]["warnings"]
    assert any("using the only served model" in warning for warning in warnings)


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

def _load_schema(name: str):
    root = Path(__file__).resolve().parents[2] / "schemas"
    return json.loads((root / f"{name}.json").read_text(encoding="utf-8"))


def test_remote_language_source_server_validates_single_and_batch(capsys, tmp_path):
    files = [write_wav(tmp_path / f"{i}.wav", seconds=0.2) for i in range(2)]
    with fake_server() as (url, _handler):
        code, out = _run(
            capsys,
            ["--server", url, "transcribe", str(files[0]), "--json"],
        )
        assert code == 0
        single = _payload(out)
        jsonschema.validate(single, _load_schema("transcribe-item"))
        assert single["result"]["language_source"] == "server"

        code, out = _run(
            capsys,
            ["--server", url, "transcribe", *[str(f) for f in files], "--json"],
        )
        assert code == 0
        batch = _payload(out)
        jsonschema.validate(batch, _load_schema("transcribe-batch"))
        assert {item["result"]["language_source"] for item in batch["items"]} == {"server"}


def test_remote_cross_origin_redirect_is_refused_without_leaking_token(
    capsys, wav_file, monkeypatch
):
    monkeypatch.setenv("QASR_SERVE_TOKEN", "cross-origin-secret")
    with fake_server() as (url_a, handler_a), fake_server() as (url_b, handler_b):
        handler_a.redirect_to = f"{url_b}/v1/audio/transcriptions"
        code, out = _run(
            capsys,
            ["--server", url_a, "transcribe", str(wav_file), "--retries", "0", "--json"],
        )
    assert code == 3
    assert "redirect" in _payload(out)["error"]["message"]
    assert [path for method, path in handler_b.requests if method == "POST"] == []
    assert all("Authorization" not in headers for headers in handler_b.headers)
