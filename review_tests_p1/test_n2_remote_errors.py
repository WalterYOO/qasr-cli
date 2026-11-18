"""N2: HTTP status, timeout, retry, and partial-failure mapping."""

from __future__ import annotations

import shutil
import socket
import time
from pathlib import Path

import pytest
from fake_openai import error_response, json_response, text_response
from helpers import assert_exit, parse_json, write_wav


def _fail_all(status: int, message: str = "remote failure"):
    def handler(request, srv):
        return error_response(message, status=status)

    return handler


def test_401_maps_to_input_error(qasr, fake_openai, short_wav):
    server = fake_openai(handler=_fail_all(401, "unauthorized"))
    proc = qasr.run("--server", server.url, "transcribe", short_wav, "--json")
    assert_exit(proc, 4, context="remote 401")
    error = parse_json(proc)["error"]
    assert error["kind"] in {"input", "auth", "remote"}
    assert error.get("retryable") is False
    assert (error.get("details") or {}).get("http_status") == 401
    assert len(server.post_requests()) == 1


@pytest.mark.parametrize("status", [400, 403, 413, 422])
def test_4xx_maps_to_exit_4(qasr, fake_openai, short_wav, status):
    server = fake_openai(handler=_fail_all(status))
    proc = qasr.run("--server", server.url, "transcribe", short_wav, "--json")
    assert_exit(proc, 4, context=f"remote {status}")
    error = parse_json(proc)["error"]
    assert error["code"] == 4
    assert error.get("retryable") is False
    assert (error.get("details") or {}).get("http_status") == status
    assert len(server.requests_for("/v1/audio/transcriptions")) == 1


def test_404_falls_back_to_chat_then_reports_404(qasr, fake_openai, short_wav):
    server = fake_openai(handler=lambda request, srv: error_response("missing", status=404))
    proc = qasr.run("--server", server.url, "transcribe", short_wav, "--json")
    assert_exit(proc, 4, context="remote 404")
    assert len(server.requests_for("/v1/audio/transcriptions")) == 1
    assert len(server.requests_for("/v1/chat/completions")) == 1
    error = parse_json(proc)["error"]
    assert (error.get("details") or {}).get("http_status") == 404


def test_405_falls_back_to_chat(qasr, fake_openai, short_wav):
    def handler(request, srv):
        path = request.path.split("?", 1)[0]
        if path == "/v1/audio/transcriptions":
            return error_response("method not allowed", status=405)
        if path == "/v1/chat/completions":
            return srv.chat_response(request, srv)
        return None

    server = fake_openai(handler=handler)
    proc = qasr.run("--server", server.url, "transcribe", short_wav, "--json")
    assert_exit(proc, 0, context="remote 405 fallback")
    assert len(server.requests_for("/v1/chat/completions")) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_retryable_status_maps_to_exit_3_and_retries(qasr, fake_openai, short_wav, status):
    server = fake_openai(handler=_fail_all(status))
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--retries", "2", "--json",
    )
    assert_exit(proc, 3, context=f"remote {status}")
    error = parse_json(proc)["error"]
    assert error["code"] == 3
    assert error.get("retryable") is True
    assert (error.get("details") or {}).get("http_status") == status
    # Natural reading of "--retries 2": two retries after the first attempt.
    assert len(server.requests_for("/v1/audio/transcriptions")) == 3


def test_retry_can_eventually_succeed(qasr, fake_openai, short_wav):
    server = fake_openai()
    calls = {"n": 0}

    def handler(request, srv):
        calls["n"] += 1
        if calls["n"] == 1:
            return error_response("temporary", status=503)
        return json_response({"text": "language Chinese<asr_text>重试成功"})

    server.handler = handler
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--retries", "1", "--json",
    )
    assert_exit(proc, 0, context="retry success")
    assert parse_json(proc)["result"]["text"] == "重试成功"
    assert calls["n"] == 2


def test_timeout_maps_to_exit_3(qasr, fake_openai, short_wav):
    server = fake_openai(
        handler=lambda request, srv: error_response("slow", status=200, delay=2.0)
    )
    started = time.monotonic()
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--timeout", "1", "--retries", "1", "--json",
        timeout=30,
    )
    elapsed = time.monotonic() - started
    assert_exit(proc, 3, context="remote timeout")
    error = parse_json(proc)["error"]
    assert error["code"] == 3
    assert error.get("retryable") is True
    assert elapsed < 20, elapsed
    assert len(server.requests_for("/v1/audio/transcriptions")) == 2


def test_connection_refused_maps_to_exit_3(qasr, short_wav):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    proc = qasr.run(
        "--server", f"http://127.0.0.1:{port}", "transcribe", short_wav,
        "--retries", "0", "--json", timeout=30,
    )
    assert_exit(proc, 3, context="connection refused")
    error = parse_json(proc)["error"]
    assert error["code"] == 3
    assert error.get("retryable") is True


def test_http_408_is_retryable_and_exits_3(qasr, fake_openai, short_wav):
    server = fake_openai(handler=_fail_all(408, "request timeout"))
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--retries", "2", "--json", timeout=60,
    )
    assert_exit(proc, 3, context="remote 408")
    error = parse_json(proc)["error"]
    assert error.get("retryable") is True
    assert (error.get("details") or {}).get("http_status") == 408
    assert len(server.requests_for("/v1/audio/transcriptions")) == 3


def test_error_body_is_truncated_and_secret_free(qasr, fake_openai, short_wav):
    token = "{{SECRET_h4n8s2kd}}"
    huge = "Authorization: Bearer " + token + " " + ("x" * 100_000)

    def handler(request, srv):
        return text_response(huge, status=500)

    server = fake_openai(handler=handler)
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav, "--json",
        env={"QASR_SERVE_TOKEN": token}, timeout=30,
    )
    assert_exit(proc, 3, context="huge error body")
    assert len(proc.stdout) < 20_000, len(proc.stdout)
    parse_json(proc)
    assert token not in proc.stdout
    assert token not in proc.stderr


def test_language_rejection_does_not_retry_forever(qasr, fake_openai, short_wav):
    server = fake_openai(
        handler=lambda request, srv: error_response("Unsupported language: zh", status=400)
    )
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--language", "zh", "--retries", "2", "--json",
    )
    assert_exit(proc, 4, context="language always rejected")
    assert len(server.requests_for("/v1/audio/transcriptions")) == 2
    assert parse_json(proc)["error"]["code"] == 4


@pytest.mark.slow
def test_upload_over_25mb_is_chunked_client_side(qasr, fake_openai, tmp_path):
    """P1: >25 MB input must be split; every upload stays under the limit."""
    server = fake_openai()
    big = write_wav(tmp_path / "big.wav", seconds=850.0)
    big_size = Path(big).stat().st_size
    assert big_size > 25 * 1024 * 1024
    proc = qasr.run(
        "--server", server.url, "transcribe", big, "--json", timeout=120
    )
    assert_exit(proc, 0, context="over 25 MB chunked")
    payload = parse_json(proc)
    assert payload.get("ok") is True, payload
    assert (payload.get("result") or {}).get("text") is not None, payload
    uploads = server.requests_for("/v1/audio/transcriptions")
    assert len(uploads) >= 2, "850s audio at the 270s default must be split"
    for request in uploads:
        assert len(request.body) < 25 * 1024 * 1024, len(request.body)
    largest = max(len(request.body) for request in uploads)
    assert largest < big_size, (
        "a single request carried the whole original file: "
        f"{largest} >= {big_size}"
    )


def test_404_fallback_with_explicit_language_warns(qasr, fake_openai, short_wav):
    server = fake_openai()

    def handler(request, srv):
        path = request.path.split("?", 1)[0]
        if path == "/v1/audio/transcriptions":
            return error_response("not found", status=404)
        if path == "/v1/chat/completions":
            return json_response(
                {"choices": [{"message": {
                    "role": "assistant",
                    "content": "language English<asr_text>hello",
                }}]}
            )
        return None

    server.handler = handler
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--language", "zh", "--json",
    )
    assert_exit(proc, 0, context="404 fallback language")
    result = parse_json(proc)["result"]
    assert result["language"] == "English"
    assert result["text"] == "hello"
    warnings = " ".join(str(item) for item in result.get("warnings") or [])
    assert "language_not_enforced_on_chat" in warnings, warnings
    assert "server_ignored_language" in warnings, warnings


def test_all_remote_failures_use_first_error_code(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai(handler=_fail_all(500, "boom"))
    out = tmp_path / "out"
    second = tmp_path / "second.wav"
    shutil.copyfile(short_wav, second)
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav, second,
        "--out-dir", out, "--json", timeout=60,
    )
    assert_exit(proc, 3, context="all remote failed")
    payload = parse_json(proc)
    assert payload["summary"]["failed"] == 2
    assert payload.get("errors")
    assert payload["errors"][0]["error"]["code"] == 3


def test_partial_remote_failure_exits_6(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai()

    def handler(request, srv):
        if request.path.split("?", 1)[0] != "/v1/audio/transcriptions":
            return None
        fields, files = request.form
        name = files["file"].filename if "file" in files else ""
        if "bad" in name:
            return error_response("bad audio", status=400)
        return json_response({"text": "language Chinese<asr_text>ok"})

    server.handler = handler
    bad = tmp_path / "bad.wav"
    shutil.copyfile(short_wav, bad)
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", bad, short_wav,
        "--out-dir", out, "--json",
    )
    assert_exit(proc, 6, context="partial remote failure")
    payload = parse_json(proc)
    assert payload["summary"] == {"total": 2, "ok": 1, "failed": 1, "skipped": 0}
    assert payload["ok"] is False
    assert payload["errors"][0]["id"] == "bad"
    assert payload["errors"][0]["error"]["code"] == 4


def test_retryable_flag_on_batch_items(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai(handler=_fail_all(503, "unavailable"))
    second = tmp_path / "second.wav"
    shutil.copyfile(short_wav, second)
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav, second,
        "--out-dir", out, "--json", "--retries", "0",
    )
    assert_exit(proc, 3, context="retryable batch item")
    payload = parse_json(proc)
    assert payload["summary"]["failed"] == 2
    assert all(entry["error"]["retryable"] is True for entry in payload["errors"])


def test_first_error_is_selected_by_input_order(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai()

    def handler(request, srv):
        if request.path.split("?", 1)[0] != "/v1/audio/transcriptions":
            return None
        fields, files = request.form
        name = files["file"].filename if "file" in files else ""
        if name.startswith("first"):
            return error_response("first bad", status=400, delay=0.4)
        return error_response("second unavailable", status=500)

    server.handler = handler
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    shutil.copyfile(short_wav, first)
    shutil.copyfile(short_wav, second)
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", first, second,
        "--out-dir", out, "--retries", "0", "--json", timeout=60,
    )
    assert_exit(proc, 4, context="first error by input order")
    payload = parse_json(proc)
    assert payload["errors"][0]["id"] == "first", payload
    assert payload["errors"][0]["error"]["code"] == 4


def test_429_retry_after_within_ten_seconds_is_respected(
    qasr, fake_openai, short_wav
):
    server = fake_openai(
        handler=lambda request, srv: error_response(
            "slow down", status=429, headers={"Retry-After": "1"}
        )
    )
    started = time.monotonic()
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--retries", "1", "--json", timeout=30,
    )
    elapsed = time.monotonic() - started
    assert_exit(proc, 3, context="429 retry-after")
    assert len(server.requests_for("/v1/audio/transcriptions")) == 2
    assert elapsed >= 0.9, elapsed
