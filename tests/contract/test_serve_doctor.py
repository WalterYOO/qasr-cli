"""N1: serve dry-run/passthrough and doctor --server."""

from __future__ import annotations

import json
import pathlib
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from qasr_cli import serve as serve_core
from qasr_cli.cli import main
from qasr_cli.remote import probe_server, redact_url


class _Handler(BaseHTTPRequestHandler):
    status = 200
    payload = {"object": "list", "data": [{"id": "Qwen/Qwen3-ASR-1.7B"}]}
    delay = 0.0
    expected_token = None
    redirect_to = None
    requests = []
    seen_headers = {}

    def do_GET(self):  # noqa: N802 - stdlib naming
        type(self).requests.append(self.path)
        if self.redirect_to and self.path == "/v1/models":
            self.send_response(302)
            self.send_header("Location", self.redirect_to)
            self.end_headers()
            return
        if self.path not in ("/v1/models", "/v1/models2"):
            self.send_error(404)
            return
        if self.delay:
            import time

            time.sleep(self.delay)
        type(self).seen_headers = dict(self.headers)
        if self.expected_token and self.headers.get("Authorization") != f"Bearer {self.expected_token}":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"unauthorized"}')
            return
        body = json.dumps(self.payload).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence test output
        pass


@contextmanager
def fake_server(**attrs):
    handler = type("Handler", (_Handler,), {**attrs, "seen_headers": {}, "requests": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _serve_payload(capsys, argv):
    code = main(argv)
    out = capsys.readouterr()
    assert code == 0, out.err
    return json.loads(out.out)


def test_serve_dry_run_json_uses_local_complete_model(capsys, model_factory):
    model = model_factory("Qwen3-ASR-1.7B")
    payload = _serve_payload(capsys, ["serve", "--dry-run", "--json"])
    assert payload["command"] == "serve"
    assert payload["resolved"] == str(model.resolve())
    assert payload["source"] == "local"
    assert payload["argv"][0].endswith("qwen-asr-serve")
    assert "vllm" not in sys.modules


def test_serve_passthrough_and_known_flags(capsys):
    payload = _serve_payload(
        capsys,
        ["serve", "0.6b", "--host", "0.0.0.0", "--port", "9000",
         "--gpu-memory-utilization", "0.8", "--tensor-parallel-size", "2",
         "--dry-run", "--json"],
    )
    argv = payload["argv"]
    assert payload["resolved"] == "Qwen/Qwen3-ASR-0.6B"
    assert payload["source"] == "repo"
    assert argv[1:3] == ["Qwen/Qwen3-ASR-0.6B", "--host"]
    assert "--port" in argv and "9000" in argv
    assert "--gpu-memory-utilization" in argv and "0.8" in argv
    assert argv[-2:] == ["--tensor-parallel-size", "2"]


def test_serve_unknown_flag_value_is_not_the_model(capsys):
    payload = _serve_payload(
        capsys, ["--json", "serve", "--tensor-parallel-size", "2", "--dry-run"]
    )
    assert payload["model"] == "1.7b"
    assert payload["resolved"] == "Qwen/Qwen3-ASR-1.7B"
    assert payload["argv"][-2:] == ["--tensor-parallel-size", "2"]


def test_serve_missing_binary_exits_three(capsys, monkeypatch):
    monkeypatch.setattr(serve_core.shutil, "which", lambda name: None)
    code = main(["serve", "0.6b", "--json"])
    out = capsys.readouterr()
    assert code == 3
    payload = json.loads(out.out)
    assert payload["error"]["kind"] == "model"
    assert "qwen-asr-serve" in payload["error"]["message"]


def test_serve_launch_execs_with_normalized_argv(capsys, monkeypatch):
    calls = {}
    monkeypatch.setattr(serve_core.shutil, "which", lambda name: "/opt/bin/qwen-asr-serve")

    def fake_execvp(path, argv):
        calls["path"] = path
        calls["argv"] = list(argv)

    monkeypatch.setattr(serve_core.os, "execvp", fake_execvp)
    code = main(["serve", "0.6b", "--host", "127.0.0.1", "--port", "8000",
                 "--tensor-parallel-size", "2"])
    capsys.readouterr()
    assert code == 0
    assert calls["path"] == "/opt/bin/qwen-asr-serve"
    assert calls["argv"][0] == "/opt/bin/qwen-asr-serve"
    assert calls["argv"][1] == "Qwen/Qwen3-ASR-0.6B"
    assert calls["argv"][-2:] == ["--tensor-parallel-size", "2"]


def test_doctor_server_reports_models_and_latency(capsys):
    with fake_server() as (url, _handler):
        code = main(["doctor", "--server", url, "--json"])
    out = capsys.readouterr()
    assert code == 0, out.err
    payload = json.loads(out.out)
    check = next(item for item in payload["checks"] if item["name"] == "server")
    assert check["status"] == "ok"
    assert check["value"]["models"] == ["Qwen/Qwen3-ASR-1.7B"]
    assert isinstance(check["value"]["latency_ms"], int)


def test_doctor_server_401_warns_and_strict_fails(capsys):
    with fake_server(status=401) as (url, _handler):
        code = main(["doctor", "--server", url, "--json"])
        out = capsys.readouterr()
        assert code == 0
        payload = json.loads(out.out)
        check = next(item for item in payload["checks"] if item["name"] == "server")
        assert check["status"] == "warn"
        assert payload["status"] in ("warn", "fail")  # torch is absent in the test venv
        assert "token" in (check.get("hint") or "").casefold()

        assert main(["doctor", "--server", url, "--json", "--strict"]) == 3
        capsys.readouterr()


def test_doctor_server_invalid_url_is_fail_but_exit_zero(capsys):
    code = main(["doctor", "--server", "not-a-url", "--json"])
    out = capsys.readouterr()
    assert code == 0
    payload = json.loads(out.out)
    check = next(item for item in payload["checks"] if item["name"] == "server")
    assert check["status"] == "fail"


def test_probe_server_times_out_without_hanging():
    with fake_server(delay=0.5) as (url, _handler):
        info = probe_server(url, timeout=0.1)
    assert info["reachable"] is False
    assert info["timed_out"] is True


def test_serve_token_never_leaks(capsys, monkeypatch):
    secret = "test-secret-token-123"
    monkeypatch.setenv("QASR_SERVE_TOKEN", secret)
    with fake_server(expected_token=secret) as (url, handler):
        assert main(["doctor", "--server", url, "--json"]) == 0
        doctor_out = capsys.readouterr().out
        doctor_payload = json.loads(doctor_out)
        server_check = next(
            item for item in doctor_payload["checks"] if item["name"] == "server"
        )
        assert server_check["status"] == "ok"
        assert handler.seen_headers.get("Authorization") == f"Bearer {secret}"

    assert main(["config", "show", "--json"]) == 0
    config_out = capsys.readouterr().out

    assert main(["serve", "0.6b", "--dry-run", "--json"]) == 0
    serve_out = capsys.readouterr().out

    for text in (doctor_out, config_out, serve_out):
        assert secret not in text
    assert server_check["value"]["token_set"] is True


def test_help_does_not_import_inference_libraries():
    import os
    import subprocess

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    script = """
import sys
from qasr_cli.cli import main
assert main(["--help"]) == 0
loaded = {"vllm", "torch", "qwen_asr"} & set(sys.modules)
assert not loaded, loaded
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=repo_root, capture_output=True, text=True,
        env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_server_conflicts_with_explicit_backend(capsys):
    code = main(["--server", "http://127.0.0.1:9", "--backend", "transformers",
                 "languages", "--json"])
    out = capsys.readouterr()
    assert code == 2
    assert "conflicts" in json.loads(out.out)["error"]["message"]


def test_backend_remote_requires_server(capsys):
    code = main(["--backend", "remote", "languages", "--json"])
    out = capsys.readouterr()
    assert code == 2
    assert "--server" in json.loads(out.out)["error"]["message"]


def test_server_userinfo_is_redacted(capsys):
    code = main(["doctor", "--server", "http://alice:s3cret@127.0.0.1:9", "--json"])
    out = capsys.readouterr()
    assert code == 0
    payload = json.loads(out.out)
    check = next(item for item in payload["checks"] if item["name"] == "server")
    assert check["value"]["url"] == "http://127.0.0.1:9"
    assert "s3cret" not in out.out


def test_same_origin_redirect_is_followed_with_authorization(monkeypatch):
    secret = "same-origin-token"
    monkeypatch.setenv("QASR_SERVE_TOKEN", secret)
    with fake_server() as (url, handler):
        handler.redirect_to = "/v1/models2"
        info = probe_server(url)
    assert info["http_status"] == 200
    assert info["models"] == ["Qwen/Qwen3-ASR-1.7B"]
    assert handler.requests == ["/v1/models", "/v1/models2"]
    assert handler.seen_headers.get("Authorization") == f"Bearer {secret}"


def test_cross_origin_redirect_is_refused_before_sending_credentials(monkeypatch):
    monkeypatch.setenv("QASR_SERVE_TOKEN", "cross-origin-token")
    with fake_server() as (url_a, handler_a), fake_server() as (url_b, handler_b):
        handler_a.redirect_to = url_b + "/v1/models"
        info = probe_server(url_a)
    assert info["http_status"] == 302
    assert "cross-origin" in (info.get("error") or "")
    assert handler_b.requests == []
    assert handler_b.seen_headers == {}


def test_redirect_chain_is_capped(monkeypatch):
    monkeypatch.setenv("QASR_SERVE_TOKEN", "loopy")
    with fake_server() as (url, handler):
        handler.redirect_to = "/v1/models"
        info = probe_server(url)
    assert info["http_status"] == 302  # urllib raises after max_redirections
    assert len(handler.requests) <= 1 + 3


def test_redact_url_masks_userinfo_and_sensitive_query():
    assert redact_url("http://alice:pw@host:8000/v1?x=1") == "http://host:8000/v1?x=1"
    redacted = redact_url("http://host/v1?token=abc&x=1")
    assert "abc" not in redacted
    assert "token=%2A%2A%2A" in redacted or "token=***" in redacted


def test_serve_known_flag_after_unknown_value_keeps_model(capsys):
    payload = _serve_payload(
        capsys,
        ["serve", "1.7b", "--tensor-parallel-size", "2", "--host", "0.0.0.0",
         "--dry-run", "--json"],
    )
    assert payload["resolved"] == "Qwen/Qwen3-ASR-1.7B"
    assert payload["model"] == "1.7b"
    assert "--host" in payload["argv"] and "0.0.0.0" in payload["argv"]
    assert payload["argv"][-2:] == ["--tensor-parallel-size", "2"]
