"""P1 security: QASR_SERVE_TOKEN must never leak to output or disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fake_openai import Response, error_response
from helpers import (
    assert_exit,
    parse_json,
    scan_text_for_secret,
    scan_tree_for_secret,
)


def _assert_clean(proc, secret: str) -> None:
    assert not scan_text_for_secret(proc.stdout, secret), proc.stdout
    assert not scan_text_for_secret(proc.stderr, secret), proc.stderr


def test_doctor_401_echo_does_not_leak_token(qasr, fake_openai):
    token = "{{SECRET_p3k9v1zn}}"

    def models_response(request, srv):
        return error_response(
            f"invalid auth header: {request.authorization}", status=401
        )

    server = fake_openai(
        auth_token=token, require_auth=True, models_response=models_response
    )
    proc = qasr.run(
        "doctor", "--server", server.url, "--json",
        env={"QASR_SERVE_TOKEN": token}, timeout=30,
    )
    assert_exit(proc, 0, context="doctor token echo")
    _assert_clean(proc, token)


def test_verbose_remote_logs_do_not_leak_token(qasr, fake_openai, short_wav):
    token = "{{SECRET_r7m2x8cw}}"
    server = fake_openai(auth_token=token, require_auth=True)
    proc = qasr.run(
        "--server", server.url, "--verbose", "transcribe", short_wav, "--json",
        env={"QASR_SERVE_TOKEN": token},
    )
    assert_exit(proc, 0, context="verbose token")
    _assert_clean(proc, token)


def test_token_is_not_written_to_config_or_state(
    qasr, fake_openai, tmp_path, short_wav
):
    token = "{{SECRET_t5q8w2ah}}"
    server = fake_openai(auth_token=token, require_auth=True)
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--out-dir", out, "--json",
        env={"QASR_SERVE_TOKEN": token},
    )
    assert_exit(proc, 0, context="token on disk")
    _assert_clean(proc, token)
    hits = scan_tree_for_secret(qasr.home, token)
    hits += scan_tree_for_secret(out, token)
    assert not hits, hits


def test_token_is_not_written_to_jobs(qasr, fake_openai, tmp_path, short_wav):
    token = "{{SECRET_t5q8w2ah}}"
    server = fake_openai(auth_token=token, require_auth=True)
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--out-dir", out, "--json",
        env={"QASR_SERVE_TOKEN": token},
    )
    assert_exit(proc, 0, context="token on disk")
    jobs = qasr.run("jobs", "list", "--json")
    assert_exit(jobs, 0, context="jobs list token")
    _assert_clean(jobs, token)


def test_config_show_does_not_include_token(qasr):
    token = "{{SECRET_g2b6n4ye}}"
    proc = qasr.run(
        "config", "show", "--json", env={"QASR_SERVE_TOKEN": token}
    )
    assert_exit(proc, 0, context="config show token")
    _assert_clean(proc, token)
    assert not scan_tree_for_secret(qasr.home, token)


def test_config_show_redacts_server_userinfo(qasr):
    secret = "{{SECRET_h3t6b9ym}}"
    proc = qasr.run(
        "--server", f"http://user:{secret}@127.0.0.1:9",
        "config", "show", "--json",
    )
    assert_exit(proc, 0, context="config show userinfo")
    assert not scan_text_for_secret(proc.stdout, secret), proc.stdout


def test_serve_dry_run_does_not_print_token(qasr):
    token = "{{SECRET_c8f1u6jd}}"
    proc = qasr.run(
        "serve", "--dry-run", "--json", env={"QASR_SERVE_TOKEN": token}
    )
    assert_exit(proc, 0, context="serve dry-run token")
    _assert_clean(proc, token)


def test_server_url_userinfo_is_not_echoed(qasr, fake_openai, short_wav):
    secret = "{{SECRET_w4s7r2pl}}"
    server = fake_openai()
    url = server.url.replace("http://", f"http://user:{secret}@", 1)
    proc = qasr.run("--server", url, "transcribe", short_wav, "--json", timeout=30)
    assert proc.returncode in {0, 2, 3, 4}, proc.stderr
    _assert_clean(proc, secret)
    assert not scan_tree_for_secret(qasr.home, secret)


def test_cross_origin_redirect_does_not_forward_token(qasr, fake_openai, short_wav):
    token = "{{SECRET_a9d3k5mv}}"
    target = fake_openai(auth_token=token, require_auth=True)

    def redirect(request, srv):
        if request.path.split("?", 1)[0] != "/v1/audio/transcriptions":
            return None
        return Response(
            status=302,
            body=b"",
            headers={"Location": f"{target.url}/v1/audio/transcriptions"},
        )

    source = fake_openai(handler=redirect)
    proc = qasr.run(
        "--server", source.url, "transcribe", short_wav, "--json",
        env={"QASR_SERVE_TOKEN": token}, timeout=30,
    )
    # Following, refusing, or reporting the redirect are all acceptable; sending
    # the bearer token to a different origin is not.
    assert proc.returncode in {0, 3, 4}, proc.stderr
    leaked = [request.authorization for request in target.requests if request.authorization]
    assert not leaked, leaked
    _assert_clean(proc, token)


def test_401_body_echo_is_truncated_and_clean(qasr, fake_openai, short_wav):
    token = "{{SECRET_e6h1y8tb}}"

    def handler(request, srv):
        return error_response(
            "authorization was " + request.authorization + " " + "z" * 50_000,
            status=401,
        )

    server = fake_openai(handler=handler)
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--json", env={"QASR_SERVE_TOKEN": token},
    )
    assert_exit(proc, 4, context="401 body echo")
    _assert_clean(proc, token)
    assert len(proc.stdout) < 20_000
    parse_json(proc)


def test_failed_remote_job_state_has_no_token(
    qasr, fake_openai, tmp_path, short_wav
):
    token = "{{SECRET_v2n4j7qs}}"
    server = fake_openai(
        handler=lambda request, srv: error_response(
            "failure with " + request.authorization, status=500
        )
    )
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--out-dir", out, "--retries", "0", "--json",
        env={"QASR_SERVE_TOKEN": token},
    )
    assert_exit(proc, 3, context="failed job token")
    _assert_clean(proc, token)
    hits = scan_tree_for_secret(out, token)
    assert not hits, hits


def test_manifest_and_json_do_not_store_token(qasr, fake_openai, tmp_path, short_wav):
    token = "{{SECRET_m1q7z3ld}}"
    server = fake_openai(auth_token=token, require_auth=True)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "one", "audio": str(short_wav), "formats": ["txt"]}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", "--manifest", manifest,
        "--out-dir", out, "--json", env={"QASR_SERVE_TOKEN": token},
    )
    assert_exit(proc, 0, context="manifest token")
    _assert_clean(proc, token)
    assert not scan_tree_for_secret(out, token)
    for path in Path(out).rglob("*.json*"):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert token not in text, path
