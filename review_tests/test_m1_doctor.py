"""DOC-01..06: doctor JSON contract, exit codes, and missing-runtime guidance."""

from __future__ import annotations

import json

from helpers import assert_exit, parse_json


def _doctor(qasr, *args, env=None):
    proc = qasr.run("--json", "doctor", *args, env=env, timeout=120)
    return proc, parse_json(proc)


def test_doctor_json_shape(qasr):
    proc, payload = _doctor(qasr)
    assert_exit(proc, 0, context="doctor --json")
    assert payload.get("schema_version") == 1
    assert isinstance(payload.get("ok"), bool)
    assert payload.get("status") in {"ok", "warn", "fail"}
    checks = payload.get("checks")
    assert isinstance(checks, list) and checks, f"doctor has no checks: {payload!r}"
    for check in checks:
        assert isinstance(check, dict), f"check is not an object: {check!r}"
        assert check.get("name"), f"check has no name: {check!r}"
        assert check.get("status") in {"ok", "warn", "fail"}, check
        assert "value" in check, f"check has no value: {check!r}"


def test_doctor_default_exit_is_zero_even_with_warnings(qasr):
    proc, payload = _doctor(qasr)
    assert_exit(proc, 0, context="doctor default exit")
    assert payload.get("status") in {"ok", "warn", "fail"}


def test_doctor_reports_missing_runtime_under_blocker(qasr, import_blocker):
    proc, payload = _doctor(qasr, env=import_blocker.env())
    assert_exit(proc, 0, context="doctor --json without runtime")
    assert payload.get("status") in {"warn", "fail"}, f"blocked runtime reported as {payload.get('status')}"

    hints = " ".join(
        str(check.get("hint", ""))
        for check in payload.get("checks", [])
        if isinstance(check, dict)
    ).lower()
    messages = json.dumps(payload, ensure_ascii=False).lower()
    assert (
        "qasr-cli[local]" in messages
        or "uv tool install" in messages
        or "pip install" in messages
    ), f"doctor gives no install hint: {hints or messages[:2000]}"


def test_doctor_strict_with_missing_runtime_exits_3(qasr, import_blocker):
    proc, payload = _doctor(qasr, "--strict", env=import_blocker.env())
    assert_exit(proc, 3, context="doctor --strict without runtime")
    assert payload.get("status") in {"warn", "fail"}


def test_doctor_server_flag_is_available(qasr):
    proc = qasr.run("--json", "doctor", "--server", "http://127.0.0.1:9")
    assert_exit(proc, 0, context="doctor --server")
    payload = json.loads(proc.stdout)
    server_checks = [c for c in payload["checks"] if c["name"] == "server"]
    assert len(server_checks) == 1, payload


def test_doctor_stdout_is_pure_json(qasr):
    proc = qasr.run("--json", "doctor")
    assert_exit(proc, 0)
    payload = json.loads(proc.stdout)
    assert payload.get("schema_version") == 1
