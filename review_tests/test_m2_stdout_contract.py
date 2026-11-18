"""PUR-01..09: stdout purity, error envelopes, and JSONL batch lines."""

from __future__ import annotations

import json

from helpers import assert_error, assert_exit, empty_wav, parse_json, parse_jsonl, zero_byte


def test_languages_stdout_is_one_json_value(qasr):
    proc = qasr.run("--json", "languages")
    assert_exit(proc, 0)
    payload = parse_json(proc)
    assert payload.get("schema_version") == 1
    assert payload.get("ok") is True


def test_error_envelope_has_all_contract_fields(qasr):
    proc = qasr.run("--json", "transcribe", "missing.wav")
    assert_exit(proc, 4, context="missing input error envelope")
    payload = parse_json(proc)
    error = assert_error(payload, code=4)
    assert payload.get("schema_version") == 1
    assert payload.get("command") == "transcribe"
    for key in ("kind", "code", "message", "hint", "retryable", "details"):
        assert key in error, f"error envelope is missing {key!r}: {error!r}"
    assert isinstance(error["code"], int)
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error["retryable"], bool)
    assert isinstance(error["details"], dict)


def test_jsonl_all_failed_has_one_line_per_item(qasr):
    bad1 = zero_byte(qasr.work / "bad1.wav")
    bad2 = zero_byte(qasr.work / "bad2.wav")
    out = qasr.work / "out"
    proc = qasr.run("--jsonl", "transcribe", bad1, bad2, "-o", out)
    assert_exit(proc, 4, context="all inputs failed")
    lines = parse_jsonl(proc)
    assert len(lines) == 2, f"expected one line per item, got {len(lines)}: {lines!r}"
    for line in lines:
        assert line.get("ok") is False
        assert "error" in line
    assert not any(line.get("summary") for line in lines), "JSONL must not append a summary row"


def test_jsonl_single_failure(qasr):
    bad = zero_byte(qasr.work / "bad.wav")
    proc = qasr.run("--jsonl", "transcribe", bad, "-o", qasr.work / "out")
    assert_exit(proc, 4)
    lines = parse_jsonl(proc)
    assert len(lines) == 1
    assert lines[0].get("ok") is False


def test_json_batch_envelope_all_failed(qasr):
    bad1 = zero_byte(qasr.work / "bad1.wav")
    bad2 = zero_byte(qasr.work / "bad2.wav")
    proc = qasr.run("--json", "transcribe", bad1, bad2, "-o", qasr.work / "out")
    assert_exit(proc, 4, context="all inputs failed, first error wins")
    payload = parse_json(proc)
    assert payload.get("ok") is False
    summary = payload.get("summary")
    assert isinstance(summary, dict), f"batch JSON missing summary: {payload!r}"
    assert summary.get("total") == 2
    assert summary.get("failed") == 2
    assert isinstance(payload.get("items"), list)
    assert isinstance(payload.get("errors"), list)
    assert len(payload["errors"]) == 2


def test_help_writes_nothing_to_stderr(qasr):
    proc = qasr.run("--help")
    assert_exit(proc, 0)
    assert proc.stderr == ""


def test_argparse_error_with_json_is_machine_readable(qasr):
    proc = qasr.run("--json", "transcribe")
    assert_exit(proc, 2)
    payload = json.loads(proc.stdout)
    assert_error(payload, code=2, kind="usage")


def test_models_list_stdout_is_pure_json(qasr):
    proc = qasr.run("--json", "models", "list")
    assert_exit(proc, 0)
    payload = json.loads(proc.stdout)
    assert payload.get("ok") is True


def test_config_path_prints_one_path(qasr):
    proc = qasr.run("config", "path")
    assert_exit(proc, 0)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].endswith("config.json")


def test_jsonl_flag_after_subcommand(qasr):
    bad = zero_byte(qasr.work / "bad.wav")
    proc = qasr.run("transcribe", "--jsonl", bad, "-o", qasr.work / "out")
    assert_exit(proc, 4)
    assert len(parse_jsonl(proc)) == 1


def test_multiple_formats_to_stdout_are_usage_error(qasr):
    audio = empty_wav(qasr.work / "empty.wav")
    proc = qasr.run("transcribe", audio, "-f", "txt,json")
    assert_exit(proc, 2, context="multiple formats without --out-dir/--output")
    text = (proc.stdout + proc.stderr).lower()
    assert "format" in text or "out-dir" in text or "output" in text, (
        f"usage error did not explain the output channel: {text!r}"
    )
