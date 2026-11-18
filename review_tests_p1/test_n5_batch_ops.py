"""N5: stdin-list, skip-existing, retry, tsv/md, and batch error semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_openai import error_response, json_response
from helpers import assert_exit, parse_json, read_job_index

pytestmark = pytest.mark.n5


def test_usage_error_does_not_register_job(qasr, tmp_path, short_wav):
    out = tmp_path / "out"
    proc = qasr.run(
        "--json", "transcribe", short_wav, "-o", out, "-f", "json",
    )
    assert_exit(proc, 2, context="format/json conflict")
    assert read_job_index(qasr.home) == []


def _empty(path: Path) -> bool:
    return path.stat().st_size == 0


def test_stdin_list_basic_and_blank_lines(qasr, empty_wavs):
    one, two = empty_wavs
    proc = qasr.run(
        "transcribe", "--stdin-list", "--json",
        stdin=f"{one}\n\n{two}\n",
    )
    assert_exit(proc, 0, context="stdin-list")
    payload = parse_json(proc)
    assert payload["summary"]["total"] == 2
    assert [item["input"]["id"] for item in payload["items"]] == [
        "empty-one", "empty-two"
    ]


def test_stdin_list_plus_positionals_orders_stdin_first(qasr, empty_wavs):
    one, two = empty_wavs
    proc = qasr.run(
        "transcribe", "--stdin-list", one, "--json",
        stdin=f"{two}\n{one}\n",
    )
    assert_exit(proc, 0, context="stdin-list + positional")
    payload = parse_json(proc)
    assert [item["input"]["id"] for item in payload["items"]] == [
        "empty-two", "empty-one"
    ]
    assert payload["summary"]["total"] == 2


def test_stdin_dash_is_not_a_path(qasr, empty_wavs):
    one, _two = empty_wavs
    proc = qasr.run(
        "transcribe", "--stdin-list", "--json",
        stdin=f"{one}\n-\n",
    )
    # T5.1: '-' is not a path. The chosen contract (fixed in e47e563) is a
    # usage error rather than silently dropping the line.
    assert_exit(proc, 2, context="stdin dash")
    payload = parse_json(proc)
    error = payload.get("error") or {}
    assert error.get("code") == 2, payload
    assert "not an input path" in str(error.get("message", "")), payload


def test_stdin_list_empty_is_usage_error(qasr):
    proc = qasr.run("transcribe", "--stdin-list", "--json", stdin="")
    assert_exit(proc, 2, context="empty stdin-list")


def test_stdin_list_and_manifest_are_mutually_exclusive(qasr, tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"audio": "a.wav"}\n', encoding="utf-8")
    proc = qasr.run(
        "transcribe", "--stdin-list", "--manifest", manifest,
        "--out-dir", tmp_path / "out", "--json", stdin="a.wav\n",
    )
    assert_exit(proc, 2, context="stdin-list + manifest")


def test_stdin_list_missing_file_is_item_failure(qasr, tmp_path, empty_wavs):
    one, _two = empty_wavs
    proc = qasr.run(
        "transcribe", "--stdin-list", "--json",
        stdin=f"{one}\n{tmp_path / 'missing.wav'}\n",
    )
    assert_exit(proc, 6, context="stdin-list missing")
    payload = parse_json(proc)
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["ok"] == 1
    assert payload["summary"]["failed"] == 1


def _remote_output(qasr, fake_openai, audio, out) -> Path:
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>已有文本"}
    )
    proc = qasr.run(
        "--server", server.url, "transcribe", audio, "--out-dir", out, "--json"
    )
    assert_exit(proc, 0, context="seed output")
    parse_json(proc)
    files = [
        path for path in Path(out).rglob("*")
        if path.is_file() and ".qasr" not in path.parts
    ]
    assert len(files) == 1, files
    return files[0]


def test_skip_existing_does_not_load_model(qasr, fake_openai, tmp_path, short_wav):
    out = tmp_path / "out"
    output = _remote_output(qasr, fake_openai, short_wav, out)
    before = output.read_text(encoding="utf-8")
    proc = qasr.run(
        "transcribe", short_wav, "--out-dir", out, "--skip-existing",
        "--model", tmp_path / "does-not-exist", "--json",
    )
    assert_exit(proc, 0, context="skip-existing")
    payload = parse_json(proc)
    # One successful/skipped input uses the item envelope, not a batch summary.
    assert payload["ok"] is True, payload
    assert payload.get("skipped") is True, payload
    assert (payload.get("result") or {}).get("skipped") is True, payload
    assert "model loaded" not in proc.stderr.lower(), proc.stderr
    assert output.read_text(encoding="utf-8") == before


def test_skip_existing_requires_non_empty_output(
    qasr, fake_openai, tmp_path, short_wav
):
    out = tmp_path / "out"
    output = _remote_output(qasr, fake_openai, short_wav, out)
    output.write_text("", encoding="utf-8")
    proc = qasr.run(
        "transcribe", short_wav, "--out-dir", out, "--skip-existing",
        "--model", tmp_path / "does-not-exist", "--json",
    )
    assert_exit(proc, 3, context="empty existing output is not skipped")
    assert _empty(output), "model failure must preserve the existing empty output"


def test_skip_existing_and_resume_can_be_combined(
    qasr, fake_openai, tmp_path, short_wav
):
    out = tmp_path / "out"
    _remote_output(qasr, fake_openai, short_wav, out)
    proc = qasr.run(
        "transcribe", short_wav, "--out-dir", out,
        "--skip-existing", "--resume", "--model", tmp_path / "missing", "--json",
    )
    assert_exit(proc, 0, context="skip-existing + resume")
    payload = parse_json(proc)
    assert payload.get("skipped") is True, payload
    assert "model loaded" not in proc.stderr.lower(), proc.stderr


def test_retry_reruns_retryable_failure_once(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai()
    calls = {"n": 0}

    def handler(request, srv):
        calls["n"] += 1
        if calls["n"] <= 2:
            return error_response("temporary", status=503)
        return json_response({"text": "language Chinese<asr_text>最终成功"})

    server.handler = handler
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--retries", "1", "--retry", "--out-dir", out, "--json", timeout=60,
    )
    assert_exit(proc, 0, context="--retry success")
    assert calls["n"] == 3, calls
    payload = parse_json(proc)
    # A single successful input uses the item envelope, not the batch summary.
    assert payload["ok"] is True, payload
    assert payload["result"]["text"] == "最终成功", payload
    assert payload["attempts"] == 2, payload


def test_retry_does_not_retry_non_retryable(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai()
    calls = {"n": 0}

    def handler(request, srv):
        calls["n"] += 1
        if calls["n"] == 1:
            return error_response("bad request", status=400)
        return json_response({"text": "language Chinese<asr_text>不应发生"})

    server.handler = handler
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--retries", "1", "--retry", "--out-dir", out, "--json", timeout=60,
    )
    assert_exit(proc, 4, context="--retry non-retryable")
    assert calls["n"] == 1, calls


def test_retry_and_fail_fast_are_mutually_exclusive(qasr, short_wav, tmp_path):
    proc = qasr.run(
        "transcribe", short_wav, "--retry", "--fail-fast",
        "--out-dir", tmp_path / "out", "--json",
    )
    assert_exit(proc, 2, context="--retry + --fail-fast")


def test_partial_local_failure_continues_and_exits_6(qasr, tmp_path, empty_wavs):
    one, _two = empty_wavs
    missing = tmp_path / "missing.wav"
    out = tmp_path / "out"
    proc = qasr.run(
        "transcribe", one, missing, "--out-dir", out, "--json"
    )
    assert_exit(proc, 6, context="continue-on-error default")
    payload = parse_json(proc)
    assert payload["summary"] == {"total": 2, "ok": 1, "failed": 1, "skipped": 0}


def test_tsv_output_escapes_tabs_and_newlines(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>hello\tworld\nnext line"}
    )
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "-f", "tsv", "--output", "-",
    )
    assert_exit(proc, 0, context="tsv output")
    lines = proc.stdout.splitlines()
    assert lines[0] == "id\tstart\tend\ttext\tlanguage"
    assert len(lines) == 2, proc.stdout
    data = lines[1]
    assert data.count("\t") == 4, data
    assert "hello world next line" in data, data
    assert data.startswith("short\t"), data


def test_md_output_has_item_heading(qasr, fake_openai, short_wav):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>第一段。第二段。"}
    )
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "-f", "md", "--output", "-",
    )
    assert_exit(proc, 0, context="md output")
    assert "## short" in proc.stdout, proc.stdout
    assert "第一段。第二段。" in proc.stdout


def test_output_accepts_single_input_only(qasr, tmp_path, empty_wavs):
    one, two = empty_wavs
    proc = qasr.run(
        "transcribe", one, two, "--output", tmp_path / "one.txt", "--json"
    )
    assert_exit(proc, 2, context="--output with two inputs")


def test_json_format_and_json_flag_stay_mutually_exclusive(qasr, short_wav):
    proc = qasr.run(
        "transcribe", short_wav, "-f", "json", "--json",
    )
    assert_exit(proc, 2, context="-f json + --json")


@pytest.mark.parametrize("fmt,flag", [("tsv", "--json"), ("md", "--jsonl")])
def test_tsv_md_are_mutually_exclusive_with_json_flags(
    qasr, fake_openai, short_wav, fmt, flag
):
    server = fake_openai()
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "-f", fmt, flag,
    )
    assert_exit(proc, 2, context=f"-f {fmt} + {flag}")
    assert not server.post_requests()


def test_tsv_language_column_uses_result_language(
    qasr, fake_openai, short_wav
):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>你好"}
    )
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "-f", "tsv", "--output", "-",
    )
    assert_exit(proc, 0, context="tsv language")
    header, row = proc.stdout.splitlines()[:2]
    assert header == "id\tstart\tend\ttext\tlanguage"
    assert row.endswith("\tChinese"), (header, row)
