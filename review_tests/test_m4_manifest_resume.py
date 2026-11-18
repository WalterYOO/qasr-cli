"""MAN-01..24: manifest schema, path safety, lock, and state bookkeeping."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from helpers import (
    assert_error,
    assert_exit,
    empty_wav,
    find_values,
    make_wav,
    parse_json,
    write_manifest,
    zero_byte,
)


def _manifest(qasr, records, name="jobs.jsonl") -> Path:
    return write_manifest(qasr.work / name, records)


def _dry_run_manifest(qasr, manifest, out=None, *extra):
    out = out or (qasr.work / "out")
    return qasr.run(
        "--json",
        "--dry-run",
        "transcribe",
        "--manifest",
        manifest,
        "-o",
        out,
        *extra,
    )


def _inputs(payload):
    plan = payload.get("plan")
    assert isinstance(plan, dict), f"missing plan: {payload!r}"
    inputs = plan.get("inputs")
    assert isinstance(inputs, list), f"missing plan.inputs: {payload!r}"
    return inputs


def test_valid_manifest_dry_run(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    b = make_wav(qasr.work / "b.wav", seconds=0.2)
    manifest = _manifest(
        qasr,
        [
            {"id": "a1", "audio": str(a), "formats": ["txt"]},
            {"id": "b1", "audio": str(b), "formats": ["json"]},
        ],
    )
    proc = _dry_run_manifest(qasr, manifest)
    assert_exit(proc, 0, context="manifest dry-run")
    payload = parse_json(proc)
    inputs = _inputs(payload)
    assert len(inputs) == 2
    ids = {value for value in find_values(payload, "id") if isinstance(value, str)}
    assert {"a1", "b1"} <= ids, f"manifest ids were not preserved: {payload!r}"


def test_duplicate_ids_are_rejected(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(
        qasr,
        [{"id": "same", "audio": str(a)}, {"id": "same", "audio": str(a)}],
    )
    proc = _dry_run_manifest(qasr, manifest)
    assert_exit(proc, 2, context="duplicate manifest id")
    error = assert_error(parse_json(proc), code=2)
    assert "same" in json.dumps(error).lower()


def test_ids_that_sanitize_to_same_output_are_rejected(qasr):
    """Two distinct raw ids must not share one output file, even with --overwrite."""
    audio = empty_wav(qasr.work / "empty.wav")
    manifest = _manifest(
        qasr,
        [
            {"id": "a/b", "audio": str(audio), "formats": ["txt"]},
            {"id": "a_b", "audio": str(audio), "formats": ["txt"]},
        ],
    )
    proc = qasr.run(
        "--json",
        "transcribe",
        "--manifest",
        manifest,
        "-o",
        qasr.work / "out",
        "--overwrite",
    )
    assert_exit(proc, 2, context="sanitized id collision")
    assert_error(parse_json(proc), code=2)


def test_ids_that_sanitize_to_same_output_are_rejected_without_overwrite(qasr):
    """The collision must be a preflight usage error even without --overwrite."""
    audio = empty_wav(qasr.work / "empty.wav")
    manifest = _manifest(
        qasr,
        [
            {"id": "a/b", "audio": str(audio), "formats": ["txt"]},
            {"id": "a_b", "audio": str(audio), "formats": ["txt"]},
        ],
    )
    proc = qasr.run(
        "--json",
        "transcribe",
        "--manifest",
        manifest,
        "-o",
        qasr.work / "out",
    )
    assert_exit(proc, 2, context="sanitized id collision without overwrite")
    assert_error(parse_json(proc), code=2)


def test_unknown_field_is_rejected(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(qasr, [{"id": "a", "audio": str(a), "mystery": 1}])
    proc = _dry_run_manifest(qasr, manifest)
    assert_exit(proc, 2, context="unknown manifest field")
    assert "mystery" in json.dumps(parse_json(proc)).lower()


def test_audio_and_base64_are_mutually_exclusive(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(
        qasr,
        [{"id": "a", "audio": str(a), "audio_base64": "ZGF0YQ=="}],
    )
    proc = _dry_run_manifest(qasr, manifest)
    assert_exit(proc, 2, context="audio + audio_base64")


def test_audio_or_base64_is_required(qasr):
    manifest = _manifest(qasr, [{"id": "a"}])
    proc = _dry_run_manifest(qasr, manifest)
    assert_exit(proc, 2, context="manifest without audio")


@pytest.mark.parametrize(
    "output",
    ["../escape.txt", "/tmp/escape.txt", "nested/../../escape.txt", "bad\u0000name.txt"],
)
def test_output_path_traversal_is_rejected(qasr, output):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(qasr, [{"id": "a", "audio": str(a), "output": output}])
    proc = _dry_run_manifest(qasr, manifest)
    assert_exit(proc, 2, context=f"output traversal {output!r}")
    assert_error(parse_json(proc), code=2)


def test_formats_must_be_a_list(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(qasr, [{"id": "a", "audio": str(a), "formats": "txt"}])
    proc = _dry_run_manifest(qasr, manifest)
    assert_exit(proc, 2, context="manifest formats as string")


def test_p1_format_in_manifest_is_rejected(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(qasr, [{"id": "a", "audio": str(a), "formats": ["tsv"]}])
    proc = _dry_run_manifest(qasr, manifest)
    assert_exit(proc, 2, context="manifest formats tsv")


def test_manifest_timestamps_auto_is_capability_error(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(
        qasr,
        [{"id": "a", "audio": str(a), "timestamps": True, "language": "auto"}],
    )
    proc = _dry_run_manifest(qasr, manifest)
    assert_exit(proc, 4, context="manifest timestamps + auto")
    assert_error(parse_json(proc), code=4)


def test_manifest_and_positional_inputs_are_mutually_exclusive(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(qasr, [{"id": "a", "audio": str(a)}])
    proc = qasr.run(
        "--json",
        "--dry-run",
        "transcribe",
        "--manifest",
        manifest,
        a,
        "-o",
        qasr.work / "out",
    )
    assert_exit(proc, 2, context="manifest + positional")


def test_manifest_all_failures_use_batch_envelope(qasr):
    bad1 = zero_byte(qasr.work / "bad1.wav")
    bad2 = zero_byte(qasr.work / "bad2.wav")
    manifest = _manifest(qasr, [{"id": "b1", "audio": str(bad1)}, {"id": "b2", "audio": str(bad2)}])
    proc = qasr.run("--json", "transcribe", "--manifest", manifest, "-o", qasr.work / "out")
    assert_exit(proc, 4, context="manifest all failed")
    payload = parse_json(proc)
    assert payload.get("ok") is False
    summary = payload.get("summary")
    assert isinstance(summary, dict), f"manifest batch should expose summary: {payload!r}"
    assert summary.get("total") == 2
    assert summary.get("failed") == 2
    assert len(payload.get("errors", [])) == 2


def test_active_lock_blocks_second_run(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(qasr, [{"id": "a", "audio": str(audio)}])
    out = qasr.work / "out"
    (out / ".qasr").mkdir(parents=True)
    (out / ".qasr" / "lock").write_text(str(os.getpid()), encoding="utf-8")

    proc = qasr.run("--json", "transcribe", "--manifest", manifest, "-o", out, timeout=90)
    assert_exit(proc, 2, context="active lock")
    error = assert_error(parse_json(proc), code=2)
    assert "lock" in json.dumps(error).lower() or "another" in json.dumps(error).lower()


def test_stale_lock_is_recovered(qasr):
    audio = zero_byte(qasr.work / "bad.wav")
    manifest = _manifest(qasr, [{"id": "a", "audio": str(audio)}])
    out = qasr.work / "out"
    (out / ".qasr").mkdir(parents=True)
    stale = _dead_pid()
    (out / ".qasr" / "lock").write_text(str(stale), encoding="utf-8")

    proc = qasr.run("--json", "transcribe", "--manifest", manifest, "-o", out, timeout=90)
    assert proc.returncode != 2, (
        "stale lock from a dead PID must be recovered, not treated as active:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert_exit(proc, {3, 4, 6}, context="stale lock then corrupt input")


def test_manifest_without_out_dir_is_usage_error(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(qasr, [{"id": "a", "audio": str(a)}])
    proc = qasr.run("--json", "transcribe", "--manifest", manifest)
    assert_exit(proc, 2, context="manifest without out-dir")


def test_missing_manifest_is_input_error(qasr):
    proc = qasr.run("--json", "transcribe", "--manifest", qasr.work / "nope.jsonl", "-o", qasr.work / "out")
    assert_exit(proc, 4, context="missing manifest")
    assert_error(parse_json(proc), code=4)


def test_bad_manifest_line_reports_line_number(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    path = qasr.work / "jobs.jsonl"
    path.write_text(
        json.dumps({"id": "a", "audio": str(a)}) + "\n" + "{broken json\n",
        encoding="utf-8",
    )
    proc = _dry_run_manifest(qasr, path)
    assert_exit(proc, 2, context="bad manifest line")
    text = json.dumps(parse_json(proc), ensure_ascii=False)
    lower = text.lower()
    assert (
        "line 2" in lower
        or "line=2" in lower
        or "第 2 行" in text
        or "第2行" in text
    ), f"error does not mention the offending line: {text[:1000]}"


def test_invalid_base64_is_rejected(qasr):
    manifest = _manifest(
        qasr,
        [{"id": "b64", "audio_base64": "!!!definitely-not-base64!!!"}],
    )
    proc = qasr.run("--json", "transcribe", "--manifest", manifest, "-o", qasr.work / "out")
    assert_exit(proc, {2, 4}, context="invalid base64")
    assert_error(parse_json(proc))


def test_valid_base64_reaches_model_resolution(qasr):
    import base64

    audio = make_wav(qasr.work / "a.wav", seconds=0.2)
    encoded = base64.b64encode(audio.read_bytes()).decode("ascii")
    manifest = _manifest(qasr, [{"id": "b64", "audio_base64": encoded}])
    proc = qasr.run(
        "--json",
        "--model",
        "/nonexistent/model",
        "transcribe",
        "--manifest",
        manifest,
        "-o",
        qasr.work / "out",
    )
    assert_exit(proc, 3, context="valid base64 accepted")


def test_resume_without_out_dir_is_usage_error(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    manifest = _manifest(qasr, [{"id": "a", "audio": str(a)}])
    proc = qasr.run("--json", "transcribe", "--manifest", manifest, "--resume")
    assert_exit(proc, 2, context="--resume without out-dir")


def test_state_jsonl_is_complete_after_failure(qasr):
    bad = zero_byte(qasr.work / "bad.wav")
    manifest = _manifest(qasr, [{"id": "bad", "audio": str(bad)}])
    out = qasr.work / "out"
    proc = qasr.run("--json", "transcribe", "--manifest", manifest, "-o", out, "--resume")
    assert_exit(proc, {4, 6}, context="state after failure")

    state = out / ".qasr" / "state.jsonl"
    if not state.exists():
        pytest.fail(f"missing state file after a manifest run: {state}")
    records = [json.loads(line) for line in state.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records, "state.jsonl is empty"
    assert any(record.get("id") == "bad" for record in records), records
    assert all(record.get("status") in {"pending", "running", "done", "failed", "skipped"} for record in records)
    assert any(record.get("status") == "failed" for record in records), records


def _dead_pid() -> int:
    pid = 999_999
    while pid > 100_000:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except PermissionError:
            pass
        pid -= 1
    pytest.skip("could not find an unused PID for the stale-lock test")
