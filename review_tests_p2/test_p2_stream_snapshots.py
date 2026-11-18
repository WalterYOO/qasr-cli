"""P2 S3/S4: cumulative snapshots, final/error lines, human mode, purity."""

from __future__ import annotations

import select
import signal
import time

import pytest
from helpers import (
    assert_exit,
    assert_no_traceback,
    parse_jsonl,
    pcm_s16le,
    validate_schema,
)

pytestmark = [pytest.mark.stream, pytest.mark.simulated]


def _stream(qasr, fake_stream, fake_model, *args, stdin=None, timeout=90, env=None):
    values = fake_stream.env()
    if env:
        values.update(env)
    return qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm",
        *args,
        env=values,
        stdin=stdin,
        timeout=timeout,
    )


def _pcm(seconds: float) -> bytes:
    return pcm_s16le(int(seconds * 16_000))


def test_each_success_line_matches_stream_schema(
    qasr, fake_stream, fake_model
):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(3),
    )
    assert_exit(proc, 0, context="stream schema")
    values = parse_jsonl(proc)
    for value in values:
        assert value["command"] == "stream"
        validate_schema(value, "stream.json")
    assert values[-1]["is_final"] is True


def test_seq_is_contiguous_and_final_is_last(
    qasr, fake_stream, fake_model
):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(3),
    )
    assert_exit(proc, 0)
    values = parse_jsonl(proc)
    assert [item["seq"] for item in values] == list(range(len(values)))
    assert [item["is_final"] for item in values].count(True) == 1
    assert values[-1]["is_final"] is True


def test_elapsed_s_is_non_decreasing(qasr, fake_stream, fake_model):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(3),
    )
    assert_exit(proc, 0)
    values = parse_jsonl(proc)
    elapsed = [item["elapsed_s"] for item in values]
    assert all(value >= 0 for value in elapsed), elapsed
    assert all(b >= a for a, b in zip(elapsed, elapsed[1:])), elapsed


def test_text_is_cumulative_not_delta(qasr, fake_stream, fake_model):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(3),
    )
    assert_exit(proc, 0)
    values = parse_jsonl(proc)
    texts = [item["text"] for item in values]
    for before, after in zip(texts, texts[1:]):
        assert after.startswith(before), texts


def test_human_non_tty_prints_only_final_text(
    qasr, fake_stream, fake_model
):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "text",
        stdin=_pcm(3),
    )
    assert_exit(proc, 0, context="human text")
    assert "{" not in proc.stdout
    assert proc.stdout.count("\n") == 1, repr(proc.stdout)
    assert proc.stdout.strip()


def test_human_tty_uses_carriage_return(
    qasr, fake_stream, fake_model, five_second_wav, pty_runner
):
    rc, output = pty_runner(
        ["--model", fake_model, "stream", "--backend", "vllm",
         "--input", five_second_wav, "--chunk-ms", "1000", "--emit", "text"],
        env=fake_stream.env(),
    )
    assert rc == 0, output
    assert "\r" in output, repr(output)
    assert output.endswith("\n"), repr(output)
    assert "{" not in output


def test_midstream_failure_appends_one_error_line(
    qasr, fake_stream, fake_model
):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(4),
        env={"QASR_FAKE_STREAM_FAIL": "chunk:2"},
    )
    assert_exit(proc, 3, context="midstream failure")
    values = parse_jsonl(proc)
    assert values[-1]["ok"] is False
    assert values[-1]["is_final"] is True
    assert values[-1]["error"]["code"] == 3
    assert [item["seq"] for item in values] == list(range(len(values)))
    assert sum(1 for item in values if item["is_final"]) == 1
    assert_no_traceback(proc)


def test_failure_before_any_snapshot_uses_plain_envelope(
    qasr, fake_stream, fake_model
):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(4),
        env={"QASR_FAKE_STREAM_FAIL": "chunk:1"},
    )
    assert_exit(proc, 3, context="pre-snapshot failure")
    # P2-Q11/Q15: before the state machine has produced a line, the failure
    # goes through the ordinary error envelope (human text without --json).
    assert proc.stdout.strip() == ""
    assert "error" in proc.stderr.lower()
    assert "stream" in proc.stderr.lower()
    assert_no_traceback(proc)


def test_max_stream_seconds_emits_single_quality_final(
    qasr, fake_stream, fake_model
):
    """P2-Q1: exactly one is_final line, ok=false, error.kind=quality."""
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--max-stream-seconds", "2",
        "--emit", "jsonl",
        stdin=_pcm(5),
    )
    assert_exit(proc, 5, context="max-stream-seconds")
    values = parse_jsonl(proc)
    assert sum(1 for item in values if item["is_final"]) == 1, values
    final = values[-1]
    assert final["is_final"] is True
    assert final["ok"] is False
    assert final["error"]["kind"] == "quality"
    assert final["seq"] == len(values) - 1
    assert final.get("audio_s", 0) <= 2.1
    validate_schema(final, "stream.json")
    assert_no_traceback(proc)


def test_max_stream_seconds_before_first_chunk_still_has_final(
    qasr, fake_stream, fake_model
):
    """P2-Q1: cap smaller than one chunk still emits the ok:false snapshot."""
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--max-stream-seconds", "0.1",
        "--emit", "jsonl",
        stdin=_pcm(2),
    )
    assert_exit(proc, 5, context="cap before first chunk")
    values = parse_jsonl(proc)
    assert len(values) == 1, values
    assert values[0]["is_final"] is True
    assert values[0]["ok"] is False
    assert values[0]["error"]["kind"] == "quality"
    assert values[0]["seq"] == 0
    validate_schema(values[0], "stream.json")


def test_max_stream_seconds_has_no_ok_true_final(
    qasr, fake_stream, fake_model
):
    """P2-Q1: the cap path must not emit an ok:true final before rc 5."""
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--max-stream-seconds", "3",
        "--emit", "jsonl",
        stdin=_pcm(5),
    )
    assert_exit(proc, 5, context="cap exactly at chunk boundary")
    values = parse_jsonl(proc)
    finals = [item for item in values if item["is_final"]]
    assert len(finals) == 1, values
    assert finals[0]["ok"] is False
    assert "quality" in finals[0]["error"]["kind"]


def test_midstream_error_line_seq_is_next_and_schema_valid(
    qasr, fake_stream, fake_model
):
    """P2-Q11: the error line takes the next seq and matches stream.json."""
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(4),
        env={"QASR_FAKE_STREAM_FAIL": "chunk:2"},
    )
    assert_exit(proc, 3, context="midstream schema")
    values = parse_jsonl(proc)
    for item in values[:-1]:
        assert item["ok"] is True
        validate_schema(item, "stream.json")
    error_line = values[-1]
    assert error_line["seq"] == len(values) - 1
    validate_schema(error_line, "stream.json")


def test_finish_stage_failure_appends_error_line(
    qasr, fake_stream, fake_model
):
    """A failure while flushing the tail still terminates the JSONL stream."""
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(1.5),
        env={"QASR_FAKE_STREAM_FAIL": "finish"},
    )
    assert_exit(proc, 3, context="finish-stage failure")
    values = parse_jsonl(proc)
    assert len(values) >= 2, values
    for item in values[:-1]:
        assert item["ok"] is True, item
        validate_schema(item, "stream.json")
    final = values[-1]
    assert final["ok"] is False
    assert final["is_final"] is True
    assert final["seq"] == len(values) - 1
    validate_schema(final, "stream.json")
    assert_no_traceback(proc)


@pytest.mark.parametrize("value", ["nan", "inf", "1e309"])
def test_non_finite_max_stream_seconds_is_usage_error(
    qasr, fake_stream, fake_model, value
):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--max-stream-seconds", value,
        stdin=_pcm(1),
    )
    assert_exit(proc, 2, context="non-finite max-stream-seconds")
    assert "internal error" not in proc.stderr.lower()
    assert_no_traceback(proc)


def test_elapsed_s_includes_model_load(qasr, fake_stream, fake_model):
    started = time.monotonic()
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(1),
        env={"QASR_FAKE_STREAM_INIT_SLEEP": "1.0"},
    )
    wall = time.monotonic() - started
    assert_exit(proc, 0, context="elapsed model load")
    values = parse_jsonl(proc)
    final = values[-1]
    assert final["is_final"] is True
    assert final["elapsed_s"] >= 0.9, (final["elapsed_s"], wall)
    assert final["elapsed_s"] <= wall + 0.5, (final["elapsed_s"], wall)
    assert_no_traceback(proc)


@pytest.mark.slow
def test_sigint_flushes_final_snapshot_and_exits_130(
    qasr, fake_stream, fake_model
):
    proc = qasr.popen(
        "--model", fake_model,
        "stream", "--backend", "vllm",
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        env=fake_stream.env(QASR_FAKE_STREAM_SLEEP="0.4"),
        binary=True,
    )
    assert proc.stdin is not None
    try:
        proc.stdin.write(_pcm(8))
        proc.stdin.flush()
        readable, _, _ = select.select([proc.stdout], [], [], 8)
        assert readable, "no first snapshot before SIGINT"
        first = proc.stdout.readline().decode("utf-8")
        assert first.strip(), first
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=30)
    except Exception:
        proc.kill()
        raise
    stdout = stdout.decode("utf-8", "replace")
    stderr = stderr.decode("utf-8", "replace")
    assert proc.returncode == 130, (proc.returncode, stdout, stderr)
    output = first + stdout
    values = parse_jsonl(
        type("P", (), {"stdout": output, "stderr": stderr,
                       "returncode": proc.returncode})()
    )
    assert values[-1]["is_final"] is True
    assert sum(1 for item in values if item["is_final"]) == 1
    assert_no_traceback(
        type("P", (), {"stdout": output, "stderr": stderr,
                       "returncode": proc.returncode})()
    )


def test_vllm_banner_does_not_pollute_jsonl(
    qasr, fake_stream, fake_model
):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(2),
        env={"QASR_FAKE_STREAM_BANNER": "1"},
    )
    assert_exit(proc, 0, context="banner purity")
    assert "FAKE-VLLM" not in proc.stdout
    values = parse_jsonl(proc)
    assert values[-1]["is_final"] is True


def test_stream_writes_no_state_or_jobs(
    qasr, fake_stream, fake_model
):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(2),
    )
    assert_exit(proc, 0)
    jobs = qasr.home / "jobs"
    assert not jobs.exists() or not list(jobs.iterdir()), list(jobs.iterdir())
    assert not (qasr.work / ".qasr").exists()


def test_stream_does_not_leak_serve_token(
    qasr, fake_stream, fake_model
):
    secret = "p2-secret-token-value"
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(2),
        env={"QASR_SERVE_TOKEN": secret},
    )
    assert_exit(proc, 0)
    assert secret not in proc.stdout
    assert secret not in proc.stderr


def test_context_and_language_reach_backend(
    qasr, fake_stream, fake_model
):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--context", "背景是医院", "--language", "zh",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(2),
    )
    assert_exit(proc, 0)
    init = fake_stream.events("init")[-1]
    assert init["context"] == "背景是医院"
    assert init["language"] in {"zh", "Chinese"}
    values = parse_jsonl(proc)
    assert values[-1]["language"] == "Chinese"


def test_context_file_is_read(qasr, fake_stream, fake_model, tmp_path):
    context = tmp_path / "context.txt"
    context.write_text("来自文件的背景", encoding="utf-8")
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--context-file", context,
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(2),
    )
    assert_exit(proc, 0)
    assert fake_stream.events("init")[-1]["context"] == "来自文件的背景"


def test_missing_context_file_is_input_error(qasr, fake_stream, fake_model, tmp_path):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--context-file", tmp_path / "missing.txt",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(1),
    )
    assert_exit(proc, 4, context="missing context file")
    assert_no_traceback(proc)


def test_context_file_content_not_echoed_on_failure(
    qasr, fake_stream, fake_model, tmp_path
):
    secret = "CONTEXT-SECRET-123"
    context = tmp_path / "context.txt"
    context.write_text(secret, encoding="utf-8")
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--context-file", context,
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=_pcm(2),
        env={"QASR_FAKE_STREAM_FAIL": "chunk:1"},
    )
    assert proc.returncode != 0
    assert secret not in proc.stdout
    assert secret not in proc.stderr


def test_unfixed_and_chunk_flags_reach_backend(
    qasr, fake_stream, fake_model
):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1500", "--unfixed-chunks", "3",
        "--unfixed-tokens", "7", "--emit", "jsonl",
        stdin=_pcm(3),
    )
    assert_exit(proc, 0)
    init = fake_stream.events("init")[-1]
    assert init["chunk_size_sec"] == pytest.approx(1.5)
    assert init["unfixed_chunk_num"] == 3
    assert init["unfixed_token_num"] == 7


def test_silence_language_is_null(qasr, fake_stream, fake_model):
    proc = _stream(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "1000", "--emit", "jsonl",
        stdin=b"",
        env={"QASR_FAKE_STREAM_LANGUAGE": "none"},
    )
    assert_exit(proc, 0)
    assert parse_jsonl(proc)[-1]["language"] is None
