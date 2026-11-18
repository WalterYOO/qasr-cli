"""P2 S1: stream CLI surface, defaults, mutual exclusions and capability errors."""

from __future__ import annotations

import pytest
from helpers import assert_error, assert_exit, assert_no_traceback, parse_json

pytestmark = pytest.mark.stream


def test_stream_requires_input(qasr):
    proc = qasr.run("stream")
    assert_exit(proc, 2, context="stream without --input")
    assert_no_traceback(proc)


def test_stream_input_requires_value(qasr):
    proc = qasr.run("stream", "--input")
    assert_exit(proc, 2, context="--input without value")
    assert_no_traceback(proc)


def test_stdin_requires_explicit_s16le_pcm(qasr):
    proc = qasr.run("stream", "--input", "stdin")
    assert_exit(proc, 2, context="stdin without --pcm")
    assert "pcm" in (proc.stdout + proc.stderr).lower()
    assert_no_traceback(proc)


@pytest.mark.parametrize("value", ["s16be", "f32le", "u8", ""])
def test_pcm_only_accepts_s16le(qasr, value):
    proc = qasr.run("stream", "--input", "stdin", "--pcm", value)
    assert_exit(proc, 2, context=f"--pcm {value!r}")
    assert_no_traceback(proc)


@pytest.mark.parametrize("rate", ["8000", "44100", "0", "-1", "abc"])
def test_sample_rate_must_be_16000(qasr, rate):
    proc = qasr.run(
        "stream", "--input", "stdin", "--pcm", "s16le",
        "--sample-rate", rate,
    )
    assert_exit(proc, 2, context=f"--sample-rate {rate}")
    assert_no_traceback(proc)


@pytest.mark.parametrize("channels", ["0", "3", "-1", "abc"])
def test_channels_only_accept_1_or_2(qasr, channels):
    proc = qasr.run(
        "stream", "--input", "stdin", "--pcm", "s16le",
        "--channels", channels,
    )
    assert_exit(proc, 2, context=f"--channels {channels}")
    assert_no_traceback(proc)


@pytest.mark.parametrize("value", ["99", "10001", "105", "-1", "abc"])
def test_chunk_ms_range_and_10ms_multiple(qasr, value):
    proc = qasr.run(
        "stream", "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", value,
    )
    assert_exit(proc, 2, context=f"--chunk-ms {value}")
    assert_no_traceback(proc)


@pytest.mark.parametrize("flag", ["--unfixed-chunks", "--unfixed-tokens"])
@pytest.mark.parametrize("value", ["-1", "abc"])
def test_unfixed_values_are_non_negative(qasr, flag, value):
    proc = qasr.run(
        "stream", "--input", "stdin", "--pcm", "s16le", flag, value,
    )
    assert_exit(proc, 2, context=f"{flag} {value}")
    assert_no_traceback(proc)


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_max_stream_seconds_must_be_positive(qasr, value):
    proc = qasr.run(
        "stream", "--input", "stdin", "--pcm", "s16le",
        "--max-stream-seconds", value,
    )
    assert_exit(proc, 2, context=f"--max-stream-seconds {value}")
    assert_no_traceback(proc)


@pytest.mark.parametrize("value", ["yaml", "JSON", ""])
def test_emit_only_accepts_text_or_jsonl(qasr, value):
    proc = qasr.run(
        "stream", "--input", "stdin", "--pcm", "s16le", "--emit", value,
    )
    assert_exit(proc, 2, context=f"--emit {value!r}")
    assert_no_traceback(proc)


def test_realtime_is_rejected_for_stdin(qasr):
    proc = qasr.run(
        "stream", "--input", "stdin", "--pcm", "s16le", "--realtime",
    )
    assert_exit(proc, 2, context="realtime + stdin")
    assert_no_traceback(proc)


@pytest.mark.parametrize(
    "value",
    ["subdir", "https://example.invalid/a.wav", "*.wav"],
)
def test_bad_input_forms_are_input_errors(
    qasr, fake_stream, fake_model, tmp_path, value
):
    """P2-Q7: directory / glob / URL are input errors (4), not usage."""
    (qasr.work / "subdir").mkdir(exist_ok=True)
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", value,
        env=fake_stream.env(),
    )
    assert_exit(proc, 4, context=f"--input {value}")
    assert_no_traceback(proc)


def test_dash_input_is_usage_error(qasr):
    """P2-Q7: '-' is reserved; hint points at --input stdin."""
    proc = qasr.run("stream", "--input", "-")
    assert_exit(proc, 2, context="--input -")
    assert "stdin" in (proc.stdout + proc.stderr).lower()
    assert_no_traceback(proc)


def test_repeated_input_uses_last_value(qasr, fake_stream, fake_model, short_wav):
    """P2-Q7: a repeated --input is argparse last-wins, not an error."""
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm",
        "--input", short_wav, "--input", qasr.work / "missing.wav",
        "--emit", "jsonl",
        env=fake_stream.env(),
    )
    assert_exit(proc, 4, context="last --input wins")
    assert_no_traceback(proc)


@pytest.mark.parametrize(
    "extra",
    [
        ("--pcm", "s16le"),
        ("--sample-rate", "16000"),
        ("--channels", "1"),
    ],
)
def test_stdin_only_flags_are_rejected_for_file(
    qasr, fake_stream, fake_model, short_wav, extra
):
    """P2-Q8: --pcm/--sample-rate/--channels only apply to stdin."""
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", short_wav, *extra,
        env=fake_stream.env(),
    )
    assert_exit(proc, 2, context=f"file + {extra[0]}")
    assert_no_traceback(proc)


@pytest.mark.parametrize(
    "extra",
    [
        ("--pcm", "s16le"),
        ("--sample-rate", "16000"),
        ("--channels", "1"),
    ],
)
def test_stdin_only_flags_are_rejected_for_mic(
    qasr, fake_stream, fake_model, extra
):
    """P2-Q8: same rule for microphone input."""
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", "mic", *extra,
        env=fake_stream.env(),
    )
    assert_exit(proc, 2, context=f"mic + {extra[0]}")
    assert_no_traceback(proc)


@pytest.mark.parametrize("flags", [
    ("--backend", "transformers"),
    ("--backend", "remote"),
    ("--server", "http://127.0.0.1:9"),
])
def test_stream_rejects_non_vllm_backends(qasr, short_wav, flags):
    proc = qasr.run("stream", "--input", short_wav, *flags)
    assert_exit(proc, 4, context=f"stream {flags}")
    text = (proc.stdout + proc.stderr).lower()
    assert "vllm" in text
    assert_no_traceback(proc)


def test_stream_without_vllm_is_capability_error(qasr, short_wav):
    """P2-Q14: explicit vllm that cannot run here is a capability error (4)."""
    proc = qasr.run("stream", "--input", short_wav, "--backend", "vllm")
    assert_exit(proc, 4, context="stream without vllm")
    assert "vllm" in (proc.stdout + proc.stderr).lower()
    assert_no_traceback(proc)


def test_auto_without_vllm_is_capability_error(qasr, short_wav):
    proc = qasr.run("stream", "--input", short_wav, "--backend", "auto")
    assert_exit(proc, 4, context="stream auto without vllm")
    assert_no_traceback(proc)


@pytest.mark.parametrize(
    "flags",
    [
        ("--backend", "transformers"),
        ("--backend", "remote"),
        ("--server", "http://127.0.0.1:9"),
    ],
)
def test_capability_error_precedes_model_resolution(qasr, short_wav, flags):
    """P2-Q14: capability is decided before the model is resolved (4 not 3)."""
    proc = qasr.run(
        "stream", "--input", short_wav,
        "--model", "/nonexistent/model", *flags,
    )
    assert_exit(proc, 4, context=f"capability before model: {flags}")
    assert_no_traceback(proc)


@pytest.mark.parametrize("language", ["xx", "klingon", "not-a-language"])
def test_invalid_forced_language_is_input_error(qasr, short_wav, language):
    """Same semantics as P0 transcribe (``languages.resolve_forceable`` → 4)."""
    proc = qasr.run("stream", "--input", short_wav, "--language", language)
    assert_exit(proc, 4, context=f"--language {language}")
    assert_no_traceback(proc)


def test_context_and_context_file_are_mutually_exclusive(qasr, short_wav, tmp_path):
    ctx = tmp_path / "context.txt"
    ctx.write_text("背景", encoding="utf-8")
    proc = qasr.run(
        "stream", "--input", short_wav,
        "--context", "a", "--context-file", ctx,
    )
    assert_exit(proc, 2, context="context + context-file")
    assert_no_traceback(proc)


@pytest.mark.parametrize("emit", ["text", "jsonl"])
def test_json_is_unconditionally_incompatible_with_stream(qasr, short_wav, emit):
    """P2-Q2 (522a130): --json x stream is always a usage error, any emit."""
    proc = qasr.run("stream", "--input", short_wav, "--json", "--emit", emit)
    assert_exit(proc, 2, context=f"--json + --emit {emit}")
    assert_error(parse_json(proc), code=2, kind="usage")
    assert_no_traceback(proc)


def test_json_before_subcommand_also_conflicts(qasr, short_wav):
    proc = qasr.run("--json", "stream", "--input", short_wav, "--emit", "jsonl")
    assert_exit(proc, 2, context="--json before stream")
    assert_error(parse_json(proc), code=2, kind="usage")
    assert_no_traceback(proc)


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "abc"])
def test_gpu_memory_utilization_range(qasr, fake_stream, fake_model, short_wav, value):
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", short_wav,
        "--gpu-memory-utilization", value, "--emit", "jsonl",
        env=fake_stream.env(),
    )
    assert_exit(proc, 2, context=f"--gpu-memory-utilization {value}")
    assert_no_traceback(proc)


@pytest.mark.simulated
def test_gpu_memory_utilization_reaches_backend(
    qasr, fake_stream, fake_model, short_wav
):
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", short_wav,
        "--gpu-memory-utilization", "0.3", "--emit", "jsonl",
        env=fake_stream.env(),
    )
    assert_exit(proc, 0, context="gpu util passthrough")
    kwargs = fake_stream.events("model_init")[-1]["kwargs"]
    assert kwargs.get("gpu_memory_utilization") == pytest.approx(0.3)


@pytest.mark.simulated
def test_explicit_cuda_index_maps_to_cuda_visible_devices(
    qasr, fake_stream, fake_model, short_wav
):
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", short_wav,
        "--device", "cuda:3", "--emit", "jsonl",
        env=fake_stream.env(),
    )
    assert_exit(proc, 0, context="cuda index mapping")
    init = fake_stream.events("model_init")[-1]
    assert init.get("cuda_visible_devices") == "3"


def test_missing_file_is_input_error_with_fake_backend(
    qasr, fake_stream, fake_model, tmp_path
):
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", tmp_path / "missing.wav",
        env=fake_stream.env(),
    )
    assert_exit(proc, 4, context="missing stream file")
    assert_no_traceback(proc)


def test_stream_help_does_not_import_heavy_runtimes(qasr, import_blocker):
    blocker, marker = import_blocker
    proc = qasr.run(
        "stream", "--help",
        env={"PYTHONPATH": str(blocker)},
    )
    assert_exit(proc, 0, context="stream --help")
    assert not marker.exists(), marker.read_text(encoding="utf-8")
    assert "stream" in proc.stdout.lower()


def test_stream_help_mentions_core_flags(qasr):
    proc = qasr.run("stream", "--help")
    assert_exit(proc, 0, context="stream --help flags")
    text = proc.stdout
    for flag in (
        "--input", "--pcm", "--sample-rate", "--channels",
        "--chunk-ms", "--unfixed-chunks", "--unfixed-tokens",
        "--max-stream-seconds", "--emit", "--realtime", "--language",
        "--context", "--context-file",
    ):
        assert flag in text, f"{flag} missing from stream --help"
