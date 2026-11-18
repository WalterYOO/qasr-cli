"""P2 S1–S3 real-vLLM acceptance (gpu06); skipped unless RUN_GPU=1.

Requires ``QASR_TEST_MODEL_DIR`` (containing Qwen3-ASR-0.6B or 1.7B) and
``QASR_TEST_AUDIO`` (a short 16 kHz WAV). These tests are the independent
evidence path for the stream success contract; the local ``simulated``
harness is only supporting coverage.
"""

from __future__ import annotations

import json
import os
import select
import signal
import wave
from pathlib import Path

import pytest
from helpers import assert_exit, assert_no_traceback, parse_jsonl

pytestmark = [pytest.mark.stream, pytest.mark.gpu]


def _model(qasr_env) -> str:
    root = os.environ.get("QASR_TEST_MODEL_DIR")
    if not root:
        pytest.skip("QASR_TEST_MODEL_DIR is not set")
    for name in ("Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B"):
        candidate = Path(root) / name
        if candidate.is_dir():
            return str(candidate)
    pytest.skip(f"no Qwen3-ASR model under {root}")


def _audio() -> Path:
    value = os.environ.get("QASR_TEST_AUDIO")
    if not value:
        pytest.skip("QASR_TEST_AUDIO is not set")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"QASR_TEST_AUDIO does not exist: {path}")
    return path


def _wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16_000
        return handle.readframes(handle.getnframes())


def test_gpu_file_stream_emits_valid_cumulative_jsonl(qasr):
    model, audio = _model(qasr), _audio()
    proc = qasr.run(
        "--model", model,
        "stream", "--backend", "vllm", "--input", audio,
        "--chunk-ms", "2000", "--emit", "jsonl",
        timeout=600,
    )
    assert_exit(proc, 0, context="gpu file stream")
    values = parse_jsonl(proc)
    assert [item["seq"] for item in values] == list(range(len(values)))
    assert sum(1 for item in values if item["is_final"]) == 1
    assert values[-1]["is_final"] is True
    assert values[-1]["text"].strip()
    texts = [item["text"] for item in values]
    for before, after in zip(texts, texts[1:]):
        # Cumulative snapshots may revise the tail but never become a delta.
        common = os.path.commonprefix([before, after])
        assert len(common) >= max(0, len(before) - 8), (before, after)
    assert_no_traceback(proc)


def test_gpu_stdin_pcm_matches_file_semantics(qasr):
    model, audio = _model(qasr), _audio()
    proc = qasr.run(
        "--model", model,
        "stream", "--backend", "vllm", "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "2000", "--emit", "jsonl",
        stdin=_wav_pcm(audio),
        timeout=600,
    )
    assert_exit(proc, 0, context="gpu stdin stream")
    values = parse_jsonl(proc)
    assert values[-1]["is_final"] is True
    assert values[-1]["text"].strip()


def test_gpu_max_stream_seconds_exits_5_with_quality_final(qasr):
    model, audio = _model(qasr), _audio()
    proc = qasr.run(
        "--model", model,
        "stream", "--backend", "vllm", "--input", audio,
        "--chunk-ms", "2000", "--max-stream-seconds", "1",
        "--emit", "jsonl",
        timeout=600,
    )
    assert_exit(proc, 5, context="gpu cap")
    values = parse_jsonl(proc)
    finals = [item for item in values if item["is_final"]]
    assert len(finals) == 1, values
    assert finals[0]["ok"] is False
    assert finals[0]["error"]["kind"] == "quality"


def test_gpu_sigint_flushes_final_snapshot(qasr):
    model, audio = _model(qasr), _audio()
    proc = qasr.popen(
        "--model", model,
        "stream", "--backend", "vllm", "--input", audio,
        "--chunk-ms", "1000", "--emit", "jsonl",
    )
    assert proc.stdout is not None
    try:
        readable, _, _ = select.select([proc.stdout], [], [], 300)
        assert readable, "no snapshot before SIGINT"
        first = proc.stdout.readline()
        assert json.loads(first)["ok"] is True
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=300)
    except Exception:
        proc.kill()
        raise
    assert proc.returncode == 130, (proc.returncode, stdout, stderr)
    values = parse_jsonl(
        type("P", (), {"stdout": first + stdout, "stderr": stderr,
                       "returncode": proc.returncode})()
    )
    assert values[-1]["is_final"] is True


@pytest.mark.parametrize(
    "flags",
    [
        ("--backend", "transformers"),
        ("--server", "http://127.0.0.1:9"),
    ],
)
def test_gpu_stream_rejects_non_vllm_backends(qasr, flags):
    model, audio = _model(qasr), _audio()
    proc = qasr.run(
        "--model", model, "stream", "--input", audio, *flags, timeout=120,
    )
    assert_exit(proc, 4, context=f"gpu stream {flags}")
    assert_no_traceback(proc)
