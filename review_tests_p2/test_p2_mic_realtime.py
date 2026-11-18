"""P2 S2: microphone source and --realtime pacing (P2-Q8/Q9)."""

from __future__ import annotations

import json
import select
import signal
import time

import pytest
from helpers import (
    assert_exit,
    assert_no_traceback,
    parse_jsonl,
    pcm_s16le,
    read_jsonl_file,
    write_wav,
)

pytestmark = [pytest.mark.stream, pytest.mark.simulated]


def _env(fake_stream, fake_mic=None, **overrides):
    values = fake_stream.env()
    if fake_mic is not None:
        values["PYTHONPATH"] = fake_mic.pythonpath(fake_stream.root)
    values.update(overrides)
    return values


def test_missing_sounddevice_is_a_capability_error(qasr, fake_stream, fake_model, tmp_path):
    blocker = tmp_path / "no_sounddevice"
    blocker.mkdir()
    (blocker / "sounddevice.py").write_text(
        "raise ImportError('sounddevice is not installed')\n", encoding="utf-8"
    )
    env = _env(fake_stream)
    env["PYTHONPATH"] = str(blocker) + ":" + env["PYTHONPATH"]
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", "mic",
        "--emit", "jsonl",
        env=env,
    )
    assert_exit(proc, 4, context="mic without sounddevice")
    assert "mic" in (proc.stdout + proc.stderr).lower()
    assert_no_traceback(proc)


def test_missing_sounddevice_fails_before_model_load(
    qasr, fake_stream, fake_model, tmp_path
):
    blocker = tmp_path / "no_sounddevice_early"
    blocker.mkdir()
    (blocker / "sounddevice.py").write_text(
        "raise ImportError('sounddevice is not installed')\n", encoding="utf-8"
    )
    env = _env(fake_stream)
    env["PYTHONPATH"] = str(blocker) + ":" + env["PYTHONPATH"]
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", "mic",
        env=env,
    )
    assert_exit(proc, 4, context="mic preflight")
    assert fake_stream.events("model_init") == [], fake_stream.records()
    assert_no_traceback(proc)


def test_mic_success_emits_final_snapshot(
    qasr, fake_stream, fake_mic, fake_model, tmp_path
):
    """A real mic has no EOF; stop it with SIGINT and expect Q15 semantics."""
    mic_pcm = pcm_s16le(16_000)
    pcm_path = tmp_path / "mic.pcm"
    pcm_path.write_bytes(mic_pcm)
    log = tmp_path / "mic.log"
    env = _env(
        fake_stream, fake_mic,
        QASR_FAKE_MIC_PCM=str(pcm_path),
        QASR_FAKE_MIC_LOG=str(log),
    )
    proc = qasr.popen(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", "mic",
        "--chunk-ms", "100", "--emit", "jsonl",
        env=env,
    )
    assert proc.stdout is not None
    try:
        lines: list[str] = []
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            readable, _, _ = select.select([proc.stdout], [], [], 0.5)
            if not readable:
                if proc.poll() is not None:
                    break
                continue
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("audio_s", 0) >= 0.8:
                break
        assert lines, "no snapshot from mic"
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=30)
    except Exception:
        proc.kill()
        raise
    assert proc.returncode == 130, (proc.returncode, stdout, stderr)
    values = parse_jsonl(
        type("P", (), {"stdout": "".join(lines) + stdout, "stderr": stderr,
                       "returncode": proc.returncode})()
    )
    assert values[-1]["is_final"] is True
    assert values[-1]["audio_s"] >= 0.8
    opened = read_jsonl_file(log)
    assert opened and opened[0]["event"] == "open"
    assert opened[0]["samplerate"] == 16_000
    assert opened[0]["channels"] == 1
    assert opened[0]["blocksize"] == 1_600  # P2-Q8: fixed 100 ms blocks
    assert opened[0]["dtype"] == "float32"


def test_mic_device_open_failure_is_runtime_error(
    qasr, fake_stream, fake_mic, fake_model
):
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", "mic",
        "--emit", "jsonl",
        env=_env(fake_stream, fake_mic, QASR_FAKE_MIC_FAIL="open"),
    )
    assert_exit(proc, 3, context="mic open failure")
    text = (proc.stdout + proc.stderr).lower()
    assert "microphone" in text or "device" in text
    assert "fake microphone open failure" in text
    assert_no_traceback(proc)


def test_mic_rejects_realtime(qasr, fake_stream, fake_model):
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", "mic", "--realtime",
        env=fake_stream.env(),
    )
    assert_exit(proc, 2, context="mic + realtime")
    assert_no_traceback(proc)


@pytest.mark.slow
def test_realtime_paces_file_input(qasr, fake_stream, fake_model, tmp_path):
    audio = write_wav(tmp_path / "three.wav", seconds=3.0)
    started = time.monotonic()
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", audio,
        "--chunk-ms", "1000", "--realtime", "--emit", "jsonl",
        env=fake_stream.env(),
        timeout=30,
    )
    elapsed = time.monotonic() - started
    assert_exit(proc, 0, context="realtime file")
    assert parse_jsonl(proc)[-1]["is_final"] is True
    assert elapsed >= 1.7, elapsed
    assert elapsed <= 8.0, elapsed


@pytest.mark.slow
def test_realtime_does_not_sleep_after_eof_tail(
    qasr, fake_stream, fake_model, tmp_path
):
    """P2-Q9: the EOF partial tail is not followed by a pacing sleep."""
    audio = write_wav(tmp_path / "tail.wav", seconds=1.5)
    started = time.monotonic()
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", audio,
        "--chunk-ms", "1000", "--realtime", "--emit", "jsonl",
        env=fake_stream.env(),
        timeout=30,
    )
    elapsed = time.monotonic() - started
    assert_exit(proc, 0, context="realtime tail")
    # One 1 s chunk is paced; the 0.5 s tail should not add another full 1 s.
    assert elapsed < 1.75, elapsed


def test_non_realtime_file_is_immediate(qasr, fake_stream, fake_model, tmp_path):
    audio = write_wav(tmp_path / "three.wav", seconds=3.0)
    started = time.monotonic()
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm", "--input", audio,
        "--chunk-ms", "1000", "--emit", "jsonl",
        env=fake_stream.env(),
        timeout=30,
    )
    elapsed = time.monotonic() - started
    assert_exit(proc, 0)
    assert elapsed < 2.0, elapsed
