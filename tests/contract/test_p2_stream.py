"""P2: qasr stream sources, snapshots and capability boundaries."""

from __future__ import annotations

import io
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from qasr_cli import stream as core_stream
from qasr_cli.cli import main
from qasr_cli.commands import stream as stream_cmd
from qasr_cli.errors import CapabilityError, RuntimeFailure
from tests.conftest import write_wav

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def validate(name, payload):
    schema = json.loads((SCHEMA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)


class FakeState:
    def __init__(self, chunk_samples):
        self.chunk_samples = chunk_samples
        self.chunk_id = 0
        self.buffer = np.zeros(0, dtype=np.float32)
        self.text = ""
        self.language = ""


class FakeStreamBackend:
    def __init__(self, *, fail_at=None, fail_finish=False):
        self.fail_at = fail_at
        self.fail_finish = fail_finish
        self.calls = 0
        self.loaded = False
        self.last_init = None

    def ensure_loaded(self):
        self.loaded = True

    def init_streaming_state(self, **kwargs):
        self.last_init = kwargs
        chunk_samples = int(round(float(kwargs["chunk_size_sec"]) * 16000))
        self.state = FakeState(chunk_samples)
        return self.state

    def _decode(self, state):
        state.chunk_id += 1
        state.text = f"text-{state.chunk_id}"
        state.language = "Chinese"

    def streaming_transcribe(self, pcm, state):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeFailure("stream decode failed")
        pieces = [state.buffer, np.asarray(pcm, dtype=np.float32).reshape(-1)]
        state.buffer = np.concatenate(pieces)
        while state.buffer.size >= state.chunk_samples:
            state.buffer = state.buffer[state.chunk_samples:]
            self._decode(state)
        return state

    def finish_streaming_transcribe(self, state):
        if self.fail_finish:
            raise RuntimeFailure("stream finish failed")
        if state.buffer.size:
            state.buffer = np.zeros(0, dtype=np.float32)
            self._decode(state)
        return state


@pytest.fixture
def stream_env(monkeypatch, model_factory):
    model_factory("Qwen3-ASR-1.7B")
    monkeypatch.setattr(stream_cmd, "require_vllm", lambda: None)
    backend = FakeStreamBackend()

    def _install(fake=None):
        target = fake or backend
        monkeypatch.setattr(stream_cmd, "build_vllm_backend", lambda *a, **k: target)
        return target

    _install()
    return backend, _install


def _stream_args(wav, *extra):
    return [
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "stream", "--input", str(wav), *extra,
    ]


def _lines(capsys):
    out = capsys.readouterr()
    return [json.loads(line) for line in out.out.splitlines() if line]


def test_file_jsonl_snapshots_are_cumulative_and_final(capsys, tmp_path, stream_env):
    backend, _ = stream_env
    wav = write_wav(tmp_path / "a.wav", seconds=4.0)
    code = main(_stream_args(wav, "--chunk-ms", "1000", "--emit", "jsonl"))
    lines = _lines(capsys)
    assert code == 0, lines
    assert [line["seq"] for line in lines] == [0, 1, 2, 3, 4]
    assert [line["is_final"] for line in lines] == [False] * 4 + [True]
    assert [line["text"] for line in lines] == [
        "text-1", "text-2", "text-3", "text-4", "text-4",
    ]
    assert all(line["language"] == "Chinese" for line in lines)
    assert lines[-1]["audio_s"] == pytest.approx(4.0)
    for line in lines:
        validate("stream", line)
    assert backend.last_init["chunk_size_sec"] == 1.0
    assert backend.last_init["unfixed_chunk_num"] == 2
    assert backend.last_init["unfixed_token_num"] == 5


def test_short_tail_still_emits_one_final_snapshot(capsys, tmp_path, stream_env):
    wav = write_wav(tmp_path / "a.wav", seconds=1.5)
    code = main(_stream_args(wav, "--chunk-ms", "2000", "--emit", "jsonl"))
    lines = _lines(capsys)
    assert code == 0, lines
    assert len(lines) == 1
    assert lines[0]["seq"] == 0
    assert lines[0]["is_final"] is True
    assert lines[0]["text"] == "text-1"


@pytest.mark.parametrize("emit", ["text", "jsonl"])
def test_json_is_always_rejected(capsys, tmp_path, stream_env, emit):
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    code = main(["--json"] + _stream_args(wav, "--emit", emit))
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert "--json" in payload["error"]["message"]


def test_text_mode_non_tty_prints_only_final(capsys, tmp_path, stream_env):
    wav = write_wav(tmp_path / "a.wav", seconds=3.0)
    code = main(_stream_args(wav, "--chunk-ms", "1000", "--emit", "text"))
    out = capsys.readouterr().out
    assert code == 0
    assert out == "text-3\n"


def test_text_mode_tty_overwrites_and_finishes_line(
    capsys, tmp_path, stream_env, monkeypatch
):
    wav = write_wav(tmp_path / "a.wav", seconds=2.0)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    code = main(_stream_args(wav, "--chunk-ms", "1000", "--emit", "text"))
    out = capsys.readouterr().out
    assert code == 0
    assert "\rtext-1" in out
    assert out.endswith("text-2\n")


def test_error_after_snapshot_appends_error_line(capsys, tmp_path, stream_env):
    _, install = stream_env
    install(FakeStreamBackend(fail_at=2))
    wav = write_wav(tmp_path / "a.wav", seconds=4.0)
    code = main(_stream_args(wav, "--chunk-ms", "1000", "--emit", "jsonl"))
    lines = _lines(capsys)
    assert code == 3
    assert len(lines) == 2
    assert lines[0]["ok"] is True
    assert lines[1]["ok"] is False
    assert lines[1]["seq"] == 1
    assert lines[1]["is_final"] is True
    assert lines[1]["error"]["message"] == "stream decode failed"
    validate("stream", lines[1])


def test_error_before_snapshot_uses_normal_error(capsys, tmp_path, stream_env):
    _, install = stream_env
    install(FakeStreamBackend(fail_at=1))
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    code = main(_stream_args(wav, "--chunk-ms", "1000", "--emit", "jsonl"))
    captured = capsys.readouterr()
    assert code == 3
    assert captured.out == ""
    assert "stream decode failed" in captured.err


def test_ctrl_c_flushes_final_snapshot_then_130(capsys, tmp_path, stream_env):
    _, install = stream_env
    backend = FakeStreamBackend()
    calls = {"n": 0}
    original = backend.streaming_transcribe

    def _interrupt(pcm, state):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return original(pcm, state)

    backend.streaming_transcribe = _interrupt
    install(backend)
    wav = write_wav(tmp_path / "a.wav", seconds=4.0)
    code = main(_stream_args(wav, "--chunk-ms", "1000", "--emit", "jsonl"))
    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines() if line]
    assert code == 130
    assert len(lines) == 2
    assert lines[0]["ok"] is True and lines[0]["is_final"] is False
    assert lines[-1]["ok"] is True and lines[-1]["is_final"] is True
    assert lines[-1]["seq"] == 1


def test_max_stream_seconds_stops_with_final_and_exit_5(capsys, tmp_path, stream_env):
    wav = write_wav(tmp_path / "a.wav", seconds=4.0)
    code = main(
        _stream_args(
            wav, "--chunk-ms", "1000", "--emit", "jsonl",
            "--max-stream-seconds", "2",
        )
    )
    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines() if line]
    assert code == 5, lines
    assert len(lines) == 3
    assert lines[0]["ok"] is True and lines[0]["seq"] == 0
    assert lines[1]["ok"] is True and lines[1]["seq"] == 1
    assert lines[-1]["ok"] is False
    assert lines[-1]["seq"] == 2
    assert lines[-1]["is_final"] is True
    assert lines[-1]["error"]["kind"] == "quality"
    assert "max-stream-seconds" in lines[-1]["error"]["message"]
    assert captured.err == ""
    for line in lines:
        validate("stream", line)


def test_stdin_pcm_eof_and_odd_tail(capsys, tmp_path, stream_env, monkeypatch):
    backend, _ = stream_env
    samples = np.zeros(1600 * 3, dtype="<i2").tobytes() + b"\x01"
    monkeypatch.setattr(
        "sys.stdin", type("FakeStdin", (), {"buffer": io.BytesIO(samples)})()
    )
    code = main([
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "stream", "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "100", "--emit", "jsonl",
    ])
    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines() if line]
    assert code == 0, lines
    assert len(lines) == 4
    assert lines[-1]["is_final"] is True
    assert "trailing byte" in captured.err
    assert backend.calls == 3


def test_stdin_requires_pcm(capsys, stream_env):
    code = main([
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "stream", "--input", "stdin", "--emit", "jsonl",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "--pcm" in captured.err


def test_pcm_stereo_downmix_and_short_frames():
    payload = (
        np.array([1000, 3000, -1000, 1000], dtype="<i2").tobytes()
        + b"\x00"
    )
    chunks = list(core_stream.iter_pcm_s16le(io.BytesIO(payload), 4, 2))
    assert len(chunks) == 1
    assert chunks[0].tolist() == pytest.approx([2000 / 32768, 0.0])


def test_mic_without_sounddevice_is_capability_error(
    capsys, tmp_path, stream_env, monkeypatch
):
    monkeypatch.setattr(
        core_stream,
        "_import_sounddevice",
        lambda: (_ for _ in ()).throw(
            CapabilityError("microphone input requires sounddevice")
        ),
    )
    code = main([
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "stream", "--input", "mic", "--emit", "jsonl",
        "--max-stream-seconds", "0.2",
    ])
    captured = capsys.readouterr()
    assert code == 4
    assert "sounddevice" in captured.err


def test_mic_fake_sounddevice_respects_cap(capsys, stream_env, monkeypatch):
    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, frames):
            return np.zeros((frames, 1), dtype=np.float32), False

    class FakeSD:
        InputStream = lambda *a, **k: FakeStream()  # noqa: E731

    monkeypatch.setattr(core_stream, "_import_sounddevice", lambda: FakeSD())
    code = main([
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "stream", "--input", "mic", "--emit", "jsonl",
        "--max-stream-seconds", "0.2",
    ])
    lines = _lines(capsys)
    assert code == 5, lines
    assert lines[-1]["is_final"] is True
    assert lines[-1]["error"]["kind"] == "quality"


def test_realtime_paces_file_and_rejects_stdin(capsys, tmp_path, stream_env, monkeypatch):
    wav = write_wav(tmp_path / "a.wav", seconds=2.0)
    sleeps = []
    monkeypatch.setattr(core_stream.time, "sleep", lambda value: sleeps.append(value))
    code = main(_stream_args(wav, "--chunk-ms", "1000", "--realtime", "--emit", "jsonl"))
    capsys.readouterr()
    assert code == 0
    assert sleeps == [1.0]

    code = main([
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "stream", "--input", "stdin", "--pcm", "s16le", "--realtime",
        "--emit", "jsonl",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "--realtime" in captured.err


@pytest.mark.parametrize(
    "args",
    [
        ["--sample-rate", "8000"],
        ["--channels", "3"],
        ["--chunk-ms", "50"],
        ["--chunk-ms", "2055"],
        ["--unfixed-chunks", "-1"],
        ["--unfixed-tokens", "-1"],
        ["--max-stream-seconds", "0"],
    ],
)
def test_numeric_validation(capsys, tmp_path, stream_env, args):
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    code = main(_stream_args(wav, *args, "--emit", "jsonl"))
    capsys.readouterr()
    assert code == 2


@pytest.mark.parametrize("args", [
    ["--pcm", "s16le"],
    ["--sample-rate", "16000"],
    ["--channels", "2"],
])
def test_stdin_only_flags_rejected_for_file(capsys, tmp_path, stream_env, args):
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    code = main(_stream_args(wav, *args, "--emit", "jsonl"))
    captured = capsys.readouterr()
    assert code == 2
    assert "only valid with --input stdin" in captured.err


def test_explicit_vllm_unavailable_is_exit_4(capsys, tmp_path, stream_env, monkeypatch):
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    monkeypatch.setattr(
        stream_cmd,
        "require_vllm",
        lambda: (_ for _ in ()).throw(RuntimeFailure("vllm is not importable")),
    )
    code = main(_stream_args(wav, "--emit", "jsonl"))
    captured = capsys.readouterr()
    assert code == 4
    assert "vllm" in captured.err.lower()


def test_backend_capability_errors(capsys, tmp_path, stream_env):
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    code = main([
        "--backend", "transformers", "--device", "cpu", "--dtype", "fp32",
        "stream", "--input", str(wav), "--emit", "jsonl",
    ])
    captured = capsys.readouterr()
    assert code == 4
    assert "vLLM" in captured.err or "vllm" in captured.err

    code = main([
        "--server", "http://127.0.0.1:18000",
        "stream", "--input", str(wav), "--emit", "jsonl",
    ])
    captured = capsys.readouterr()
    assert code == 4


def test_input_errors(capsys, tmp_path, stream_env):
    code = main([
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "stream", "--input", "-", "--emit", "jsonl",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "--input stdin" in captured.err

    code = main([
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "stream", "--input", "https://example.com/a.wav", "--emit", "jsonl",
    ])
    captured = capsys.readouterr()
    assert code == 4

    code = main([
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "stream", "--input", str(tmp_path / "missing.wav"), "--emit", "jsonl",
    ])
    captured = capsys.readouterr()
    assert code == 4
    assert "not found" in captured.err


def test_missing_model_is_exit_3(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(stream_cmd, "require_vllm", lambda: None)
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    code = main([
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "--model", "missing/model", "stream", "--input", str(wav),
        "--emit", "jsonl",
    ])
    captured = capsys.readouterr()
    assert code == 3
    assert "not available locally" in captured.err


def test_gpu_memory_utilization_is_forwarded(
    capsys, tmp_path, stream_env, monkeypatch
):
    seen = {}
    backend, _ = stream_env

    def _build(path, **kwargs):
        seen.update(kwargs)
        return backend

    monkeypatch.setattr(stream_cmd, "build_vllm_backend", _build)
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    code = main([
        "--backend", "vllm", "--device", "cpu", "--dtype", "fp32",
        "--gpu-memory-utilization", "0.3",
        "stream", "--input", str(wav), "--emit", "jsonl",
    ])
    capsys.readouterr()
    assert code == 0
    assert seen["gpu_memory_utilization"] == 0.3


def test_context_and_language_are_forwarded(capsys, tmp_path, stream_env):
    backend, _ = stream_env
    wav = write_wav(tmp_path / "a.wav", seconds=1.0)
    context_file = tmp_path / "ctx.txt"
    context_file.write_text("keyword\n", encoding="utf-8")
    code = main(_stream_args(
        wav, "--language", "zh", "--context-file", str(context_file),
        "--emit", "jsonl",
    ))
    capsys.readouterr()
    assert code == 0
    assert backend.last_init["language"] == "Chinese"
    assert backend.last_init["context"] == "keyword"
