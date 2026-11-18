"""N4: long-audio windowing in the transcribe pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from qasr_cli import transcribe as core
from qasr_cli.audio import AudioInfo
from qasr_cli.backends import AlignResult, Segment, TranscribeResult
from qasr_cli.errors import EXIT_OK
from qasr_cli.inputs import InputItem
from tests.conftest import write_wav


class WindowBackend:
    name = "fake"

    def __init__(self):
        self.calls = []
        self.align_calls = []

    def ensure_loaded(self, need_align=False):
        return None

    def transcribe(self, jobs, opts):
        results = []
        for job in jobs:
            self.calls.append(job.item.id)
            suffix = job.item.id.rsplit("#", 1)[-1]
            results.append(
                TranscribeResult(
                    text=f"片段{suffix}",
                    language="Chinese",
                    language_source="forced",
                    segments=[],
                )
            )
        return results

    def align(self, jobs, opts):
        results = []
        for job in jobs:
            self.align_calls.append(job.text)
            step = 0.2
            items = [
                Segment(ch, i * step, (i + 1) * step, "char")
                for i, ch in enumerate(job.text)
            ]
            results.append(AlignResult(items=items))
        return results


@pytest.fixture
def long_audio(monkeypatch, tmp_path):
    path = write_wav(tmp_path / "long.wav", seconds=1.0)
    info = AudioInfo(
        path=path, duration_s=400.0, sample_rate=16000, channels=1,
        frames=400 * 16000, format="WAV",
    )
    monkeypatch.setattr(
        core, "read_window_16k",
        lambda *a, **k: np.zeros(16000, dtype="float32"),
    )
    monkeypatch.setattr(core, "probe", lambda _p: info)
    return path, info


def _spec(path, info, *, timestamps=False, language="Chinese"):
    item = InputItem(id="long", source=str(path), path=path)
    return core.ItemSpec(
        item=item,
        language=language,
        formats=["txt"],
        output_base="long",
        info=info,
        timestamps=timestamps,
    )


def _settings(tmp_path, **kwargs):
    values = dict(
        model_path=tmp_path / "model",
        model_id="Qwen/Qwen3-ASR-0.6B",
        device="cpu",
        dtype="fp32",
        batch_size=1,
        max_inference_batch_size=1,
        formats=["txt"],
        stdout_mode="none",
    )
    values.update(kwargs)
    return core.TranscribeSettings(**values)


def test_plain_long_audio_concatenates_windows(monkeypatch, tmp_path, long_audio):
    path, info = long_audio
    backend = WindowBackend()
    monkeypatch.setattr(core, "build_transformers_backend", lambda *a, **k: backend)
    out = tmp_path / "out"
    result = core.run(
        _settings(tmp_path, out_dir=out, formats=["txt"]),
        [_spec(path, info)],
    )
    assert result.exit_code == EXIT_OK
    assert backend.calls == ["long#0", "long#1"]
    assert (out / "long.txt").read_text(encoding="utf-8").strip() == "片段0片段1"


def test_long_audio_writes_chunk_checkpoints_and_resumes(
    monkeypatch, tmp_path, long_audio
):
    path, info = long_audio
    backend = WindowBackend()
    monkeypatch.setattr(core, "build_transformers_backend", lambda *a, **k: backend)
    out = tmp_path / "out"
    settings = _settings(tmp_path, out_dir=out, formats=["txt"])
    core.run(settings, [_spec(path, info)])
    chunks = sorted((out / ".qasr" / "chunks").glob("*/*.json"))
    assert len(chunks) == 2
    records = [json.loads(chunk.read_text()) for chunk in chunks]
    assert all("chunk_key" in record for record in records)

    backend.calls.clear()
    resumed = core.run(
        _settings(tmp_path, out_dir=out, formats=["txt"], resume=True),
        [_spec(path, info)],
    )
    assert resumed.exit_code == EXIT_OK
    assert backend.calls == []


def test_long_audio_cache_invalidated_when_window_changes(
    monkeypatch, tmp_path, long_audio
):
    path, info = long_audio
    backend = WindowBackend()
    monkeypatch.setattr(core, "build_transformers_backend", lambda *a, **k: backend)
    out = tmp_path / "out"
    core.run(_settings(tmp_path, out_dir=out, formats=["txt"]), [_spec(path, info)])
    (out / "long.txt").unlink()
    backend.calls.clear()
    core.run(
        _settings(tmp_path, out_dir=out, formats=["txt"], resume=True,
                  max_chunk_seconds=100.0),
        [_spec(path, info)],
    )
    assert backend.calls  # new window layout must not reuse old chunks


def test_long_timestamps_use_canonical_text_and_windowed_alignment(
    monkeypatch, tmp_path, long_audio
):
    path, info = long_audio
    backend = WindowBackend()
    monkeypatch.setattr(core, "build_transformers_backend", lambda *a, **k: backend)
    out = tmp_path / "out"

    spec = _spec(path, info, timestamps=True)
    result = core.run(
        _settings(
            tmp_path, out_dir=out, formats=["txt", "json"], max_chunk_seconds=180.0,
        ),
        [core.ItemSpec(
            item=spec.item, language=spec.language, formats=["txt", "json"],
            output_base="long", info=info, timestamps=True,
        )],
    )
    assert result.exit_code == EXIT_OK
    # Canonical ASR sees the full file once, then each window is an anchor.
    assert backend.calls == ["long", "long#0", "long#1", "long#2"]
    written = json.loads((out / "long.json").read_text())
    assert written["result"]["text"] == "片段long"
    items = written["result"]["segments"]
    starts = [item["start"] for item in items]
    assert starts == sorted(starts)
    assert len(items) > 0


def test_dry_run_reports_window_count(monkeypatch, tmp_path, long_audio):
    path, info = long_audio
    backend = WindowBackend()
    monkeypatch.setattr(core, "build_transformers_backend", lambda *a, **k: backend)
    result = core.run(
        _settings(tmp_path, out_dir=tmp_path / "out", stdout_mode="json", dry_run=True),
        [_spec(path, info)],
    )
    payload = json.loads(result.stdout)
    assert payload["plan"]["inputs"][0]["windows"] == 2
    assert backend.calls == []


def test_dry_run_exact_max_chunk_duration_is_one_window(
    monkeypatch, tmp_path
):
    path = write_wav(tmp_path / "exact.wav", seconds=1.0)
    info = AudioInfo(
        path=path, duration_s=180.0, sample_rate=16000, channels=1,
        frames=180 * 16000, format="WAV",
    )
    monkeypatch.setattr(core, "probe", lambda _p: info)
    result = core.run(
        _settings(
            tmp_path, out_dir=tmp_path / "out", stdout_mode="json",
            dry_run=True, max_chunk_seconds=180.0,
        ),
        [_spec(path, info)],
    )
    payload = json.loads(result.stdout)
    assert payload["plan"]["inputs"][0]["windows"] == 1
