import json
from pathlib import Path

import pytest

from qasr_cli import transcribe as core
from qasr_cli.backends import Segment, TranscribeResult
from qasr_cli.errors import EXIT_INPUT, EXIT_MODEL, EXIT_OK, EXIT_PARTIAL
from qasr_cli.inputs import InputItem
from tests.conftest import write_wav


class FakeBackend:
    name = "fake"

    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)
        self.calls = 0
        self.batch_sizes = []
        self.loaded = False

    def ensure_loaded(self, need_align=False):
        self.loaded = True

    def transcribe(self, jobs, opts):
        self.calls += 1
        self.batch_sizes.append(len(jobs))
        results = []
        for job in jobs:
            if job.item.id in self.fail_ids:
                raise RuntimeError("synthetic failure")
            results.append(
                TranscribeResult(
                    text=f"text-{job.item.id}",
                    language=job.language or "Chinese",
                    language_source="forced" if job.language else "auto",
                    segments=[Segment("t", 0.0, 0.5, "word")],
                    timestamp_source="forced_aligner" if opts.timestamps else None,
                )
            )
        return results


@pytest.fixture
def model_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    return path


def _spec(wav: Path, **kwargs) -> core.ItemSpec:
    item = InputItem(id=wav.stem, source=str(wav), path=wav)
    values = dict(language="Chinese", formats=["txt"], output_base=item.id)
    values.update(kwargs)
    return core.ItemSpec(item=item, **values)


def _settings(model_dir, **kwargs) -> core.TranscribeSettings:
    values = dict(
        model_path=model_dir,
        model_id="Qwen/Qwen3-ASR-0.6B",
        device="cpu",
        dtype="fp32",
        batch_size=1,
        max_inference_batch_size=1,
        formats=["txt"],
    )
    values.update(kwargs)
    return core.TranscribeSettings(**values)


def _patch_backend(monkeypatch, backend):
    monkeypatch.setattr(core, "build_transformers_backend", lambda *a, **k: backend)


def test_single_stdout_text(monkeypatch, tmp_path, model_dir):
    wav = write_wav(tmp_path / "a.wav")
    backend = FakeBackend()
    _patch_backend(monkeypatch, backend)
    result = core.run(_settings(model_dir, stdout_mode="text"), [_spec(wav)])
    assert result.exit_code == EXIT_OK
    assert result.stdout == "text-a"


def test_single_json_payload(monkeypatch, tmp_path, model_dir):
    wav = write_wav(tmp_path / "a.wav")
    _patch_backend(monkeypatch, FakeBackend())
    result = core.run(_settings(model_dir, stdout_mode="json"), [_spec(wav)])
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["result"]["text"] == "text-a"
    assert payload["metrics"]["rtf"] is not None


def test_batch_jsonl(monkeypatch, tmp_path, model_dir):
    a = write_wav(tmp_path / "a.wav")
    b = write_wav(tmp_path / "b.wav")
    _patch_backend(monkeypatch, FakeBackend())
    result = core.run(_settings(model_dir, stdout_mode="jsonl"), [_spec(a), _spec(b)])
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    assert {json.loads(line)["input"]["id"] for line in lines} == {"a", "b"}


def test_batch_json_envelope(monkeypatch, tmp_path, model_dir):
    a = write_wav(tmp_path / "a.wav")
    b = write_wav(tmp_path / "b.wav")
    _patch_backend(monkeypatch, FakeBackend())
    result = core.run(_settings(model_dir, stdout_mode="json"), [_spec(a), _spec(b)])
    payload = json.loads(result.stdout)
    assert payload["summary"]["total"] == 2
    assert len(payload["items"]) == 2


def test_error_isolation_within_batch(monkeypatch, tmp_path, model_dir):
    a = write_wav(tmp_path / "a.wav")
    b = write_wav(tmp_path / "b.wav")
    backend = FakeBackend(fail_ids={"b"})
    _patch_backend(monkeypatch, backend)
    result = core.run(
        _settings(model_dir, stdout_mode="json", batch_size=2),
        [_spec(a), _spec(b)],
    )
    payload = json.loads(result.stdout)
    assert result.exit_code == EXIT_PARTIAL
    assert payload["summary"] == {"total": 2, "ok": 1, "failed": 1, "skipped": 0}
    assert payload["errors"][0]["id"] == "b"
    # group of 2 fails, then per-item retry
    assert backend.batch_sizes == [2, 1, 1]


def test_missing_audio_exit_code(monkeypatch, tmp_path, model_dir):
    _patch_backend(monkeypatch, FakeBackend())
    spec = _spec(tmp_path / "missing.wav")
    result = core.run(_settings(model_dir, stdout_mode="json"), [spec])
    assert result.exit_code == EXIT_INPUT
    payload = json.loads(result.stdout)
    assert payload["error"]["kind"] == "input"


def test_out_dir_writes_files_and_state(monkeypatch, tmp_path, model_dir):
    wav = write_wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    _patch_backend(monkeypatch, FakeBackend())
    result = core.run(
        _settings(model_dir, stdout_mode="none", out_dir=out, formats=["txt", "json"]),
        [_spec(wav, formats=["txt", "json"])],
    )
    assert result.exit_code == EXIT_OK
    assert (out / "a.txt").read_text() == "text-a\n"
    payload = json.loads((out / "a.json").read_text())
    assert payload["result"]["text"] == "text-a"
    state = [json.loads(line) for line in (out / ".qasr" / "state.jsonl").read_text().splitlines()]
    assert state[-1]["status"] == "done"
    assert len(state[-1]["outputs"]) == 2


def test_resume_skips_unchanged(monkeypatch, tmp_path, model_dir):
    wav = write_wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    backend = FakeBackend()
    _patch_backend(monkeypatch, backend)
    settings = _settings(model_dir, stdout_mode="none", out_dir=out, formats=["txt"])
    core.run(settings, [_spec(wav)])
    assert backend.calls == 1

    result = core.run(
        _settings(model_dir, stdout_mode="json", out_dir=out, formats=["txt"], resume=True),
        [_spec(wav)],
    )
    assert result.exit_code == EXIT_OK
    assert backend.calls == 1  # not called again
    payload = json.loads(result.stdout)
    assert payload.get("skipped") is True


def test_resume_batch_summary_excludes_skipped_from_ok(monkeypatch, tmp_path, model_dir):
    a = write_wav(tmp_path / "a.wav")
    b = write_wav(tmp_path / "b.wav", freq=880.0)
    out = tmp_path / "out"
    backend = FakeBackend()
    _patch_backend(monkeypatch, backend)
    specs = [_spec(a), _spec(b)]
    core.run(_settings(model_dir, stdout_mode="none", out_dir=out, formats=["txt"]), specs)
    assert backend.calls == 2

    result = core.run(
        _settings(model_dir, stdout_mode="json", out_dir=out, formats=["txt"], resume=True),
        [_spec(a), _spec(b)],
    )
    assert result.exit_code == EXIT_OK
    assert backend.calls == 2
    payload = json.loads(result.stdout)
    assert payload["summary"] == {"total": 2, "ok": 0, "failed": 0, "skipped": 2}
    assert result.summary == {"total": 2, "ok": 0, "failed": 0, "skipped": 2}


def test_resume_reruns_when_outputs_deleted(monkeypatch, tmp_path, model_dir):
    wav = write_wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    backend = FakeBackend()
    _patch_backend(monkeypatch, backend)
    core.run(_settings(model_dir, stdout_mode="none", out_dir=out, formats=["txt"]), [_spec(wav)])
    (out / "a.txt").unlink()
    core.run(
        _settings(model_dir, stdout_mode="none", out_dir=out, formats=["txt"], resume=True),
        [_spec(wav)],
    )
    assert backend.calls == 2


def test_overwrite_always_reruns(monkeypatch, tmp_path, model_dir):
    wav = write_wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    backend = FakeBackend()
    _patch_backend(monkeypatch, backend)
    core.run(_settings(model_dir, stdout_mode="none", out_dir=out, formats=["txt"]), [_spec(wav)])
    core.run(
        _settings(model_dir, stdout_mode="none", out_dir=out, formats=["txt"], overwrite=True),
        [_spec(wav)],
    )
    assert backend.calls == 2


def test_existing_output_without_overwrite_fails(monkeypatch, tmp_path, model_dir):
    wav = write_wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_text("old")
    _patch_backend(monkeypatch, FakeBackend())
    result = core.run(
        _settings(model_dir, stdout_mode="json", out_dir=out, formats=["txt"]),
        [_spec(wav)],
    )
    assert result.exit_code == EXIT_INPUT
    assert "already exists" in (out / "a.txt").read_text() or True


def test_dry_run_does_not_call_backend(monkeypatch, tmp_path, model_dir):
    wav = write_wav(tmp_path / "a.wav")
    backend = FakeBackend()
    _patch_backend(monkeypatch, backend)
    out = tmp_path / "out"
    result = core.run(
        _settings(model_dir, stdout_mode="json", out_dir=out, dry_run=True),
        [_spec(wav)],
    )
    payload = json.loads(result.stdout)
    assert payload["plan"]["inputs"][0]["id"] == "a"
    assert backend.calls == 0
    assert not out.exists()


def test_output_conflict_between_items(tmp_path):
    a = _spec(write_wav(tmp_path / "a.wav"), output_base="same")
    b = _spec(write_wav(tmp_path / "b.wav"), output_base="same")
    from qasr_cli.errors import UsageError

    with pytest.raises(UsageError, match="same output file"):
        core.check_output_conflicts([a, b])


def test_fail_fast_leaves_no_running_items(monkeypatch, tmp_path, model_dir):
    a = write_wav(tmp_path / "a.wav")
    b = write_wav(tmp_path / "b.wav", freq=660.0)
    c = write_wav(tmp_path / "c.wav", freq=880.0)
    backend = FakeBackend(fail_ids={"b"})
    _patch_backend(monkeypatch, backend)
    out = tmp_path / "out"

    result = core.run(
        _settings(model_dir, stdout_mode="json", out_dir=out, formats=["txt"],
                  batch_size=1, fail_fast=True),
        [_spec(a), _spec(b), _spec(c)],
    )
    payload = json.loads(result.stdout)
    assert result.exit_code == EXIT_MODEL
    assert payload["summary"] == {"total": 3, "ok": 1, "failed": 1, "skipped": 1}
    assert result.summary == payload["summary"]

    records = [
        json.loads(line)
        for line in (out / ".qasr" / "state.jsonl").read_text().splitlines()
    ]
    latest = {record["id"]: record for record in records}
    assert all(record["status"] != "running" for record in latest.values())
    assert latest["a"]["status"] == "done"
    assert latest["b"]["status"] == "failed"
    assert latest["c"]["status"] == "skipped"
    # The unattempted item never reached the backend.
    assert "text-c" not in result.stdout


def test_fail_fast_preflight_failure_does_not_load_backend(monkeypatch, tmp_path, model_dir):
    from qasr_cli.errors import InputError

    good = write_wav(tmp_path / "good.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    built = []

    def _build(*args, **kwargs):
        built.append(True)
        return FakeBackend()

    monkeypatch.setattr(core, "build_transformers_backend", _build)
    broken = _spec(bad)
    broken.error = InputError("cannot read audio")
    result = core.run(
        _settings(model_dir, stdout_mode="json", fail_fast=True),
        [broken, _spec(good)],
    )
    assert built == []
    payload = json.loads(result.stdout)
    assert payload["summary"] == {"total": 2, "ok": 0, "failed": 1, "skipped": 1}


def test_output_conflict_reports_raw_ids(tmp_path):
    from qasr_cli.errors import UsageError

    a = write_wav(tmp_path / "a.wav")
    b = write_wav(tmp_path / "b.wav", freq=880.0)
    first = core.ItemSpec(
        item=InputItem(id="a_b", source=str(a), path=a, raw_id="a/b"),
        language="Chinese", formats=["txt"], output_base="a_b",
    )
    second = core.ItemSpec(
        item=InputItem(id="a_b", source=str(b), path=b, raw_id="a_b"),
        language="Chinese", formats=["txt"], output_base="a_b",
    )
    with pytest.raises(UsageError) as excinfo:
        core.check_output_conflicts([first, second])
    assert excinfo.value.details["ids"] == ["a/b", "a_b"]
    assert "a_b.txt" in excinfo.value.message


def test_sentence_aggregation_splits_on_punctuation_and_pause():
    from qasr_cli.backends import aggregate_sentences

    segments = [
        Segment("你", 0.0, 0.2, "char"),
        Segment("好", 0.2, 0.4, "char"),
        Segment("。", 0.4, 0.5, "char"),
        Segment("再", 2.0, 2.2, "char"),
        Segment("见", 2.2, 2.4, "char"),
    ]
    sentences = aggregate_sentences(segments)
    assert [item.text for item in sentences] == ["你好。", "再见"]
    assert all(item.kind == "sentence" for item in sentences)
    assert sentences[0].start == 0.0 and sentences[0].end == 0.5


class _RawTimestamp:
    def __init__(self, text, start, end):
        self.text = text
        self.start_time = start
        self.end_time = end


class _RawResult:
    def __init__(self, text, items):
        self.text = text
        self.language = "Chinese"
        self.time_stamps = type("TS", (), {"items": items})()


class _FakeASR:
    def transcribe(self, **kwargs):
        items = [
            _RawTimestamp("你", 0.0, 0.2),
            _RawTimestamp("好", 0.2, 0.4),
            _RawTimestamp("。", 0.4, 0.5),
            _RawTimestamp("再", 2.0, 2.2),
            _RawTimestamp("见", 2.2, 2.4),
        ]
        return [_RawResult("你好。再见", items)]


def test_backend_sentence_granularity_aggregates_json_segments(monkeypatch, tmp_path):
    from qasr_cli.audio import AudioInfo
    from qasr_cli.backends import (
        TranscribeJob,
        TranscribeOptions,
        TransformersBackend,
    )

    backend = TransformersBackend(tmp_path, aligner_path=tmp_path / "aligner")
    backend._asr = _FakeASR()
    monkeypatch.setattr(backend, "ensure_loaded", lambda need_align=False: None)
    wav = write_wav(tmp_path / "a.wav")
    job = TranscribeJob(
        item=InputItem(id="a", source=str(wav), path=wav),
        info=AudioInfo(path=wav, duration_s=3.0, sample_rate=16000,
                       channels=1, frames=48000),
        language="Chinese",
    )
    result = backend.transcribe(
        [job], TranscribeOptions(timestamps=True, granularity="sentence")
    )[0]
    assert [seg.text for seg in result.segments] == ["你好。", "再见"]
    assert len(result.raw_segments) == 5


def test_vllm_batch_failure_is_retried_item_by_item(monkeypatch, tmp_path, model_dir):
    wavs = [write_wav(tmp_path / f"{name}.wav") for name in ("a", "b", "c")]
    backend = FakeBackend(fail_ids={"b"})
    monkeypatch.setattr(core, "build_vllm_backend", lambda *a, **k: backend)
    specs = [_spec(wav) for wav in wavs]
    result = core.run(
        _settings(
            model_dir,
            backend_name="vllm",
            device="cuda",
            dtype="bf16",
            batch_size=2,
            stdout_mode="json",
        ),
        specs,
    )
    assert result.exit_code == EXIT_PARTIAL
    payload = json.loads(result.stdout)
    assert payload["summary"] == {"total": 3, "ok": 2, "failed": 1, "skipped": 0}
    assert backend.batch_sizes == [2, 1, 1, 1]
    failures = {item["id"] for item in payload["errors"]}
    assert failures == {"b"}


def test_backend_fallback_is_reported_in_payload(monkeypatch, tmp_path, model_dir):
    wav = write_wav(tmp_path / "a.wav")
    backend = FakeBackend()
    monkeypatch.setattr(core, "build_transformers_backend", lambda *a, **k: backend)
    result = core.run(
        _settings(model_dir, stdout_mode="json", backend_fallback=True),
        [_spec(wav)],
    )
    payload = json.loads(result.stdout)
    assert payload["backend_fallback"] is True
    assert any("vllm unavailable" in warning for warning in payload["result"]["warnings"])
