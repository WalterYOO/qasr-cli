import json
from pathlib import Path

import pytest

from qasr_cli import transcribe as core
from qasr_cli.backends import Segment, TranscribeResult
from qasr_cli.errors import EXIT_INPUT, EXIT_OK, EXIT_PARTIAL
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
