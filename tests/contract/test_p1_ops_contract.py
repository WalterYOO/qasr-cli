"""P1 N5: jobs, bench, models show/rm, config get/set contracts."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from qasr_cli import transcribe as core
from qasr_cli.backends import TranscribeResult
from qasr_cli.cli import main
from qasr_cli.config import qasr_home
from tests.conftest import write_wav

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def validate(name, payload):
    schema = json.loads((SCHEMA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)


class FakeBackend:
    name = "fake"

    def ensure_loaded(self, need_align=False):
        return None

    def transcribe(self, jobs, opts):
        return [
            TranscribeResult(
                text=f"text-{job.item.id}",
                language="Chinese",
                language_source="auto",
                segments=[],
            )
            for job in jobs
        ]


@pytest.fixture
def fake_backend(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(core, "build_transformers_backend", lambda *a, **k: backend)
    return backend


def _run_batch(capsys, tmp_path, model_factory, fake_backend):
    model_factory("Qwen3-ASR-1.7B")
    wav = write_wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    code = main([
        "transcribe", str(wav), "--device", "cpu", "--dtype", "fp32",
        "-o", str(out), "--json",
    ])
    captured = capsys.readouterr()
    assert code == 0, captured
    return out


def test_jobs_list_and_show(capsys, tmp_path, model_factory, fake_backend):
    out = _run_batch(capsys, tmp_path, model_factory, fake_backend)
    code = main(["jobs", "list", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    validate("jobs-list", payload)
    assert len(payload["jobs"]) == 1
    job = payload["jobs"][0]
    assert job["out_dir"] == str(out.resolve())
    assert job["state"] == "present"

    code = main(["jobs", "show", job["job_id"], "--json"])
    shown = json.loads(capsys.readouterr().out)
    assert code == 0
    validate("jobs-show", shown)
    assert shown["counts"]["done"] == 1

    code = main(["jobs", "show", str(out), "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["job"]["job_id"] == job["job_id"]


def test_jobs_unknown_and_bad_limit(capsys):
    code = main(["jobs", "show", "does-not-exist", "--json"])
    assert code == 4
    capsys.readouterr()
    code = main(["jobs", "list", "--limit", "0", "--json"])
    assert code == 2
    capsys.readouterr()


def test_bench_dry_run_schema(capsys):
    code = main([
        "--device", "cpu", "--dtype", "fp32",
        "bench", "--dry-run", "--json", "--duration", "2",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    validate("bench", payload)
    assert payload["plan"]["synthetic"] is True
    assert payload["plan"]["duration_s"] == 2.0


def test_bench_synthetic_json(capsys, model_factory, monkeypatch):
    model_factory("Qwen3-ASR-1.7B")
    import qasr_cli.commands.bench as bench_cmd

    monkeypatch.setattr(
        bench_cmd, "build_transformers_backend", lambda *a, **k: FakeBackend(),
        raising=False,
    )
    # build_transformers_backend is imported lazily inside handle, so patch
    # the source module instead.
    import qasr_cli.backends as backends

    monkeypatch.setattr(backends, "build_transformers_backend", lambda *a, **k: FakeBackend())
    code = main([
        "--device", "cpu", "--dtype", "fp32",
        "bench", "--json", "--duration", "0.2", "--iterations", "2",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload
    validate("bench", payload)
    assert payload["iterations"] == 2
    assert payload["synthetic"] is True


def test_models_show_and_dry_run_rm(capsys, model_factory):
    model_factory("Qwen3-ASR-1.7B")
    code = main(["models", "show", "1.7b", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    validate("models-show", payload)
    assert payload["alias"] == "1.7b"
    assert payload["present"] is True
    assert payload["file_count"] >= 1

    code = main(["models", "rm", "1.7b", "--dry-run", "--json"])
    removal = json.loads(capsys.readouterr().out)
    assert code == 0
    validate("models-show", removal)
    assert removal["removed"] is False
    assert Path(removal["path"]).is_dir()


def test_models_rm_requires_confirmation_and_deletes(capsys, model_factory):
    model_factory("Qwen3-ASR-1.7B")
    code = main(["models", "rm", "1.7b", "--json"])
    assert code == 2
    capsys.readouterr()

    code = main(["models", "rm", "1.7b", "--yes", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["removed"] is True
    assert not Path(payload["path"]).exists()


def test_config_get_set_roundtrip(capsys):
    code = main(["config", "set", "backend", "transformers", "--home", "--json"])
    assert code == 0, capsys.readouterr()
    assert json.loads(capsys.readouterr().out)["value"] == "transformers"

    code = main(["config", "get", "backend", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["value"] == "transformers"
    assert payload["source"].startswith("file:")


def test_config_set_project_scope(capsys, tmp_path):
    code = main([
        "config", "set", "window_seconds", "120", "--project", "--json",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    target = Path(payload["path"])
    assert target.name == "config.json"
    assert target.parent.name == ".qasr"
    assert json.loads(target.read_text())["window_seconds"] == 120.0


def test_config_rejects_unknown_and_symlink(capsys, tmp_path):
    code = main(["config", "set", "nope", "x", "--json"])
    assert code == 2
    capsys.readouterr()

    target = tmp_path / "real-config.json"
    target.write_text("{}")
    link = qasr_home() / "config.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists():
        link.unlink()
    link.symlink_to(target)
    code = main(["config", "set", "backend", "vllm", "--json"])
    assert code == 2
    capsys.readouterr()


def test_help_does_not_import_torch_or_vllm(capsys, monkeypatch):
    import sys

    blocker = pytest.importorskip("importlib.abc")

    class _Block(blocker.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in ("torch", "vllm", "qwen_asr"):
                raise AssertionError(f"--help imported {fullname}")
            return None

    for name in list(sys.modules):
        if name.split(".")[0] in ("torch", "vllm", "qwen_asr"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Block(), *sys.meta_path])
    main(["--help"])
    capsys.readouterr()


def test_config_show_includes_window_keys(capsys):
    main(["config", "show", "--json"])
    payload = json.loads(capsys.readouterr().out)
    for key in (
        "server", "timeout", "retries",
        "window_seconds", "window_overlap", "max_chunk_seconds",
    ):
        assert key in payload["config"], key


@pytest.mark.parametrize("target", ["../evil", "/tmp/x", "a/b"])
def test_jobs_show_rejects_non_id_paths(capsys, target):
    code = main(["jobs", "show", target, "--json"])
    assert code == 4
    capsys.readouterr()


def test_jobs_show_does_not_read_outside_index(capsys, tmp_path):
    evil = qasr_home() / "jobs" / ".." / "evil.json"
    evil.parent.mkdir(parents=True, exist_ok=True)
    (qasr_home() / "evil.json").write_text(
        json.dumps({"job_id": "evil", "out_dir": "/tmp"})
    )
    code = main(["jobs", "show", "../evil", "--json"])
    assert code == 4
    capsys.readouterr()


def test_models_rm_rejects_symlinked_alias(capsys, model_factory):
    import os

    model_factory("Qwen3-ASR-1.7B")
    real = qasr_home() / "models" / "Qwen3-ASR-1.7B"
    link = qasr_home() / "models" / "my-alias"
    if link.exists():
        link.unlink()
    os.symlink(real, link)
    code = main(["models", "rm", "my-alias", "--yes", "--json"])
    assert code == 2
    capsys.readouterr()
    assert real.is_dir()
    assert link.is_symlink()


def test_models_rm_rejects_symlinked_path(capsys, model_factory, tmp_path):
    import os

    model_factory("Qwen3-ASR-1.7B")
    real = qasr_home() / "models" / "Qwen3-ASR-1.7B"
    link = tmp_path / "linked-model"
    os.symlink(real, link)
    code = main(["models", "rm", str(link), "--yes", "--json"])
    assert code == 2
    capsys.readouterr()
    assert real.is_dir()


def test_models_rm_rejects_root_and_nested(capsys, model_factory):
    model_factory("Qwen3-ASR-1.7B")
    root = qasr_home() / "models"
    code = main(["models", "rm", str(root), "--yes", "--json"])
    assert code == 2
    capsys.readouterr()
    nested = root / "Qwen3-ASR-1.7B" / "sub"
    nested.mkdir()
    code = main(["models", "rm", str(nested), "--yes", "--json"])
    assert code == 2
    capsys.readouterr()
