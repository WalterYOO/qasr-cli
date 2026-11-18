
import pytest

from qasr_cli.catalog import (
    FingerprintCache,
    find_partial_files,
    models_list,
    models_path,
    resolve_aligner,
    resolve_model,
    spec_for,
    verify_model,
)
from qasr_cli.errors import ModelError


def test_aliases():
    assert spec_for("1.7b").repo == "Qwen/Qwen3-ASR-1.7B"
    assert spec_for("0.6b").repo == "Qwen/Qwen3-ASR-0.6B"
    assert spec_for("aligner").kind == "aligner"
    assert spec_for("Qwen/Qwen3-ASR-0.6B").alias == "0.6b"
    assert spec_for("Qwen3-ForcedAligner-0.6B").alias == "aligner"
    assert spec_for("nope") is None


def test_verify_complete_single_file(model_factory):
    model_factory("Qwen3-ASR-0.6B")
    resolved = resolve_model("0.6b")
    payload = verify_model(resolved)
    assert payload["status"] == "complete"
    assert payload["ok"] is True
    assert payload["fingerprint"].startswith("sha256:")


def test_verify_complete_sharded(model_factory):
    model_factory("Qwen3-ASR-1.7B", sharded=True)
    payload = verify_model(resolve_model("1.7b"))
    assert payload["status"] == "complete"


def test_verify_truncated_single_file_is_incomplete(model_factory):
    model_factory("Qwen3-ASR-0.6B", truncate=True)
    payload = verify_model(resolve_model("0.6b"))
    assert payload["status"] == "incomplete"
    assert payload["ok"] is False
    assert "pull 0.6b --resume" in payload["hint"]
    assert any(
        "mismatch" in check["detail"] or "header" in check["detail"]
        for check in payload["checks"]
    )


def test_verify_truncated_shard_is_incomplete(model_factory):
    model_factory("Qwen3-ASR-1.7B", sharded=True, truncate=True)
    payload = verify_model(resolve_model("1.7b"))
    assert payload["status"] == "incomplete"


def test_verify_partial_marker_is_incomplete(model_factory):
    model_factory("Qwen3-ASR-1.7B", sharded=True, partial_marker=True)
    payload = verify_model(resolve_model("1.7b"))
    assert payload["status"] == "incomplete"
    assert any(check["name"] == "partial-download" for check in payload["checks"])
    assert "pull 1.7b --resume" in payload["hint"]


def test_verify_missing_model(isolated_home):
    payload = verify_model(resolve_model("1.7b"))
    assert payload["status"] == "missing"
    assert payload["path"] is None


def test_find_partial_files(tmp_path):
    (tmp_path / "a.aria2").write_text("x")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.part").write_text("x")
    (tmp_path / "ok.safetensors").write_text("x")
    found = {p.name for p in find_partial_files(tmp_path)}
    assert found == {"a.aria2", "b.part"}


def test_resolve_model_prefers_explicit_path(model_factory, tmp_path):
    model = model_factory("Qwen3-ASR-0.6B")
    resolved = resolve_model(str(model))
    assert resolved.path == model.resolve()
    assert resolved.present


def test_resolve_aligner_auto(model_factory):
    model_factory("Qwen3-ForcedAligner-0.6B")
    resolved = resolve_aligner("auto")
    assert resolved is not None
    assert resolved.present
    assert resolved.kind == "aligner"


def test_resolve_aligner_required_missing(isolated_home):
    with pytest.raises(ModelError, match="aligner"):
        resolve_aligner("auto", required=True)


def test_resolve_aligner_none():
    assert resolve_aligner("none") is None


def test_models_path_missing(isolated_home):
    with pytest.raises(ModelError, match="not available locally"):
        models_path("0.6b")


def test_models_list(isolated_home, model_factory):
    model_factory("Qwen3-ASR-0.6B")
    payload = models_list()
    by_alias = {entry["alias"]: entry for entry in payload["models"]}
    assert by_alias["0.6b"]["status"] == "complete"
    assert by_alias["1.7b"]["status"] == "missing"
    assert by_alias["aligner"]["present"] is False


def test_fingerprint_cache_reuses_and_invalidates(tmp_path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"hello world")
    cache = FingerprintCache(tmp_path / "fp.json")
    first = cache.file_fingerprint(target)
    second = cache.file_fingerprint(target)
    assert first == second
    target.write_bytes(b"hello world!!")
    third = cache.file_fingerprint(target)
    assert third != first


def test_large_file_partial_fingerprint(tmp_path, monkeypatch):
    import qasr_cli.catalog as catalog

    monkeypatch.setattr(catalog, "LARGE_FILE_BYTES", 4)
    target = tmp_path / "big.bin"
    target.write_bytes(b"0123456789abcdef")
    fp = catalog.fingerprint_file(target)
    assert fp.startswith("sha256-partial:")
    full = catalog.fingerprint_file(target, full=True)
    assert full.startswith("sha256:")


def _fake_hub(monkeypatch, *, accept_resume=True):
    import sys
    import types

    calls = []

    def snapshot_download(*, repo_id, local_dir, **kwargs):
        calls.append({"repo_id": repo_id, "local_dir": local_dir, "kwargs": kwargs})
        if kwargs.get("resume_download") and not accept_resume:
            raise TypeError("unexpected keyword argument 'resume_download'")
        dest = __import__("pathlib").Path(local_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}", encoding="utf-8")
        (dest / "model.safetensors").write_bytes(b"\0" * 32)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def test_pull_model_resume_writes_source_json(monkeypatch):
    import json
    import pathlib

    from qasr_cli.catalog import pull_model

    calls = _fake_hub(monkeypatch)
    payload = pull_model("0.6b", revision="v1", resume=True)
    assert calls[0]["kwargs"]["resume_download"] is True
    dest = pathlib.Path(payload["path"])
    source = json.loads((dest / "source.json").read_text(encoding="utf-8"))
    assert source["repo"] == "Qwen/Qwen3-ASR-0.6B"
    assert source["revision"] == "v1"
    assert source["source"] == "hf"
    files = {entry["path"]: entry["size"] for entry in source["files"]}
    assert files["model.safetensors"] == 32
    assert source["total_size"] == sum(files.values())


def test_pull_model_resume_falls_back_when_unsupported(monkeypatch):
    from qasr_cli.catalog import pull_model

    calls = _fake_hub(monkeypatch, accept_resume=False)
    pull_model("0.6b", resume=True)
    assert len(calls) == 2
    assert "resume_download" not in calls[1]["kwargs"]
