"""MOD-01..11: aliases, path resolution, and integrity verification.

Half-pack cases are synthetic on purpose: the real 1.7B directory can be
complete on a given machine, so tests must not depend on its state.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import RUNTIME_MODULES
from helpers import assert_error, assert_exit, find_status, parse_json


def _model_dir(models_root: Path, basename: str) -> Path:
    return models_root / basename


def _verify(qasr, alias: str = "1.7b"):
    proc = qasr.run("--json", "models", "verify", alias)
    payload = parse_json(proc)
    status = find_status(payload)
    assert status is not None, f"verify response has no complete/incomplete/missing status: {payload!r}"
    return proc, payload, status


def test_models_list_aliases(qasr):
    proc = qasr.run("--json", "models", "list")
    assert_exit(proc, 0, context="models list --json")
    payload = parse_json(proc)
    models = payload.get("models")
    assert isinstance(models, list) and len(models) == 3, f"expected 3 aliases: {payload!r}"
    aliases = {item.get("alias") for item in models if isinstance(item, dict)}
    assert aliases == {"1.7b", "0.6b", "aligner"}, f"unexpected aliases: {aliases}"
    for item in models:
        assert item.get("repo"), f"missing repo: {item!r}"
        assert item.get("present") is False, f"empty QASR_HOME must report present=false: {item!r}"
        assert "path" in item


def test_models_path_resolves_alias(qasr, models_root, make_model):
    target = make_model(_model_dir(models_root, "Qwen3-ASR-1.7B"))
    proc = qasr.run("models", "path", "1.7b")
    assert_exit(proc, 0, context="models path 1.7b")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"models path printed {lines!r}"
    assert Path(lines[0]).resolve() == target.resolve()


def test_models_path_missing(qasr):
    proc = qasr.run("--json", "models", "path", "1.7b")
    assert_exit(proc, {3, 4}, context="models path missing")
    assert_error(parse_json(proc))


def test_models_verify_complete(qasr, models_root, make_model):
    make_model(_model_dir(models_root, "Qwen3-ASR-1.7B"))
    proc, payload, status = _verify(qasr)
    assert status == "complete", f"expected complete: {payload!r}"
    assert_exit(proc, 0, context="verify complete")


def test_models_verify_aria2_marker_is_incomplete(qasr, models_root, make_model):
    make_model(_model_dir(models_root, "Qwen3-ASR-1.7B"), aria2=True)
    proc, payload, status = _verify(qasr)
    assert status == "incomplete", f"expected incomplete: {payload!r}"
    assert_exit(proc, {3, 4}, context="verify .aria2")
    text = json.dumps(payload, ensure_ascii=False).lower()
    assert "pull" in text and "--resume" in text, f"missing resume hint: {payload!r}"


def test_models_verify_missing_shard_is_incomplete(qasr, models_root, make_model):
    make_model(_model_dir(models_root, "Qwen3-ASR-1.7B"), missing_shard=True)
    proc, payload, status = _verify(qasr)
    assert status == "incomplete", f"expected incomplete: {payload!r}"
    assert_exit(proc, {3, 4}, context="verify missing shard")
    assert "pull" in json.dumps(payload).lower()


def test_models_verify_truncated_safetensors_is_incomplete(qasr, models_root, make_model):
    make_model(_model_dir(models_root, "Qwen3-ASR-1.7B"), truncate_shard=True)
    proc, payload, status = _verify(qasr)
    assert status == "incomplete", f"expected incomplete: {payload!r}"
    assert_exit(proc, {3, 4}, context="verify truncated header")
    assert "traceback" not in proc.stderr.lower(), f"verify leaked a traceback:\n{proc.stderr}"


def test_models_verify_missing_directory(qasr):
    proc, payload, status = _verify(qasr)
    assert status == "missing", f"expected missing: {payload!r}"
    assert_exit(proc, {3, 4}, context="verify missing directory")


def test_models_verify_synthetic_half_pack_with_real_files(qasr, models_root, make_model):
    """A realistic mixed failure: valid config files, truncated shard + marker."""
    target = make_model(_model_dir(models_root, "Qwen3-ASR-1.7B"), truncate_shard=True)
    (target / "model-00001-of-00001.safetensors.aria2").write_bytes(b"")
    (target / "tokenizer.json").write_text("{}", encoding="utf-8")

    proc, payload, status = _verify(qasr)
    assert status == "incomplete"
    assert_exit(proc, {3, 4}, context="verify synthetic mixed half-pack")
    text = json.dumps(payload, ensure_ascii=False).lower()
    assert "pull" in text and "--resume" in text


def test_models_list_does_not_import_runtime(qasr, import_blocker):
    proc = qasr.run("--json", "models", "list", env=import_blocker.env())
    assert_exit(proc, 0, context="models list without runtime")
    blocked = [name for name in import_blocker.attempts() if name.split(".")[0] in RUNTIME_MODULES]
    assert not blocked, f"models list imported runtime modules: {blocked[:10]}"


def test_models_verify_does_not_import_runtime(qasr, models_root, make_model, import_blocker):
    make_model(_model_dir(models_root, "Qwen3-ASR-1.7B"))
    proc = qasr.run("--json", "models", "verify", "1.7b", env=import_blocker.env())
    assert_exit(proc, 0, context="models verify without runtime")
    assert find_status(parse_json(proc)) == "complete"
    blocked = [name for name in import_blocker.attempts() if name.split(".")[0] in RUNTIME_MODULES]
    assert not blocked, f"models verify imported runtime modules: {blocked[:10]}"


def test_models_verify_is_repeatable(qasr, models_root, make_model):
    make_model(_model_dir(models_root, "Qwen3-ASR-1.7B"))
    first, _, first_status = _verify(qasr)
    second, _, second_status = _verify(qasr)
    assert first_status == second_status == "complete"
    assert first.returncode == second.returncode == 0
