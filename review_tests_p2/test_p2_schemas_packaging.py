"""P2 packaging/docs/schema contract (P2-TASKS §1/§5, Q16)."""

from __future__ import annotations

import json

import pytest
import tomllib
from helpers import (
    REPO_ROOT,
    assert_exit,
    assert_no_traceback,
    load_schema,
    parse_json,
    schema_path,
    validate_schema,
)

pytestmark = pytest.mark.p2


def test_p2_schemas_exist_and_are_valid_json():
    for name in ("stream.json", "finetune-lint.json"):
        path = schema_path(name)
        assert path.is_file(), f"missing schema {path}"
        value = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(value, dict), name
        assert value.get("type") == "object", name


def test_stream_schema_has_success_and_error_variants():
    """P2-Q16: stream snapshots use oneOf for success/error."""
    schema = load_schema("stream.json")
    assert "oneOf" in schema or "anyOf" in schema, schema


def test_stream_success_sample_validates():
    sample = {
        "schema_version": 1,
        "ok": True,
        "command": "stream",
        "seq": 0,
        "audio_s": 2.0,
        "elapsed_s": 0.41,
        "language": "Chinese",
        "text": "今天天气",
        "is_final": False,
    }
    validate_schema(sample, "stream.json")


def test_stream_final_ok_sample_validates():
    sample = {
        "schema_version": 1,
        "ok": True,
        "command": "stream",
        "seq": 2,
        "audio_s": 5.0,
        "elapsed_s": 1.2,
        "language": None,
        "text": "",
        "is_final": True,
    }
    validate_schema(sample, "stream.json")


def test_stream_error_sample_validates():
    """P2-Q11: terminal error line is its own oneOf variant."""
    sample = {
        "schema_version": 1,
        "ok": False,
        "command": "stream",
        "seq": 3,
        "is_final": True,
        "error": {
            "kind": "quality",
            "code": 5,
            "message": "stream stopped at --max-stream-seconds",
            "retryable": False,
            "details": {},
        },
    }
    validate_schema(sample, "stream.json")


def test_stream_schema_rejects_unknown_success_fields():
    """The line format is additionalProperties:false (P2-TASKS §3.3)."""
    import jsonschema

    sample = {
        "schema_version": 1,
        "ok": True,
        "command": "stream",
        "seq": 0,
        "audio_s": 1.0,
        "elapsed_s": 0.1,
        "language": None,
        "text": "",
        "is_final": False,
        "unknown_field": 1,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(sample, load_schema("stream.json"))


def test_finetune_lint_sample_validates():
    sample = {
        "schema_version": 1,
        "ok": False,
        "command": "finetune",
        "action": "lint-jsonl",
        "path": "train.jsonl",
        "total": 3,
        "valid": 2,
        "errors": [{"line": 2, "field": "audio", "message": "missing"}],
        "warnings": [{"line": 3, "field": "text", "message": "unknown prefix"}],
    }
    validate_schema(sample, "finetune-lint.json")


def test_finetune_items_require_line_field_message():
    """P2-Q16: each error/warning item requires line/field/message."""
    import jsonschema

    schema = load_schema("finetune-lint.json")
    sample = {
        "schema_version": 1,
        "ok": False,
        "command": "finetune",
        "action": "lint-jsonl",
        "path": "train.jsonl",
        "total": 1,
        "valid": 0,
        "errors": [{"line": 1, "message": "missing field"}],
        "warnings": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(sample, schema)


def test_pyproject_has_mic_and_all_extras():
    data = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    extras = data["project"]["optional-dependencies"]
    assert "mic" in extras, extras
    assert any("sounddevice" in item for item in extras["mic"]), extras["mic"]
    assert "all" in extras, extras
    combined = " ".join(extras["all"])
    for name in ("local", "mic", "hf", "modelscope"):
        assert name in combined, f"{name} missing from [all]: {extras['all']}"


def test_readme_documents_p2_surface(repo_root):
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    for token in ("stream", "completion", "finetune", "lint-jsonl"):
        assert token in text, f"{token} missing from README"
    assert "vllm,mic" in text or "stream" in text
    lowered = text.lower()
    assert "macos" in lowered, "README must state macOS lacks vLLM"
    assert "vllm" in lowered, "README must mention vLLM availability constraints"
    for shell in ("bash", "zsh", "fish"):
        assert shell in text, f"README missing {shell} completion install line"


def test_root_help_and_version_do_not_import_heavy_runtimes(qasr, import_blocker):
    blocker, marker = import_blocker
    proc = qasr.run("--version", env={"PYTHONPATH": str(blocker)})
    assert_exit(proc, 0, context="--version")
    assert proc.stdout.strip()
    help_proc = qasr.run("--help", env={"PYTHONPATH": str(blocker)})
    assert_exit(help_proc, 0, context="--help")
    assert_no_traceback(help_proc)
    assert not marker.exists(), marker.read_text(encoding="utf-8")


def test_finetune_help_does_not_import_heavy_runtimes(qasr, import_blocker):
    blocker, marker = import_blocker
    proc = qasr.run("finetune", "--help", env={"PYTHONPATH": str(blocker)})
    assert_exit(proc, 0, context="finetune --help")
    assert "lint-jsonl" in proc.stdout
    assert not marker.exists(), marker.read_text(encoding="utf-8")


def test_languages_json_still_validates_after_p2(qasr):
    proc = qasr.run("languages", "--json")
    assert_exit(proc, 0)
    payload = parse_json(proc)
    validate_schema(payload, "languages.json")
