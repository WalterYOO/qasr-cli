"""P1 schema inventory and backward-compatibility checks."""

from __future__ import annotations

import json

import pytest
from helpers import load_schema

N5_SCHEMAS = (
    pytest.param("jobs-list.json"),
    pytest.param("jobs-show.json"),
    pytest.param("bench.json"),
    pytest.param("models-show.json"),
    pytest.param("config-get.json"),
)


@pytest.mark.parametrize("name", N5_SCHEMAS)
def test_p1_schema_exists_and_is_valid_json_schema(name):
    schema = load_schema(name)
    assert schema.get("$schema"), f"{name} has no $schema"
    assert schema.get("type") == "object", f"{name} must describe an object"
    required = schema.get("required") or []
    assert "schema_version" in required, f"{name} must require schema_version"
    assert "ok" in required, f"{name} must require ok"
    assert "command" in required, f"{name} must require command"


def test_transcribe_schema_allows_remote_language_source():
    schema = load_schema("transcribe-item.json")
    language_source = (
        schema["properties"]["result"]["properties"]["language_source"]
    )
    allowed = set(language_source.get("enum") or [])
    assert "server" in allowed, (
        "remote transcription emits language_source=server; "
        f"transcribe-item.json allows {sorted(allowed)}"
    )


def test_align_schema_declares_optional_language_source():
    schema = load_schema("align.json")
    properties = schema.get("properties") or {}
    field = properties.get("language_source")
    assert field is not None, (
        "P1-TASKS T4.4 requires optional align.language_source; schema does not declare it"
    )
    allowed = set(field.get("enum") or [])
    assert {"explicit", "script", "lid_probe"} <= allowed, allowed


def test_dry_run_schema_allows_window_plan():
    schema = load_schema("dry-run.json")
    plan = schema["properties"]["plan"]
    assert plan.get("additionalProperties") is not False
    # Declaring "windows" is stronger than relying on additionalProperties.
    assert "windows" in (plan.get("properties") or {}), (
        "align --dry-run must publish a window plan; declare plan.windows in dry-run.json"
    )


def test_all_schemas_remain_valid_json():
    from helpers import schema_path

    for path in sorted(schema_path("dummy.json").parent.glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
