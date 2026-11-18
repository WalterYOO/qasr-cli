"""P2 F: ``qasr finetune lint-jsonl`` (contract P2-TASKS §4, Q5/Q12/Q16)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import assert_exit, assert_no_traceback, parse_json, validate_schema

pytestmark = pytest.mark.finetune


def write_jsonl(path: Path, lines: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for line in lines:
        if isinstance(line, str) and line.startswith("\x00raw:"):
            rendered.append(line[len("\x00raw:"):])
        else:
            rendered.append(json.dumps(line, ensure_ascii=False))
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    return path


def lint(qasr, path: Path, *args, name: str | None = None):
    display = name or path.name
    return qasr.run("finetune", "lint-jsonl", display, *args, cwd=path.parent)


def lint_json(qasr, path: Path, *args, name: str | None = None):
    proc = lint(qasr, path, "--json", *args, name=name)
    return proc, parse_json(proc)


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    target = tmp_path / "audio.wav"
    target.write_bytes(b"not-decoded-by-lint")
    return target


def test_valid_plain_lines(qasr, tmp_path, audio_file):
    others = []
    for name in ("second.wav", "third.wav"):
        target = tmp_path / name
        target.write_bytes(b"audio")
        others.append(target)
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"audio": audio_file.name, "text": "你好世界"},
            {"audio": str(others[0]), "text": "hello"},
            {"audio": others[1].name, "text": "带提示", "prompt": "转写这段"},
        ],
    )
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 0, context="valid lint")
    assert payload["ok"] is True
    assert payload["command"] == "finetune"
    assert payload["action"] == "lint-jsonl"
    assert payload["path"] == "train.jsonl"  # P2-Q12: path is echoed as given
    assert payload["total"] == 3
    assert payload["valid"] == 3
    assert payload["errors"] == []
    assert payload["warnings"] == []
    validate_schema(payload, "finetune-lint.json")


@pytest.mark.parametrize(
    "text",
    [
        "language Chinese<asr_text>今天天气",
        "language English<asr_text>hello",
        "language None<asr_text>no language",
    ],
)
def test_known_language_prefixes_are_valid(qasr, tmp_path, audio_file, text):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": audio_file.name, "text": text}],
    )
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 0, context=f"prefix {text!r}")
    assert payload["valid"] == 1
    assert payload["errors"] == []


def test_dialect_prefix_is_valid(qasr, tmp_path, audio_file):
    languages = parse_json(qasr.run("languages", "--json"))
    dialect = languages["dialects"][0]
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": audio_file.name, "text": f"language {dialect}<asr_text>方言"}],
    )
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 0, context=f"dialect {dialect}")
    assert payload["valid"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "language Klingon<asr_text>nope",
        "language Chinese<asr_text>",
        "language Chinese<asr_text>   ",
        "language <asr_text>x",
    ],
)
def test_bad_language_prefix_is_an_error(qasr, tmp_path, audio_file, text):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": audio_file.name, "text": text}],
    )
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 4, context=f"bad prefix {text!r}")
    assert payload["ok"] is False
    assert payload["errors"]
    assert payload["errors"][0]["field"] in {"text"}
    validate_schema(payload, "finetune-lint.json")


@pytest.mark.parametrize(
    "line",
    [
        {"text": "missing audio"},
        {"audio": "", "text": "empty audio"},
        {"audio": 3, "text": "non-string audio"},
        {"audio": "audio.wav"},
        {"audio": "audio.wav", "text": ""},
        {"audio": "audio.wav", "text": 5},
    ],
)
def test_required_fields_are_enforced(qasr, tmp_path, audio_file, line):
    if line.get("audio") == "audio.wav":
        line = {**line, "audio": audio_file.name}
    path = write_jsonl(tmp_path / "train.jsonl", [line])
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 4, context=f"bad line {line!r}")
    assert payload["errors"]
    assert payload["valid"] == 0


def test_prompt_must_be_a_string(qasr, tmp_path, audio_file):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": audio_file.name, "text": "x", "prompt": 3}],
    )
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 4)
    assert payload["errors"][0]["field"] == "prompt"


@pytest.mark.parametrize("kind", ["missing", "directory", "nul"])
def test_audio_path_checks(qasr, tmp_path, kind):
    (tmp_path / "subdir").mkdir()
    audio = {
        "missing": "missing.wav",
        "directory": "subdir",
        "nul": "bad\u0000name.wav",
    }[kind]
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": audio, "text": "x"}],
    )
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 4, context=f"audio {kind}")
    assert payload["errors"]
    assert payload["errors"][0]["field"] == "audio"


def test_no_check_files_skips_existence_but_not_type(qasr, tmp_path):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"audio": "does-not-exist.wav", "text": "ok"},
            {"audio": "bad\u0000name.wav", "text": "still bad"},
        ],
    )
    proc, payload = lint_json(qasr, path, "--no-check-files")
    assert_exit(proc, 4, context="no-check-files + NUL")
    assert payload["total"] == 2
    assert payload["valid"] == 1
    assert [item["line"] for item in payload["errors"]] == [2]


def test_unknown_field_is_a_warning(qasr, tmp_path, audio_file):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": audio_file.name, "text": "x", "extra": "ignored"}],
    )
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 0, context="unknown field warning")
    assert payload["ok"] is True
    assert payload["valid"] == 1
    assert payload["errors"] == []
    assert len(payload["warnings"]) == 1
    warning = payload["warnings"][0]
    assert warning["line"] == 1
    assert warning["field"] == "extra"
    validate_schema(payload, "finetune-lint.json")


def test_unknown_field_is_an_error_under_strict(qasr, tmp_path, audio_file):
    """P2-Q5: every warning is mirrored into errors under --strict."""
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": audio_file.name, "text": "x", "extra": "ignored"}],
    )
    proc, payload = lint_json(qasr, path, "--strict")
    assert_exit(proc, 4, context="strict unknown field")
    assert payload["ok"] is False
    assert payload["valid"] == 0
    assert len(payload["warnings"]) == 1
    assert len(payload["errors"]) == 1
    assert payload["errors"][0]["field"] == payload["warnings"][0]["field"]
    assert payload["errors"][0]["line"] == payload["warnings"][0]["line"]


def test_duplicate_audio_is_a_warning(qasr, tmp_path, audio_file):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"audio": audio_file.name, "text": "one"},
            {"audio": f"./{audio_file.name}", "text": "two"},
        ],
    )
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 0, context="duplicate audio warning")
    assert payload["ok"] is True
    assert payload["warnings"]
    assert payload["warnings"][0]["field"] == "audio"


def test_duplicate_audio_is_an_error_under_strict(qasr, tmp_path, audio_file):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"audio": audio_file.name, "text": "one"},
            {"audio": f"./{audio_file.name}", "text": "two"},
        ],
    )
    proc, payload = lint_json(qasr, path, "--strict")
    assert_exit(proc, 4, context="strict duplicate audio")
    assert payload["ok"] is False
    assert any(item["field"] == "audio" for item in payload["errors"])


def test_blank_lines_are_skipped_and_not_counted(qasr, tmp_path, audio_file):
    raw = "\n".join(
        [
            json.dumps({"audio": audio_file.name, "text": "a"}),
            "",
            "   ",
            json.dumps({"audio": audio_file.name, "text": "b"}),
        ]
    )
    path = tmp_path / "train.jsonl"
    path.write_text(raw + "\n", encoding="utf-8")
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 0)
    assert payload["total"] == 2
    assert payload["valid"] == 2


def test_invalid_json_line_is_reported_and_later_lines_still_checked(
    qasr, tmp_path, audio_file
):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [
            "\x00raw:{not valid json",
            {"audio": audio_file.name, "text": "fine"},
        ],
    )
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 4, context="invalid JSON line")
    assert payload["total"] == 2
    assert payload["valid"] == 1
    assert payload["errors"][0]["line"] == 1
    assert payload["errors"][0]["field"] == "json"  # P2-Q12
    validate_schema(payload, "finetune-lint.json")


@pytest.mark.parametrize("bad", [["array"], "scalar", 3, None])
def test_non_object_json_lines_are_errors(qasr, tmp_path, bad):
    path = write_jsonl(tmp_path / "train.jsonl", [bad])
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 4, context=f"non-object {bad!r}")
    assert payload["errors"]


def test_missing_jsonl_file_is_an_input_error(qasr, tmp_path):
    proc = qasr.run("finetune", "lint-jsonl", "nope.jsonl", "--json", cwd=tmp_path)
    assert_exit(proc, 4, context="missing JSONL")
    payload = parse_json(proc)
    assert payload["ok"] is False
    assert_no_traceback(proc)


def test_jsonl_directory_is_an_input_error(qasr, tmp_path):
    (tmp_path / "subdir").mkdir()
    proc = qasr.run("finetune", "lint-jsonl", "subdir", "--json", cwd=tmp_path)
    assert_exit(proc, 4)


def test_warning_only_is_ok_without_strict(qasr, tmp_path, audio_file):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": audio_file.name, "text": "x", "extra": 1}],
    )
    proc, payload = lint_json(qasr, path)
    assert_exit(proc, 0)
    assert payload["ok"] is True
    assert payload["warnings"]


def test_human_output_has_path_line_field_and_summary(qasr, tmp_path):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": "missing.wav", "text": "x"}],
    )
    proc = lint(qasr, path)
    assert_exit(proc, 4)
    combined = proc.stdout + proc.stderr
    assert "train.jsonl:1" in combined
    assert "audio" in combined
    assert "summary" in combined.lower() or "total" in combined.lower()
    assert_no_traceback(proc)


def test_lint_does_not_import_heavy_runtimes(qasr, tmp_path, import_blocker, audio_file):
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": audio_file.name, "text": "x"}],
    )
    blocker, marker = import_blocker
    proc = qasr.run(
        "finetune", "lint-jsonl", path.name, "--json",
        cwd=path.parent,
        env={"PYTHONPATH": str(blocker)},
    )
    assert_exit(proc, 0, context="lint without torch/datasets")
    assert not marker.exists(), marker.read_text(encoding="utf-8")
    assert parse_json(proc)["ok"] is True


def test_lint_never_echoes_serve_token(qasr, tmp_path, audio_file):
    secret = "{{SECRET_v3pj8q2m}}"
    path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": audio_file.name, "text": "x"}],
    )
    proc = qasr.run(
        "finetune", "lint-jsonl", path.name, "--json",
        cwd=path.parent, env={"QASR_SERVE_TOKEN": secret},
    )
    assert secret not in proc.stdout + proc.stderr


def test_finetune_help_lists_subcommand_and_rejects_unknown(qasr):
    proc = qasr.run("finetune", "--help")
    assert_exit(proc, 0)
    assert "lint-jsonl" in proc.stdout
    bad = qasr.run("finetune", "bogus")
    assert_exit(bad, 2, context="unknown finetune subcommand")
    assert_no_traceback(bad)
