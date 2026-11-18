"""P2: qasr finetune lint-jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import pytest

from qasr_cli.cli import main

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def validate(name, payload):
    schema = json.loads((SCHEMA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)


def _write_jsonl(path: Path, records) -> Path:
    lines = []
    for record in records:
        if isinstance(record, str):
            lines.append(record)
        else:
            lines.append(json.dumps(record, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(capsys, path, *extra):
    code = main(["--json", "finetune", "lint-jsonl", str(path), *extra])
    out = capsys.readouterr().out
    payload = json.loads(out) if out.strip() else None
    return code, payload


def test_valid_records_and_blank_lines(capsys, tmp_path):
    for name in ("a.wav", "b.wav", "c.wav"):
        (tmp_path / name).write_bytes(b"x")
    audio = tmp_path / "b.wav"
    path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"audio": "a.wav", "text": "你好"},
            "",
            {"audio": str(audio), "text": "language Chinese<asr_text>你好",
             "prompt": "context"},
            {"audio": "c.wav", "text": "language None<asr_text>hello"},
        ],
    )
    code, payload = _run(capsys, path)
    assert code == 0, payload
    validate("finetune-lint", payload)
    assert payload["total"] == 3
    assert payload["valid"] == 3
    assert payload["errors"] == []
    assert payload["warnings"] == []
    assert payload["path"] == str(path)


def test_dialect_prefix_is_accepted(capsys, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    path = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"audio": str(audio), "text": "language Cantonese (Hong Kong)<asr_text>hi"}],
    )
    code, payload = _run(capsys, path)
    assert code == 0
    assert payload["valid"] == 1


def test_invalid_json_and_non_object(capsys, tmp_path):
    path = _write_jsonl(
        tmp_path / "train.jsonl",
        ["{not json", json.dumps([1, 2])],
    )
    code, payload = _run(capsys, path)
    assert code == 4
    validate("finetune-lint", payload)
    assert payload["total"] == 2
    assert payload["valid"] == 0
    assert [issue["line"] for issue in payload["errors"]] == [1, 2]
    assert all(issue["field"] == "json" for issue in payload["errors"])


def test_field_validation(capsys, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"text": "hi"},
            {"audio": "", "text": "hi"},
            {"audio": "a.wav"},
            {"audio": "a.wav", "text": "hi", "prompt": 3},
            {"audio": "a.wav", "text": "language Klingon<asr_text>hi"},
            {"audio": "a.wav", "text": "language Chinese<asr_text>   "},
            {"audio": "a.wav", "text": "language Chinese<asr_text>"},
            {"audio": "miss\x00ing.wav", "text": "hi"},
            {"audio": "missing.wav", "text": "hi"},
            {"audio": str(tmp_path), "text": "hi"},
        ],
    )
    code, payload = _run(capsys, path)
    assert code == 4
    assert payload["total"] == 10
    assert payload["valid"] == 0
    fields = [issue["field"] for issue in payload["errors"]]
    assert fields == [
        "audio", "audio", "text", "prompt", "text", "text", "text", "audio",
        "audio", "audio",
    ]
    assert any("NUL" in issue["message"] for issue in payload["errors"])
    assert any("not found" in issue["message"] for issue in payload["errors"])


def test_unknown_fields_and_duplicate_audio_are_warnings(capsys, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"audio": str(audio), "text": "hi"},
            {"audio": str(audio), "text": "hello", "extra": 1},
        ],
    )
    code, payload = _run(capsys, path)
    assert code == 0, payload
    validate("finetune-lint", payload)
    assert payload["valid"] == 2
    assert payload["errors"] == []
    assert {(issue["line"], issue["field"]) for issue in payload["warnings"]} == {
        (2, "audio"),
        (2, "extra"),
    }


def test_strict_upgrades_warnings_and_keeps_them(capsys, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"audio": str(audio), "text": "hi"},
            {"audio": str(audio), "text": "hello", "extra": 1},
        ],
    )
    code, payload = _run(capsys, path, "--strict")
    assert code == 4
    validate("finetune-lint", payload)
    assert payload["valid"] == 1
    assert len(payload["warnings"]) == 2
    assert {(issue["line"], issue["field"]) for issue in payload["warnings"]} == {
        (2, "audio"),
        (2, "extra"),
    }
    assert {(issue["line"], issue["field"]) for issue in payload["errors"]} == {
        (2, "audio"),
        (2, "extra"),
    }


def test_no_check_files_still_validates_types(capsys, tmp_path):
    path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"audio": "not-there.wav", "text": "hi"},
            {"audio": 5, "text": "hi"},
        ],
    )
    code, payload = _run(capsys, path, "--no-check-files")
    assert code == 4
    assert payload["valid"] == 1
    assert len(payload["errors"]) == 1
    assert payload["errors"][0]["line"] == 2
    assert payload["errors"][0]["field"] == "audio"


def test_missing_jsonl_is_exit_4(capsys, tmp_path):
    code = main(["--json", "finetune", "lint-jsonl", str(tmp_path / "nope.jsonl")])
    payload = json.loads(capsys.readouterr().out)
    assert code == 4
    assert "not found" in payload["error"]["message"]


def test_relative_paths_resolve_against_jsonl_dir(capsys, tmp_path, monkeypatch):
    audio_dir = tmp_path / "data"
    audio_dir.mkdir()
    (audio_dir / "a.wav").write_bytes(b"x")
    path = _write_jsonl(
        audio_dir / "train.jsonl",
        [{"audio": "a.wav", "text": "hi"}],
    )
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    code, payload = _run(capsys, path)
    assert code == 0, payload
    assert payload["path"] == str(path)


def test_human_output_and_summary(capsys, tmp_path):
    path = _write_jsonl(
        tmp_path / "train.jsonl",
        ['{"audio": "missing.wav", "text": "hi"}'],
    )
    code = main(["finetune", "lint-jsonl", str(path)])
    out = capsys.readouterr().out
    assert code == 4
    assert f"{path}:1: audio: audio file not found" in out
    assert "total=1 valid=0 errors=1 warnings=0" in out


def test_help_and_unknown_subcommand(capsys):
    assert main(["finetune", "--help"]) == 0
    assert "lint-jsonl" in capsys.readouterr().out
    code = main(["finetune", "unknown"])
    capsys.readouterr()
    assert code == 2


def test_lint_does_not_import_model_runtimes(capsys, tmp_path, monkeypatch):
    blocker = pytest.importorskip("importlib.abc")
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    path = _write_jsonl(tmp_path / "train.jsonl", [{"audio": "a.wav", "text": "hi"}])

    class _Block(blocker.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in (
                "torch", "vllm", "qwen_asr", "sounddevice", "datasets",
            ):
                raise AssertionError(f"finetune imported {fullname}")
            return None

    for name in list(sys.modules):
        if name.split(".")[0] in ("torch", "vllm", "qwen_asr", "sounddevice", "datasets"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Block(), *sys.meta_path])
    assert main(["finetune", "lint-jsonl", str(path), "--json"]) == 0
    capsys.readouterr()
