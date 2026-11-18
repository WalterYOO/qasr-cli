"""SFT JSONL validation for ``qasr finetune lint-jsonl`` (pure Python)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, TextIO

from .errors import InputError
from .languages import LID_LABELS

ALLOWED_FIELDS = ("audio", "text", "prompt")
PREFIX_RE = re.compile(r"^language\s+(.*?)<asr_text>(.*)$", re.DOTALL)
VALID_PREFIX_LANGUAGES = set(LID_LABELS) | {"None"}


@dataclass(frozen=True)
class Issue:
    line: int
    field: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"line": self.line, "field": self.field, "message": self.message}


@dataclass
class LintResult:
    path: str
    total: int = 0
    valid: int = 0
    errors: List[Issue] = field(default_factory=list)
    warnings: List[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def payload(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "ok": self.ok,
            "command": "finetune",
            "action": "lint-jsonl",
            "path": self.path,
            "total": self.total,
            "valid": self.valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def _add(result: LintResult, kind: str, line: int, field_name: str, message: str) -> None:
    issue = Issue(line, field_name, message)
    if kind == "error":
        result.errors.append(issue)
    else:
        result.warnings.append(issue)


def _resolve_audio(raw: str, base_dir: Path) -> Path:
    target = Path(raw).expanduser()
    if not target.is_absolute():
        target = base_dir / target
    return target.resolve()


def lint_lines(
    lines: Any,
    *,
    path: str = "train.jsonl",
    base_dir: Path | None = None,
    check_files: bool = True,
    strict: bool = False,
) -> LintResult:
    """Validate an iterable of raw lines; shared by the CLI and tests."""
    result = LintResult(path=path)
    base = base_dir or Path(".")
    seen_audio: Dict[str, int] = {}
    invalid_lines: set[int] = set()

    for line_no, raw_line in enumerate(lines, start=1):
        raw = str(raw_line).rstrip("\r\n")
        if not raw.strip():
            continue
        result.total += 1

        try:
            record = json.loads(raw)
        except ValueError:
            _add(result, "error", line_no, "json", "invalid JSON")
            invalid_lines.add(line_no)
            continue
        if not isinstance(record, dict):
            _add(result, "error", line_no, "json", "line must be a JSON object")
            invalid_lines.add(line_no)
            continue

        for key in sorted(set(record) - set(ALLOWED_FIELDS)):
            message = f"unknown field '{key}'"
            _add(result, "warning", line_no, key, message)
            if strict:
                _add(result, "error", line_no, key, message)
                invalid_lines.add(line_no)

        audio = record.get("audio")
        audio_ok = True
        if not isinstance(audio, str) or not audio.strip():
            _add(result, "error", line_no, "audio", "audio must be a non-empty string")
            audio_ok = False
            invalid_lines.add(line_no)
        elif "\x00" in audio:
            _add(result, "error", line_no, "audio", "audio path contains NUL")
            audio_ok = False
            invalid_lines.add(line_no)
        else:
            resolved = _resolve_audio(audio, base)
            if check_files:
                if not resolved.is_file():
                    _add(result, "error", line_no, "audio", "audio file not found")
                    audio_ok = False
                    invalid_lines.add(line_no)
                elif not os.access(resolved, os.R_OK):
                    _add(result, "error", line_no, "audio", "audio file is not readable")
                    audio_ok = False
                    invalid_lines.add(line_no)
            if audio_ok:
                key = str(resolved)
                if key in seen_audio:
                    message = (
                        f"duplicate audio path (first seen on line {seen_audio[key]})"
                    )
                    _add(result, "warning", line_no, "audio", message)
                    if strict:
                        _add(result, "error", line_no, "audio", message)
                        invalid_lines.add(line_no)
                else:
                    seen_audio[key] = line_no

        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            _add(result, "error", line_no, "text", "text must be a non-empty string")
            invalid_lines.add(line_no)
        else:
            match = PREFIX_RE.match(text)
            if match and text.lstrip().startswith("language "):
                language = match.group(1).strip()
                body = match.group(2).strip()
                if language not in VALID_PREFIX_LANGUAGES:
                    _add(
                        result,
                        "error",
                        line_no,
                        "text",
                        f"unknown language label '{language}' before <asr_text>",
                    )
                    invalid_lines.add(line_no)
                if not body:
                    _add(
                        result,
                        "error",
                        line_no,
                        "text",
                        "text after <asr_text> must not be empty",
                    )
                    invalid_lines.add(line_no)

        if "prompt" in record and not isinstance(record["prompt"], str):
            _add(result, "error", line_no, "prompt", "prompt must be a string")
            invalid_lines.add(line_no)

    result.valid = max(0, result.total - len(invalid_lines))
    return result


def lint_jsonl(
    path: str | Path,
    *,
    check_files: bool = True,
    strict: bool = False,
) -> LintResult:
    display = str(path)
    target = Path(path).expanduser()
    if not target.is_file():
        raise InputError(f"JSONL file not found: {display}")
    try:
        with target.open("r", encoding="utf-8") as handle:
            return lint_lines(
                handle,
                path=display,
                base_dir=target.resolve().parent,
                check_files=check_files,
                strict=strict,
            )
    except InputError:
        raise
    except (OSError, UnicodeError) as exc:
        raise InputError(f"cannot read JSONL file: {display}") from exc


def format_human(result: LintResult, stream: TextIO) -> None:
    for issue in result.errors:
        print(f"{result.path}:{issue.line}: {issue.field}: {issue.message}", file=stream)
    for issue in result.warnings:
        print(f"{result.path}:{issue.line}: {issue.field}: {issue.message}", file=stream)
    print(
        f"{result.path}: total={result.total} valid={result.valid} "
        f"errors={len(result.errors)} warnings={len(result.warnings)}",
        file=stream,
    )


__all__ = [
    "ALLOWED_FIELDS",
    "Issue",
    "LID_LABELS",
    "LintResult",
    "VALID_PREFIX_LANGUAGES",
    "format_human",
    "lint_jsonl",
    "lint_lines",
]
