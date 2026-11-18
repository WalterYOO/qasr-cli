"""CLI-01..14: entry points, usage errors, and no-runtime startup."""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest
from conftest import REPO_ROOT, RUNTIME_MODULES
from helpers import assert_error, assert_exit, parse_json


def test_version_is_single_line(qasr):
    proc = qasr.run("--version")
    assert_exit(proc, 0, context="qasr --version")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"expected one version line, got: {lines!r}"
    assert re.search(r"\d+\.\d+\.\d+", lines[0]), f"no semver in {lines[0]!r}"
    assert proc.stderr.strip() == "", f"version wrote to stderr: {proc.stderr!r}"


def test_help_lists_p0_commands(qasr):
    proc = qasr.run("--help")
    assert_exit(proc, 0, context="qasr --help")
    text = proc.stdout
    for command in ("transcribe", "align", "models", "languages", "doctor", "config"):
        assert command in text, f"--help is missing {command!r}"
    assert "usage" in text.lower()
    assert proc.stderr.strip() == "", f"--help wrote to stderr: {proc.stderr!r}"


def test_module_entrypoint(qasr):
    if qasr.uses_module:
        pytest.skip("runner already uses python -m qasr_cli")
    python = os.environ.get("QASR_PYTHON")
    if not python:
        pytest.skip("module entrypoint needs QASR_PYTHON when QASR_BIN is used")
    proc = subprocess.run(
        [python, "-m", "qasr_cli", "--version"],
        cwd=str(REPO_ROOT),
        env=qasr.env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert_exit(proc, 0, context="python -m qasr_cli --version")


def test_help_never_imports_runtime(qasr, import_blocker):
    proc = qasr.run("--help", env=import_blocker.env())
    assert_exit(proc, 0, context="--help with runtime imports blocked")
    attempts = import_blocker.attempts()
    assert attempts, "import audit hook did not run; PYTHONPATH/sitecustomize injection failed"
    blocked = [name for name in attempts if name.split(".")[0] in RUNTIME_MODULES]
    assert not blocked, f"--help imported runtime modules: {blocked[:10]}"


def test_version_never_imports_runtime(qasr, import_blocker):
    proc = qasr.run("--version", env=import_blocker.env())
    assert_exit(proc, 0, context="--version with runtime imports blocked")
    blocked = [name for name in import_blocker.attempts() if name.split(".")[0] in RUNTIME_MODULES]
    assert not blocked, f"--version imported runtime modules: {blocked[:10]}"


def test_unknown_command_json(qasr):
    proc = qasr.run("--json", "bogus")
    assert_exit(proc, 2, context="unknown command")
    payload = parse_json(proc)
    assert_error(payload, code=2, kind="usage")
    assert payload.get("schema_version") == 1


def test_missing_positional_json(qasr):
    proc = qasr.run("--json", "transcribe")
    assert_exit(proc, 2, context="transcribe without input")
    payload = parse_json(proc)
    assert_error(payload, code=2, kind="usage")
    assert payload.get("command") == "transcribe"


def test_json_and_jsonl_are_mutually_exclusive(qasr):
    proc = qasr.run("--json", "--jsonl", "languages")
    assert_exit(proc, 2, context="--json --jsonl")
    if proc.stdout.strip():
        assert_error(parse_json(proc), code=2, kind="usage")


@pytest.mark.parametrize("command", ["stream"])
def test_p1_commands_are_not_available(qasr, command):
    proc = qasr.run(command)
    assert_exit(proc, 2, context=f"P1 command {command}")


@pytest.mark.parametrize("flag", ["--skip-existing", "--retry", "--stdin-list"])
def test_p1_transcribe_flags_are_accepted(qasr, flag):
    """P1 flags must parse (the P1 suite asserts their full semantics)."""
    proc = qasr.run("transcribe", flag, "x.wav", stdin="")
    assert_exit(proc, (2, 3, 4), context=f"P1 flag {flag}")
    assert "unrecognized arguments" not in proc.stderr.lower(), proc.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("--device", "quantum", "transcribe", "x.wav"),
        ("--dtype", "int8", "transcribe", "x.wav"),
        ("--backend", "quantum", "transcribe", "x.wav"),
    ],
)
def test_invalid_global_enum_is_usage_error(qasr, args):
    proc = qasr.run("--json", *args)
    assert_exit(proc, 2, context=" ".join(args))
    assert_error(parse_json(proc), code=2)


def test_missing_config_file_is_usage_error(qasr):
    proc = qasr.run("--json", "--config", str(qasr.work / "nope.json"), "config", "show")
    assert_exit(proc, 2, context="missing --config")
    assert_error(parse_json(proc), code=2)


def test_invalid_config_json_is_usage_error(qasr):
    path = qasr.work / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    proc = qasr.run("--json", "--config", str(path), "config", "show")
    assert_exit(proc, 2, context="invalid --config")
    assert_error(parse_json(proc), code=2)


def test_languages_json_is_single_value(qasr):
    proc = qasr.run("--json", "languages")
    assert_exit(proc, 0, context="languages --json")
    payload = parse_json(proc)
    assert payload.get("ok") is True
    assert payload.get("schema_version") == 1


def test_json_after_subcommand_with_model_error(qasr, make_wav):
    """Only the model is broken; this isolates the flag-position contract."""
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = qasr.run("--model", "/nonexistent/model", "transcribe", "--json", audio)
    assert_exit(proc, 3, context="subcommand-level --json, missing model")
    payload = parse_json(proc)
    assert payload.get("ok") is False
    assert json.dumps(payload).strip()


def test_input_error_precedes_model_resolution(qasr, make_model):
    """Invalid input must win over a missing/unloadable model (exit 4, not 3)."""
    model = make_model(qasr.work / "synthetic-model")
    proc = qasr.run(
        "--model",
        str(model),
        "transcribe",
        "--json",
        qasr.work / "missing.wav",
    )
    assert_exit(proc, 4, context="input error must precede model resolution")
    payload = parse_json(proc)
    assert payload.get("ok") is False
    assert payload.get("command") == "transcribe"
