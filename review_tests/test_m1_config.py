"""CFG-01..12: config discovery, layering, and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import assert_error, assert_exit, find_values, parse_json


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _lookup(payload: dict, key: str):
    for value in find_values(payload, key):
        if value is None:
            continue
        if isinstance(value, dict):
            for inner in ("value", "effective", "id", "name", "path"):
                if inner in value:
                    return value[inner]
            continue
        return value
    return None


def _show(qasr, *global_args, env=None, cwd=None):
    proc = qasr.run(*global_args, "--json", "config", "show", env=env, cwd=cwd)
    assert_exit(proc, 0, context="config show --json")
    return parse_json(proc)


def _model_name(value) -> str:
    if isinstance(value, dict):
        value = value.get("id") or value.get("name") or value.get("path")
    return str(value)


def test_config_path_is_single_line(qasr):
    proc = qasr.run("config", "path")
    assert_exit(proc, 0, context="config path")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"config path printed {lines!r}"
    path = Path(lines[0])
    assert path.is_absolute(), f"config path is not absolute: {path}"
    assert path.name == "config.json", f"unexpected config filename: {path}"
    assert path.parent == qasr.home, f"default config should live in QASR_HOME: {path}"


def test_config_defaults(qasr):
    values = _show(qasr)
    model = _model_name(_lookup(values, "model"))
    assert model in {"1.7b", "1.7B", "Qwen/Qwen3-ASR-1.7B"} or model.endswith("Qwen3-ASR-1.7B")
    assert _lookup(values, "backend") in {"auto", "transformers", "vllm"}
    device = str(_lookup(values, "device"))
    assert device in {"auto", "cpu", "mps", "cuda"} or device.startswith("cuda:")
    assert _lookup(values, "dtype") in {"auto", "fp16", "fp32", "bf16", "float16", "float32", "bfloat16"}
    assert _lookup(values, "offline") is False


def test_home_config_applies(qasr):
    _write_json(qasr.home / "config.json", {"model": "0.6b"})
    values = _show(qasr)
    assert _model_name(_lookup(values, "model")) == "0.6b"


def test_env_overrides_home_config(qasr):
    _write_json(qasr.home / "config.json", {"model": "0.6b"})
    values = _show(qasr, env={"QASR_MODEL": "1.7b"})
    assert _model_name(_lookup(values, "model")) == "1.7b"


def test_config_flag_overrides_env(qasr):
    _write_json(qasr.home / "config.json", {"model": "0.6b"})
    override = _write_json(qasr.work / "override.json", {"model": "aligner"})
    values = _show(qasr, "--config", str(override), env={"QASR_MODEL": "1.7b"})
    assert _model_name(_lookup(values, "model")) == "aligner"


def test_cli_overrides_config_flag(qasr):
    _write_json(qasr.home / "config.json", {"model": "0.6b"})
    override = _write_json(qasr.work / "override.json", {"model": "aligner"})
    values = _show(qasr, "--config", str(override), "--model", "0.6b", env={"QASR_MODEL": "1.7b"})
    assert _model_name(_lookup(values, "model")) == "0.6b"


def test_project_config_is_found_upwards(qasr):
    project = qasr.work / "project"
    _write_json(project / ".qasr" / "config.json", {"model": "0.6b"})
    nested = project / "deep" / "nested"
    nested.mkdir(parents=True)
    values = _show(qasr, cwd=nested)
    assert _model_name(_lookup(values, "model")) == "0.6b"


def test_qasr_config_env_is_honored(qasr):
    override = _write_json(qasr.work / "env-config.json", {"model": "aligner"})
    values = _show(qasr, env={"QASR_CONFIG": str(override)})
    assert _model_name(_lookup(values, "model")) == "aligner"


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False)])
def test_offline_env_parsing(qasr, value, expected):
    values = _show(qasr, env={"QASR_OFFLINE": value})
    assert _lookup(values, "offline") is expected


def test_invalid_offline_env_is_usage_error(qasr):
    proc = qasr.run("--json", "config", "show", env={"QASR_OFFLINE": "maybe"})
    assert_exit(proc, 2, context="QASR_OFFLINE=maybe")
    assert_error(parse_json(proc), code=2)


@pytest.mark.parametrize(
    "data",
    [
        {"backend": "quantum"},
        {"dtype": "int8"},
        {"device": "tpu"},
    ],
)
def test_invalid_config_values_are_usage_errors(qasr, data):
    path = _write_json(qasr.work / "bad-values.json", data)
    proc = qasr.run("--json", "--config", str(path), "config", "show")
    assert_exit(proc, 2, context=f"invalid config {data}")
    assert_error(parse_json(proc), code=2)


def test_relative_model_dir_is_resolved_against_config_file(qasr):
    cfg_dir = qasr.work / "cfg-dir"
    path = _write_json(cfg_dir / "config.json", {"model_dir": "models"})
    values = _show(qasr, "--config", str(path))
    model_dir = _lookup(values, "model_dir")
    assert model_dir, f"config show did not expose model_dir: {values!r}"
    resolved = Path(str(model_dir)).expanduser()
    assert resolved.is_absolute(), f"model_dir not resolved: {model_dir!r}"
    assert resolved == (cfg_dir / "models"), f"model_dir resolved to {resolved}"
