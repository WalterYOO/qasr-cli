import json
from pathlib import Path

import pytest

from qasr_cli.config import (
    CONFIG_KEYS,
    DEFAULTS,
    active_config_path,
    load_config,
    qasr_home,
)
from qasr_cli.errors import ConfigError


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_defaults(isolated_home):
    cfg = load_config(cwd=isolated_home)
    for key in CONFIG_KEYS:
        assert cfg.get(key) == DEFAULTS[key]
        assert cfg.source_of(key) == "default"


def test_home_then_project_then_env_then_cli(isolated_home, tmp_path, monkeypatch):
    _write(isolated_home / "config.json", {"model": "0.6b", "batch_size": 2})
    project = tmp_path / "project"
    project.mkdir()
    _write(project / ".qasr" / "config.json", {"model": "aligner"})

    cfg = load_config(cwd=project)
    assert cfg.get("model") == "aligner"  # project beats home
    assert cfg.get("batch_size") == 2

    monkeypatch.setenv("QASR_MODEL", "1.7b")
    cfg = load_config(cwd=project)
    assert cfg.get("model") == "1.7b"  # env beats files
    assert cfg.source_of("model") == "env:QASR_MODEL"

    cli = load_config(cwd=project, cli_overrides={"model": "0.6b"})
    assert cli.get("model") == "0.6b"
    assert cli.source_of("model") == "cli"


def test_explicit_config_beats_env(isolated_home, tmp_path, monkeypatch):
    _write(isolated_home / "config.json", {"model": "0.6b"})
    explicit = _write(tmp_path / "custom.json", {"model": "aligner"})
    monkeypatch.setenv("QASR_MODEL", "1.7b")
    cfg = load_config(str(explicit), cwd=isolated_home)
    assert cfg.get("model") == "aligner"


def test_qasr_config_env_selects_file(isolated_home, tmp_path, monkeypatch):
    selected = _write(tmp_path / "env.json", {"device": "cpu"})
    monkeypatch.setenv("QASR_CONFIG", str(selected))
    cfg = load_config(cwd=isolated_home)
    assert cfg.get("device") == "cpu"
    assert active_config_path(cwd=isolated_home) == selected


def test_unknown_key_is_ignored_for_forward_compatibility(isolated_home):
    _write(isolated_home / "config.json", {"nope": 1, "model": "0.6b"})
    cfg = load_config(cwd=isolated_home)
    assert cfg.get("model") == "0.6b"
    assert "nope" not in cfg.as_dict()


def test_bad_type_is_rejected(isolated_home):
    _write(isolated_home / "config.json", {"batch_size": "many"})
    with pytest.raises(ConfigError):
        load_config(cwd=isolated_home)


def test_offline_env_coercion(isolated_home, monkeypatch):
    monkeypatch.setenv("QASR_OFFLINE", "yes")
    cfg = load_config(cwd=isolated_home)
    assert cfg.get("offline") is True


def test_missing_explicit_config(isolated_home, tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "missing.json"), cwd=isolated_home)


def test_relative_model_dir_resolves_against_config_file(isolated_home, tmp_path):
    project = tmp_path / "proj"
    _write(project / ".qasr" / "config.json", {"model_dir": "models"})
    cfg = load_config(cwd=project)
    assert cfg.get("model_dir") == str((project / ".qasr" / "models").resolve())


def test_qasr_home_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("QASR_HOME", str(tmp_path / "custom"))
    assert qasr_home() == (tmp_path / "custom").resolve()
