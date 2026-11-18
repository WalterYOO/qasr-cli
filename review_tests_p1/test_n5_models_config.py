"""N5: models show/rm path safety and config get/set semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import assert_exit, parse_json, validate_schema

pytestmark = pytest.mark.n5


def _config_file(qasr) -> Path:
    return Path(qasr.home) / "config.json"


def test_models_show_complete_model(qasr, complete_model):
    model = complete_model("Qwen3-ASR-0.6B")
    proc = qasr.run("models", "show", "0.6b", "--json")
    assert_exit(proc, 0, context="models show")
    payload = parse_json(proc)
    validate_schema(payload, "models-show.json")
    body = payload.get("model") or payload
    assert str(body.get("path")) == str(model)
    alias = body.get("alias")
    assert alias in {"0.6b", "Qwen3-ASR-0.6B"} or str(body.get("model")) in {
        "0.6b", "Qwen3-ASR-0.6B"
    }
    assert int(body.get("size_bytes") or body.get("size") or 0) > 0
    assert int(body.get("file_count") or body.get("files") or 0) >= 5
    verify = str(body.get("verify") or body.get("verify_status") or "").lower()
    assert verify in {"complete", "ok", "true"} or body.get("present") is True
    assert body.get("repo")
    # ``source`` is the resolution origin (cli/config/env), not the download
    # source; a CLI alias resolves with source="cli".
    assert body.get("source") in {"cli", "local", "cache", "modelscope", "hf", None}


def test_models_show_incomplete_model(qasr, incomplete_model):
    incomplete_model("Qwen3-ASR-0.6B")
    proc = qasr.run("models", "show", "0.6b", "--json")
    assert_exit(proc, 0, context="models show incomplete")
    payload = parse_json(proc)
    text = json.dumps(payload).lower()
    assert "incomplete" in text or "partial" in text or "missing" in text


def test_models_show_unknown_alias_reports_missing(qasr):
    proc = qasr.run("models", "show", "klingon", "--json")
    # T5.5 only requires an alias-resolution view; an unknown value is
    # informational (present=false), not an input failure. The task book
    # does not define an exit code for this case.
    assert_exit(proc, 0, context="models show unknown")
    payload = parse_json(proc)
    body = payload.get("model") or payload
    assert body.get("present") is False, payload
    assert str(body.get("status")).lower() in {"missing", "incomplete", "absent"}
    assert "Traceback" not in proc.stderr


def test_models_rm_requires_yes_in_non_tty(qasr, complete_model):
    model = complete_model("Qwen3-ASR-0.6B")
    proc = qasr.run("models", "rm", "0.6b", "--json")
    assert_exit(proc, 2, context="models rm without --yes")
    assert model.is_dir()


def test_models_rm_dry_run_keeps_model(qasr, complete_model):
    model = complete_model("Qwen3-ASR-0.6B")
    proc = qasr.run("models", "rm", "0.6b", "--yes", "--dry-run", "--json")
    assert_exit(proc, 0, context="models rm dry-run")
    assert model.is_dir()
    assert str(model) in proc.stdout


def test_models_rm_deletes_managed_model(qasr, complete_model):
    model = complete_model("Qwen3-ASR-0.6B")
    proc = qasr.run("models", "rm", "0.6b", "--yes", "--json")
    assert_exit(proc, 0, context="models rm")
    assert not model.exists()
    payload = parse_json(proc)
    text = json.dumps(payload).lower()
    assert "bytes" in text or "deleted" in text or "freed" in text


def test_models_rm_rejects_symlink_target(qasr, models_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    alias_dir = models_root / "Qwen3-ASR-0.6B"
    alias_dir.symlink_to(outside, target_is_directory=True)
    proc = qasr.run("models", "rm", "0.6b", "--yes", "--json")
    assert_exit(proc, 2, context="models rm symlink")
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "target,expected",
    [
        ("../outside", 2),
        ("..", 2),
        (".", 2),
        ("/tmp", 2),
    ],
)
def test_models_rm_rejects_non_alias_paths(qasr, target, expected):
    proc = qasr.run("models", "rm", target, "--yes", "--json")
    assert_exit(proc, expected, context=f"models rm {target}")
    assert "Traceback" not in proc.stderr


def test_models_rm_accepts_repo_id_dry_run(qasr, complete_model):
    model = complete_model("Qwen3-ASR-0.6B")
    proc = qasr.run(
        "models", "rm", "Qwen/Qwen3-ASR-0.6B", "--yes", "--dry-run", "--json"
    )
    assert_exit(proc, 0, context="models rm repo id")
    payload = parse_json(proc)
    assert str(payload.get("path")) == str(model), payload
    assert model.is_dir()


def test_models_rm_accepts_absolute_managed_path_dry_run(qasr, complete_model):
    model = complete_model("Qwen3-ASR-0.6B")
    proc = qasr.run(
        "models", "rm", str(model), "--yes", "--dry-run", "--json"
    )
    assert_exit(proc, 0, context="models rm absolute path")
    payload = parse_json(proc)
    assert str(payload.get("path")) == str(model), payload
    assert model.is_dir()


def test_models_rm_rejects_dotdot_even_when_inside_models_root(
    qasr, complete_model, models_root
):
    complete_model("Qwen3-ASR-0.6B")
    proc = qasr.run(
        "models", "rm", "../models/Qwen3-ASR-0.6B",
        "--yes", "--dry-run", "--json", cwd=models_root,
    )
    assert_exit(proc, 2, context="models rm .. inside root")


def test_config_get_value_and_json(qasr):
    plain = qasr.run("config", "get", "backend")
    assert_exit(plain, 0, context="config get backend")
    assert "auto" in plain.stdout
    proc = qasr.run("config", "get", "backend", "--json")
    assert_exit(proc, 0, context="config get --json")
    payload = parse_json(proc)
    assert payload.get("key") == "backend"
    assert payload.get("value") == "auto"
    assert payload.get("source")


def test_config_get_matches_schema(qasr):
    proc = qasr.run("config", "get", "backend", "--json")
    assert_exit(proc, 0, context="config get schema")
    validate_schema(parse_json(proc), "config-get.json")


def test_config_get_unknown_key_is_exit_2(qasr):
    proc = qasr.run("config", "get", "nope", "--json")
    assert_exit(proc, 2, context="config get unknown")


def test_config_set_home_persists_and_preserves_unknown_keys(qasr):
    path = _config_file(qasr)
    path.write_text('{"custom_key": "keep"}', encoding="utf-8")
    proc = qasr.run("config", "set", "device", "cpu", "--json")
    assert_exit(proc, 0, context="config set")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["device"] == "cpu"
    assert data["custom_key"] == "keep"
    read = qasr.run("config", "get", "device", "--json")
    assert_exit(read, 0, context="config get device")
    assert parse_json(read)["value"] == "cpu"


@pytest.mark.parametrize(
    "key,value",
    [
        ("backend", "bogus"),
        ("device", "tpu"),
        ("dtype", "int4"),
        ("offline", "maybe"),
    ],
)
def test_config_set_enum_validation(qasr, key, value):
    proc = qasr.run("config", "set", key, value, "--json")
    assert_exit(proc, 2, context=f"config set {key}={value}")
    assert not _config_file(qasr).exists() or value not in _config_file(qasr).read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "key", ["QASR_SERVE_TOKEN", "serve_token", "token", "api_key", "password"]
)
def test_config_set_rejects_token_keys(qasr, key):
    proc = qasr.run("config", "set", key, "secret", "--json")
    assert_exit(proc, 2, context=f"config set {key}")
    assert not _config_file(qasr).exists()


def test_config_set_server_is_allowed_and_env_wins(qasr):
    proc = qasr.run("config", "set", "server", "http://config-host:8000", "--json")
    assert_exit(proc, 0, context="config set server")
    env_value = "http://env-host:9000"
    get = qasr.run(
        "config", "get", "server", "--json",
        env={"QASR_SERVE_URL": env_value},
    )
    assert_exit(get, 0, context="config get server env")
    payload = parse_json(get)
    assert payload["value"] == env_value
    assert str(payload["source"]).startswith(("env", "environment", "QASR_SERVE_URL"))


def test_config_set_project_uses_nearest_git_root(qasr, tmp_path):
    project = tmp_path / "project"
    nested = project / "sub" / "dir"
    nested.mkdir(parents=True)
    (project / ".git").mkdir()
    proc = qasr.run(
        "config", "set", "backend", "vllm", "--project", "--json",
        cwd=nested,
    )
    assert_exit(proc, 0, context="config set --project")
    target = project / ".qasr" / "config.json"
    assert target.is_file(), target
    assert json.loads(target.read_text(encoding="utf-8"))["backend"] == "vllm"
    assert not (nested / ".qasr" / "config.json").exists()


def test_config_set_project_uses_pyproject_root(qasr, tmp_path):
    project = tmp_path / "project-py"
    nested = project / "sub"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    proc = qasr.run(
        "config", "set", "dtype", "fp32", "--project", "--json", cwd=nested
    )
    assert_exit(proc, 0, context="config set --project pyproject")
    assert (project / ".qasr" / "config.json").is_file()


def test_config_set_project_without_marker_uses_cwd(qasr, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    proc = qasr.run(
        "config", "set", "backend", "vllm", "--project", "--json", cwd=plain
    )
    # T5.6: fall back to cwd when no .git/pyproject.toml marker is found.
    assert_exit(proc, 0, context="config --project without root")
    target = plain / ".qasr" / "config.json"
    assert target.is_file(), target
    assert json.loads(target.read_text(encoding="utf-8"))["backend"] == "vllm"


def test_config_set_project_rejects_symlinked_parent(qasr, tmp_path):
    project = tmp_path / "project-parent-link"
    project.mkdir()
    (project / ".git").mkdir()
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (project / ".qasr").symlink_to(outside, target_is_directory=True)
    proc = qasr.run(
        "config", "set", "backend", "vllm", "--project", "--json",
        cwd=project,
    )
    assert_exit(proc, 2, context="config set symlinked parent")
    assert not (outside / "config.json").exists()


def test_config_set_project_rejects_symlink_file(qasr, tmp_path):
    project = tmp_path / "project-link"
    config_dir = project / ".qasr"
    config_dir.mkdir(parents=True)
    (project / ".git").mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"backend": "auto"}', encoding="utf-8")
    (config_dir / "config.json").symlink_to(outside)
    proc = qasr.run(
        "config", "set", "backend", "vllm", "--project", "--json",
        cwd=project,
    )
    assert proc.returncode in {2, 4}, proc.stderr
    assert json.loads(outside.read_text(encoding="utf-8"))["backend"] == "auto"


def test_config_show_includes_p1_keys(qasr):
    proc = qasr.run("config", "show", "--json")
    assert_exit(proc, 0, context="config show")
    payload = parse_json(proc)
    text = json.dumps(payload)
    for key in ("server", "timeout", "retries", "window_seconds",
                "window_overlap", "max_chunk_seconds"):
        assert key in text, f"{key} missing from config show: {payload!r}"


def test_config_jobs_is_accepted_but_forced_to_one_locally(qasr, empty_wavs):
    proc = qasr.run("config", "set", "jobs", "4", "--json")
    assert_exit(proc, 0, context="config set jobs")
    run = qasr.run("transcribe", *empty_wavs, "--json")
    assert_exit(run, 0, context="config jobs local")
    assert "jobs" in run.stderr.lower()
