"""CFG consumption: config values must affect the request, not just config show."""

from __future__ import annotations

import json
from pathlib import Path

from helpers import assert_exit, empty_wav


def _write_config(qasr, data: dict) -> Path:
    path = qasr.work / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _job_key(out: Path) -> str:
    state = out / ".qasr" / "state.jsonl"
    records = [
        json.loads(line)
        for line in state.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done = [record for record in records if record.get("status") == "done"]
    assert done, f"no done record in {state}:\n{records!r}"
    key = done[-1].get("job_key")
    assert isinstance(key, str) and key, f"missing job_key in {done[-1]!r}"
    return key


def test_config_format_controls_output_extension(qasr):
    audio = empty_wav(qasr.work / "empty.wav")
    config = _write_config(qasr, {"model": "/nonexistent/model", "format": ["json"]})
    out = qasr.work / "out-format"

    proc = qasr.run(
        "transcribe",
        audio,
        "-o",
        out,
        env={"QASR_CONFIG": str(config)},
    )
    assert_exit(proc, 0, context="config format")
    files = sorted(path.name for path in out.glob("empty.*"))
    assert files == ["empty.json"], f"config format was ignored, wrote: {files}"


def test_config_context_is_base_and_cli_appends_in_order(qasr):
    audio = empty_wav(qasr.work / "empty.wav")
    base = {"model": "/nonexistent/model", "context": ["alpha", "beta"]}
    empty = {"model": "/nonexistent/model"}

    def run(name: str, config_data: dict, *cli_args: str) -> str:
        config = qasr.work / f"{name}.json"
        config.write_text(json.dumps(config_data), encoding="utf-8")
        out = qasr.work / f"out-{name}"
        proc = qasr.run(
            "transcribe",
            audio,
            "-o",
            out,
            *cli_args,
            env={"QASR_CONFIG": str(config)},
        )
        assert_exit(proc, 0, context=name)
        return _job_key(out)

    config_only = run("config-only", base)
    config_plus_cli = run("config-plus-cli", base, "--context=gamma")
    cli_only = run("cli-only", empty, "--context=alpha", "--context=beta", "--context=gamma")
    config_dedup = run("config-dedup", base, "--context=beta")

    assert config_plus_cli == cli_only, (
        "config context must be the base and CLI context must append in order; "
        f"config+cli={config_plus_cli} cli-only={cli_only}"
    )
    assert config_dedup == config_only, (
        "CLI context already present in config must be de-duplicated; "
        f"dedup={config_dedup} config-only={config_only}"
    )
