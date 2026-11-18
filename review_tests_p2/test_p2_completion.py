"""P2 C: ``qasr completion bash|zsh|fish`` (contract P2-TASKS §3, Q3/Q13)."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest
from helpers import assert_exit, assert_no_traceback

pytestmark = pytest.mark.completion

SHELLS = ("bash", "zsh", "fish")


def _top_level_commands(qasr) -> list[str]:
    proc = qasr.run("--help")
    assert_exit(proc, 0, context="qasr --help")
    commands: list[str] = []
    in_positionals = False
    for line in proc.stdout.splitlines():
        if line.strip() == "positional arguments:":
            in_positionals = True
            continue
        if in_positionals:
            if line.startswith("  ") and not line.startswith("    "):
                # The "<command>" metavar line that precedes the choices.
                continue
            if not line.startswith("    "):
                if line.strip():
                    break
                continue
            match = re.match(r"^    ([a-z][a-z0-9-]+)\s{2,}\S", line)
            if match:
                commands.append(match.group(1))
    assert commands, proc.stdout
    return commands


def _long_options(help_text: str) -> set[str]:
    options = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", help_text))
    # argparse always injects -h/--help; completion scripts conventionally
    # omit it, and it carries no command-specific contract.
    options.discard("--help")
    return options


def _completion(qasr, shell: str):
    proc = qasr.run("completion", shell)
    assert_exit(proc, 0, context=f"completion {shell}")
    return proc


@pytest.mark.parametrize("shell", SHELLS)
def test_completion_succeeds_with_single_trailing_newline(qasr, shell):
    proc = _completion(qasr, shell)
    assert proc.stdout.endswith("\n"), repr(proc.stdout[-50:])
    assert not proc.stdout.endswith("\n\n"), repr(proc.stdout[-50:])
    assert proc.stdout.strip(), "completion script is empty"


@pytest.mark.parametrize("shell", SHELLS)
def test_completion_covers_every_top_level_command(qasr, shell):
    proc = _completion(qasr, shell)
    for command in _top_level_commands(qasr):
        assert command in proc.stdout, f"{command} missing from {shell} script"
    for required in ("transcribe", "align", "serve", "stream", "finetune", "completion"):
        assert required in proc.stdout, f"{required} missing from {shell} script"


@pytest.mark.parametrize("shell", SHELLS)
def test_completion_covers_all_long_options_from_help(qasr, shell, tmp_path):
    """P2-Q3: every long option in ``qasr <cmd> --help`` appears in the script."""
    script = _completion(qasr, shell).stdout
    root_help = qasr.run("--help")
    assert_exit(root_help, 0)
    expected = _long_options(root_help.stdout)
    for command in _top_level_commands(qasr):
        proc = qasr.run(command, "--help")
        if proc.returncode != 0:
            continue
        expected |= _long_options(proc.stdout)
    missing = sorted(option for option in expected if option not in script)
    assert not missing, f"{shell} completion misses {missing}"


@pytest.mark.parametrize("shell", SHELLS)
def test_completion_script_is_syntactically_valid(qasr, shell, tmp_path):
    script = _completion(qasr, shell).stdout
    path = tmp_path / f"completion.{shell}"
    path.write_text(script, encoding="utf-8")
    binary = shutil.which(shell)
    if binary is None:
        pytest.skip(f"{shell} is not installed on this host")
    flags = {
        "bash": ["-n", str(path)],
        "zsh": ["-n", str(path)],
        "fish": ["-n", str(path)],
    }[shell]
    proc = subprocess.run([binary, *flags], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"{shell} -n failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.parametrize("shell", SHELLS)
def test_completion_does_not_import_heavy_runtimes(qasr, import_blocker, shell):
    blocker, marker = import_blocker
    proc = qasr.run(
        "completion", shell,
        env={"PYTHONPATH": str(blocker)},
    )
    assert_exit(proc, 0, context=f"completion {shell} with import blocker")
    assert not marker.exists(), marker.read_text(encoding="utf-8")
    assert_no_traceback(proc)


@pytest.mark.parametrize("extra", ["--json", "--jsonl"])
def test_completion_rejects_machine_output_flags(qasr, extra):
    proc = qasr.run("completion", "bash", extra)
    assert_exit(proc, 2, context=f"completion + {extra}")
    assert_no_traceback(proc)


def test_completion_rejects_json_before_subcommand(qasr):
    proc = qasr.run("--json", "completion", "bash")
    assert_exit(proc, 2, context="--json before completion")
    assert_no_traceback(proc)


@pytest.mark.parametrize(
    "args",
    [
        ("completion",),
        ("completion", "powershell"),
        ("completion", ""),
        ("completion", "bash", "extra"),
    ],
)
def test_completion_usage_errors(qasr, args):
    proc = qasr.run(*args)
    assert_exit(proc, 2, context=f"completion {args[1:]}")
    assert_no_traceback(proc)


def test_completion_writes_no_files_and_leaks_no_token(qasr, tmp_path):
    secret = "review-secret-token-8p2n0x4z"
    proc = qasr.run(
        "completion", "bash",
        env={
            "QASR_SERVE_TOKEN": secret,
            "QASR_SERVE_URL": "http://user:fake@127.0.0.1:9",
        },
    )
    assert_exit(proc, 0)
    assert secret not in proc.stdout + proc.stderr
    assert "review-password-x1q8m3vz" not in proc.stdout + proc.stderr
    assert list(qasr.work.iterdir()) == []
