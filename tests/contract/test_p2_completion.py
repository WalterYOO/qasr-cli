"""P2: qasr completion scripts."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest

from qasr_cli.cli import build_parser, main

LONG_OPTION_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")


def _commands():
    parser = build_parser()
    for action in parser._actions:
        if action.__class__.__name__ == "_SubParsersAction":
            return list(action.choices)
    return []


def _help_options(capsys, argv):
    main(argv)
    out = capsys.readouterr().out
    return set(LONG_OPTION_RE.findall(out))


def _script(capsys, shell):
    code = main(["completion", shell])
    out = capsys.readouterr().out
    assert code == 0
    return out


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_script_covers_all_commands_and_help_options(capsys, shell):
    commands = _commands()
    script = _script(capsys, shell)
    for command in commands:
        assert command in script
        options = _help_options(capsys, [command, "--help"])
        for option in options:
            assert option in script, (shell, command, option)
    for option in _help_options(capsys, ["--help"]):
        assert option in script, (shell, "root", option)


def test_output_has_exactly_one_trailing_newline(capsys):
    for shell in ("bash", "zsh", "fish"):
        script = _script(capsys, shell)
        assert script.endswith("\n")
        assert not script.endswith("\n\n")


def test_bash_syntax(capsys, tmp_path):
    script = _script(capsys, "bash")
    path = tmp_path / "qasr.bash"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("shell", ["zsh", "fish"])
def test_optional_shell_syntax(capsys, tmp_path, shell):
    executable = shutil.which(shell)
    if executable is None:
        pytest.skip(f"{shell} is not installed")
    script = _script(capsys, shell)
    path = tmp_path / f"qasr.{shell}"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [executable, "-n", str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_json_is_rejected(capsys):
    code = main(["--json", "completion", "bash"])
    assert code == 2
    assert "completion" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["completion"],
        ["completion", "powershell"],
        ["completion", "bash", "extra"],
    ],
)
def test_bad_arguments(capsys, argv):
    code = main(argv)
    capsys.readouterr()
    assert code == 2


def test_completion_does_not_import_model_runtimes(capsys, monkeypatch):
    blocker = pytest.importorskip("importlib.abc")

    class _Block(blocker.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in ("torch", "vllm", "qwen_asr", "sounddevice"):
                raise AssertionError(f"completion imported {fullname}")
            return None

    for name in list(sys.modules):
        if name.split(".")[0] in ("torch", "vllm", "qwen_asr", "sounddevice"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Block(), *sys.meta_path])
    assert main(["completion", "bash"]) == 0
    capsys.readouterr()
