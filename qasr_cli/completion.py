"""Shell completion scripts generated from the argparse surface.

The parser is the single source of truth: every top-level command and every
long option shown by ``qasr <command> --help`` is discovered at runtime.
This module never imports torch/vllm/qwen_asr/sounddevice.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

SHELLS = ("bash", "zsh", "fish")


@dataclass
class CommandSpec:
    name: str
    options: List[str] = field(default_factory=list)
    value_options: Dict[str, bool] = field(default_factory=dict)


def _option_specs(parser: argparse.ArgumentParser) -> tuple[list[str], dict[str, bool]]:
    names: List[str] = []
    takes_value: Dict[str, bool] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        for option in action.option_strings:
            if option not in names:
                names.append(option)
            takes_value[option] = action.nargs != 0
    return names, takes_value


def collect_spec() -> tuple[CommandSpec, List[CommandSpec]]:
    """Return ``(root, commands)`` derived from the live CLI parser."""
    from .cli import build_parser

    parser = build_parser()
    root_names, root_values = _option_specs(parser)
    root = CommandSpec("qasr", root_names, root_values)
    commands: List[CommandSpec] = []
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, subparser in action.choices.items():
            names, values = _option_specs(subparser)
            commands.append(CommandSpec(name, names, values))
    return root, commands


def _unique(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def render_bash(root: CommandSpec, commands: Sequence[CommandSpec]) -> str:
    command_names = " ".join(cmd.name for cmd in commands)
    root_opts = " ".join(root.options)
    lines = [
        "_qasr_completion() {",
        '    local cur cmd opts',
        '    cur="${COMP_WORDS[COMP_CWORD]}"',
        f'    local commands="{command_names}"',
        f'    local root_opts="{root_opts}"',
        '    if [ "$COMP_CWORD" -eq 1 ]; then',
        '        COMPREPLY=( $(compgen -W "$commands $root_opts" -- "$cur") )',
        "        return 0",
        "    fi",
        '    cmd="${COMP_WORDS[1]}"',
        '    opts="$root_opts"',
        '    case "$cmd" in',
    ]
    for cmd in commands:
        lines.append(f'        {cmd.name}) opts="$opts {" ".join(cmd.options)}" ;;')
    lines.extend(
        [
            "    esac",
            '    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )',
            "}",
            "complete -F _qasr_completion qasr",
        ]
    )
    return "\n".join(lines) + "\n"


def render_zsh(root: CommandSpec, commands: Sequence[CommandSpec]) -> str:
    lines = [
        "#compdef qasr",
        "",
        "_qasr() {",
        "    local -a commands root_opts opts",
        "    commands=(",
    ]
    for cmd in commands:
        lines.append(f"        '{cmd.name}'")
    lines.extend(
        [
            "    )",
            f"    root_opts=({' '.join(root.options)})",
            "    if (( CURRENT == 2 )); then",
            "        _describe 'command' commands",
            "        _describe 'option' root_opts",
            "        return",
            "    fi",
            "    case $words[2] in",
        ]
    )
    for cmd in commands:
        lines.append(
            f"        {cmd.name}) opts=( {' '.join(cmd.options)} ) ;;"
        )
    lines.extend(
        [
            "        *) opts=() ;;",
            "    esac",
            "    _describe 'option' opts",
            "}",
            "",
            '_qasr "$@"',
        ]
    )
    return "\n".join(lines) + "\n"


def render_fish(root: CommandSpec, commands: Sequence[CommandSpec]) -> str:
    lines = ["complete -c qasr -f"]
    # Root options are accepted both before and after the subcommand.
    for option in root.options:
        takes_value = root.value_options.get(option, False)
        if option.startswith("--"):
            suffix = " -r" if takes_value else ""
            lines.append(f"complete -c qasr -l {option[2:]}{suffix}  # {option}")
        elif option.startswith("-") and len(option) == 2:
            suffix = " -r" if takes_value else ""
            lines.append(f"complete -c qasr -s {option[1:]}{suffix}  # {option}")
    for cmd in commands:
        lines.append(
            "complete -c qasr -n '__fish_use_subcommand' "
            f"-a {cmd.name}  # {cmd.name}"
        )
    for cmd in commands:
        for option in cmd.options:
            if option.startswith("--"):
                suffix = " -r" if cmd.value_options.get(option, False) else ""
                lines.append(
                    "complete -c qasr -n '__fish_seen_subcommand_from "
                    f"{cmd.name}' -l {option[2:]}{suffix}  # {option}"
                )
            elif option.startswith("-") and len(option) == 2:
                suffix = " -r" if cmd.value_options.get(option, False) else ""
                lines.append(
                    "complete -c qasr -n '__fish_seen_subcommand_from "
                    f"{cmd.name}' -s {option[1:]}{suffix}  # {option}"
                )
    return "\n".join(lines) + "\n"


def render(shell: str) -> str:
    if shell not in SHELLS:
        raise ValueError(f"unsupported shell: {shell}")
    root, commands = collect_spec()
    if shell == "bash":
        return render_bash(root, commands)
    if shell == "zsh":
        return render_zsh(root, commands)
    return render_fish(root, commands)


__all__ = ["SHELLS", "collect_spec", "render", "render_bash", "render_fish", "render_zsh"]
