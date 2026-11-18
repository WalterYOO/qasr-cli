"""``qasr completion bash|zsh|fish``."""

from __future__ import annotations

import sys
from typing import Any

from ..completion import SHELLS, render
from ..errors import EXIT_OK, UsageError
from .common import CommandContext


def handle(ns: Any, ctx: CommandContext) -> int:
    if ctx.json_mode:
        raise UsageError(
            "--json is not supported by completion",
            hint="completion writes a shell script to stdout",
        )
    shell = getattr(ns, "shell", None)
    if shell not in SHELLS:
        raise UsageError(
            "completion requires a shell: " + ", ".join(SHELLS),
            hint="qasr completion bash|zsh|fish",
        )
    script = render(shell)
    # Contract: exactly one trailing newline and nothing else on stdout.
    sys.stdout.write(script.rstrip("\n") + "\n")
    sys.stdout.flush()
    return EXIT_OK


__all__ = ["handle"]
