"""``qasr finetune lint-jsonl``."""

from __future__ import annotations

import json
import sys
from typing import Any

from ..errors import EXIT_INPUT, EXIT_OK, UsageError
from ..finetune import format_human, lint_jsonl
from .common import CommandContext


def handle(ns: Any, ctx: CommandContext) -> int:
    action = getattr(ns, "finetune_command", None)
    if action != "lint-jsonl":
        raise UsageError(
            "finetune requires a subcommand",
            hint="qasr finetune lint-jsonl FILE",
        )
    source = getattr(ns, "jsonl_path", None)
    if not source:
        raise UsageError("lint-jsonl requires a JSONL file path")
    result = lint_jsonl(
        source,
        check_files=not bool(getattr(ns, "no_check_files", False)),
        strict=bool(getattr(ns, "strict", False)),
    )
    if ctx.json_mode:
        ctx.emit(json.dumps(result.payload(), ensure_ascii=False))
    else:
        format_human(result, sys.stdout)
    return EXIT_OK if result.ok else EXIT_INPUT


__all__ = ["handle"]
