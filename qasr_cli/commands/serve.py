"""``qasr serve`` adapter."""

from __future__ import annotations

import json
import shlex
from typing import Any

from .. import serve as serve_core
from ..errors import EXIT_OK
from .common import CommandContext


def handle(ns: Any, ctx: CommandContext) -> int:
    plan = serve_core.build_plan(
        model=getattr(ns, "model_target", None) or getattr(ns, "model", None),
        host=getattr(ns, "host", None),
        port=getattr(ns, "port", None),
        gpu_memory_utilization=getattr(ns, "gpu_memory_utilization", None),
        passthrough=getattr(ns, "passthrough", None) or [],
        config=ctx.config,
    )
    if ctx.dry_run:
        if ctx.json_mode:
            ctx.emit(json.dumps(plan.to_dict(), ensure_ascii=False))
        else:
            print(shlex.join(plan.argv))
        return EXIT_OK

    ctx.log("exec " + shlex.join(plan.argv))
    serve_core.launch(plan)
    return EXIT_OK  # pragma: no cover - execvp replaces the process
