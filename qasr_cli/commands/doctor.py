"""``qasr doctor``."""

from __future__ import annotations

import json
from typing import Any

from ..doctor import format_doctor_text, run_doctor
from ..errors import EXIT_MODEL, EXIT_OK
from .common import CommandContext


def handle(ns: Any, ctx: CommandContext) -> int:
    payload = run_doctor(ctx.config)
    if ctx.json_mode:
        ctx.emit(json.dumps(payload, ensure_ascii=False))
    else:
        print(format_doctor_text(payload))
    if getattr(ns, "strict", False) and payload["status"] != "ok":
        return EXIT_MODEL
    return EXIT_OK
