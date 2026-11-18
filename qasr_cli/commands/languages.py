"""``qasr languages``."""

from __future__ import annotations

import json
from typing import Any

from ..errors import EXIT_OK
from ..languages import languages_payload
from .common import CommandContext


def handle(ns: Any, ctx: CommandContext) -> int:
    payload = languages_payload()
    if ctx.json_mode:
        ctx.emit(json.dumps(payload, ensure_ascii=False))
        return EXIT_OK

    print(f"forceable ({len(payload['forceable'])}):")
    for lang in payload["forceable"]:
        aliases = ", ".join(lang["aliases"][:4])
        print(f"  {lang['name']:12} {lang['iso']:4} {aliases}")
    print(f"\nalignable ({len(payload['alignable'])}): {', '.join(payload['alignable'])}")
    print(f"\ndialects ({len(payload['dialects'])}, LID output only): {', '.join(payload['dialects'])}")
    return EXIT_OK
