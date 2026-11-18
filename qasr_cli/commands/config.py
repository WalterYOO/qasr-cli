"""``qasr config show|path``."""

from __future__ import annotations

import json
from typing import Any

from .. import SCHEMA_VERSION
from ..config import active_config_path, config_search_paths, qasr_home
from ..errors import EXIT_OK
from .common import CommandContext


def handle(ns: Any, ctx: CommandContext) -> int:
    action = getattr(ns, "config_command", None)
    explicit = str(ctx.config.explicit_path) if ctx.config.explicit_path else None

    if action == "path":
        path = active_config_path(explicit)
        if ctx.json_mode:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "ok": True,
                "command": "config",
                "action": "path",
                "path": str(path),
                "exists": path.is_file(),
                "candidates": [
                    {"path": str(p), "exists": p.is_file()}
                    for p in config_search_paths(explicit)
                ],
            }
            ctx.emit(json.dumps(payload, ensure_ascii=False))
        else:
            print(str(path))
        return EXIT_OK

    # config show: never print URL userinfo or token-like query values.
    from ..remote import redact_url

    display_config = {}
    for key, entry in ctx.config.resolved().items():
        entry = dict(entry)
        if key == "server" and isinstance(entry.get("value"), str) and entry["value"]:
            entry["value"] = redact_url(entry["value"])
        display_config[key] = entry
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": "config",
        "action": "show",
        "home": str(qasr_home()),
        "files": ctx.config.files,
        "config": display_config,
    }
    if ctx.json_mode:
        ctx.emit(json.dumps(payload, ensure_ascii=False))
        return EXIT_OK

    print(f"QASR_HOME: {qasr_home()}")
    for entry in ctx.config.files:
        marker = "+" if entry.get("exists") else "-"
        print(f"  [{marker}] {entry['path']}")
    print("")
    for key, entry in payload["config"].items():
        value = entry["value"]
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        print(f"  {key:24} = {value!r:24} ({entry['source']})")
    return EXIT_OK
