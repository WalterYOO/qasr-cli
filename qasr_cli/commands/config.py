"""``qasr config show|path``."""

from __future__ import annotations

import json
from typing import Any

from .. import SCHEMA_VERSION
from ..config import (
    CONFIG_KEYS,
    active_config_path,
    coerce_value,
    config_search_paths,
    home_config_path,
    qasr_home,
    write_json_atomic,
)
from ..errors import EXIT_OK, UsageError
from .common import CommandContext

TOKEN_KEY_MARKERS = ("token", "secret", "password", "passwd", "api_key", "apikey")


def _project_root() -> Any:
    from pathlib import Path as _Path

    current = _Path.cwd().resolve()
    for parent in (current, *current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").is_file():
            return parent
    return current


def _get(ns: Any, ctx: CommandContext) -> int:
    key = getattr(ns, "config_key", None)
    if not key or key not in CONFIG_KEYS:
        raise UsageError(
            f"unknown configuration key: {key}",
            hint="known keys: " + ", ".join(CONFIG_KEYS),
        )
    entry = ctx.config.resolved()[key]
    value = entry.get("value")
    if ctx.json_mode:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "command": "config",
            "action": "get",
            "key": key,
            "value": value,
            "source": entry.get("source"),
        }
        ctx.emit(json.dumps(payload, ensure_ascii=False))
    else:
        print(value)
    return EXIT_OK


def _set(ns: Any, ctx: CommandContext) -> int:

    key = getattr(ns, "config_key", None)
    raw = getattr(ns, "config_value", None)
    if not key or key not in CONFIG_KEYS:
        raise UsageError(
            f"unknown configuration key: {key}",
            hint="known keys: " + ", ".join(CONFIG_KEYS),
        )
    lowered = str(key).casefold()
    if any(marker in lowered for marker in TOKEN_KEY_MARKERS):
        raise UsageError("server tokens are not configuration keys",
                         hint="set QASR_SERVE_TOKEN in the environment")
    value = coerce_value(key, raw, "cli:config set")

    if getattr(ns, "project", False) and getattr(ns, "home", False):
        raise UsageError("--project and --home are mutually exclusive")
    if getattr(ns, "project", False):
        target = _project_root() / ".qasr" / "config.json"
    else:
        target = home_config_path()
    if target.is_symlink():
        raise UsageError(f"refusing to write through a symlink: {target}")

    existing: dict = {}
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise UsageError(f"cannot read config file: {target}") from exc
        if not isinstance(existing, dict):
            raise UsageError(f"config file is not a JSON object: {target}")
    existing[key] = value
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, existing)

    if ctx.json_mode:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "command": "config",
            "action": "set",
            "key": key,
            "value": value,
            "path": str(target),
        }
        ctx.emit(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"{key} = {value!r}  ({target})")
    return EXIT_OK


def handle(ns: Any, ctx: CommandContext) -> int:
    action = getattr(ns, "config_command", None)
    if action == "get":
        return _get(ns, ctx)
    if action == "set":
        return _set(ns, ctx)
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
