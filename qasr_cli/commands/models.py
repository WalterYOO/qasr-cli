"""``qasr models list|path|verify|pull``."""

from __future__ import annotations

import json
from typing import Any

from .. import catalog
from ..errors import EXIT_MODEL, EXIT_OK, ModelError
from .common import CommandContext


def _emit(payload: dict, ctx: CommandContext) -> None:
    ctx.emit(json.dumps(payload, ensure_ascii=False))


def handle(ns: Any, ctx: CommandContext) -> int:
    action = getattr(ns, "models_command", None)

    if action == "list":
        payload = catalog.models_list(ctx.config)
        if ctx.json_mode:
            _emit(payload, ctx)
            return EXIT_OK
        for entry in payload["models"]:
            state = entry["status"]
            size = entry["size_bytes"]
            size_text = f"{size / (1024 ** 3):.2f} GiB" if size else "-"
            print(f"{entry['alias']:8} {state:10} {size_text:>10}  {entry['path'] or entry['repo']}")
        return EXIT_OK

    target = getattr(ns, "model_target", None)
    if not target:
        raise ModelError("a model alias is required", hint="1.7b | 0.6b | aligner")

    if action == "path":
        payload = catalog.models_path(target, ctx.config)
        if ctx.json_mode:
            _emit(payload, ctx)
        else:
            print(payload["path"])
        return EXIT_OK

    if action == "verify":
        resolved = catalog.resolve_model(target, ctx.config)
        payload = catalog.verify_model(resolved, full=bool(getattr(ns, "full", False)))
        if ctx.json_mode:
            _emit(payload, ctx)
        else:
            print(f"{payload['alias'] or target}: {payload['status']}")
            for check in payload["checks"]:
                if check["status"] != "ok":
                    print(f"  [{check['status']}] {check['name']}: {check['detail']}")
            if payload.get("hint"):
                print(f"hint: {payload['hint']}")
        return EXIT_OK if payload["status"] == "complete" else EXIT_MODEL

    if action == "pull":
        payload = catalog.pull_model(
            target,
            source=str(getattr(ns, "source", None) or "hf"),
            revision=getattr(ns, "revision", None),
            offline=bool(ctx.config.get("offline")),
        )
        if ctx.json_mode:
            _emit(payload, ctx)
        else:
            print(f"{payload['alias']}: {payload['status']} -> {payload['path']}")
        return EXIT_OK if payload["status"] == "complete" else EXIT_MODEL

    raise ModelError(f"unknown models action: {action}")
