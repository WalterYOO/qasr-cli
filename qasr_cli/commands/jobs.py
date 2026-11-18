"""``qasr jobs list|show``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import SCHEMA_VERSION
from ..errors import EXIT_OK, UsageError
from ..jobs import (
    STATE_DIRNAME,
    STATE_FILENAME,
    find_job,
    list_jobs,
    read_state,
)
from .common import CommandContext


def _limit(ns: Any) -> int | None:
    value = getattr(ns, "limit", None)
    if value is None:
        return None
    limit = int(value)
    if not 1 <= limit <= 10000:
        raise UsageError("--limit must be between 1 and 10000")
    return limit


def _list(ns: Any, ctx: CommandContext) -> int:
    records = list_jobs(_limit(ns))
    if ctx.json_mode:
        ctx.emit(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "ok": True,
                    "command": "jobs",
                    "action": "list",
                    "jobs": records,
                },
                ensure_ascii=False,
            )
        )
        return EXIT_OK
    for record in records:
        print(
            f"{record['job_id']}\t{record['updated_at']}\t{record.get('state')}\t"
            f"{record.get('command', '')}\t{record.get('out_dir', '')}"
        )
    return EXIT_OK


def _show(ns: Any, ctx: CommandContext) -> int:
    target = getattr(ns, "job_target", None)
    record = find_job(str(target or ""))
    out_dir = Path(str(record.get("out_dir") or "")).expanduser()
    state_path = out_dir / STATE_DIRNAME / STATE_FILENAME

    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0, "skipped": 0}
    first_error = None
    latest = read_state(state_path) if state_path.is_file() else {}
    for item_id, entry in latest.items():
        status = str(entry.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
        if status == "failed" and first_error is None:
            first_error = {"id": item_id, "error": entry.get("error")}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": "jobs",
        "action": "show",
        "job": record,
        "state": "present" if state_path.is_file() else "missing",
        "counts": counts,
        "first_error": first_error,
        "out_dir": str(out_dir),
        "state_path": str(state_path),
    }
    if ctx.json_mode:
        ctx.emit(json.dumps(payload, ensure_ascii=False))
        return EXIT_OK
    print(f"job: {record.get('job_id')}")
    print(f"out-dir: {out_dir}")
    print(
        "items: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    if first_error is not None:
        error = first_error.get("error") or {}
        print(f"first error: {first_error['id']}: {error.get('message', '')}")
    return EXIT_OK


def handle(ns: Any, ctx: CommandContext) -> int:
    action = getattr(ns, "jobs_command", None)
    if action == "show":
        return _show(ns, ctx)
    return _list(ns, ctx)
