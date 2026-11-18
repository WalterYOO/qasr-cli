"""Shared helpers for command adapters."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

from ..config import Config
from ..errors import RuntimeFailure


@dataclass
class CommandContext:
    config: Config
    json_mode: bool = False
    jsonl_mode: bool = False
    quiet: bool = False
    verbose: bool = False
    dry_run: bool = False

    def log(self, message: str) -> None:
        if not self.quiet:
            print(message, file=sys.stderr)

    def warn(self, message: str) -> None:
        print(f"warning: {message}", file=sys.stderr)

    def emit(self, text: str) -> None:
        if text:
            print(text)


def resolve_device_dtype(config: Config, *, optional: bool = False) -> tuple[str, str]:
    """Resolve device/dtype without loading any weights.

    With ``optional=True`` (dry-run), fall back to the requested strings when
    torch is unavailable instead of failing.
    """
    requested_device = str(config.get("device") or "auto")
    requested_dtype = str(config.get("dtype") or "auto")
    if requested_device != "auto" and requested_dtype != "auto":
        return requested_device, requested_dtype
    try:
        from ..runtime import resolve_device, resolve_dtype

        device = resolve_device(requested_device)
        dtype = resolve_dtype(requested_dtype, device)
        return device, dtype
    except RuntimeFailure:
        if optional:
            dtype = requested_dtype
            if dtype == "auto":
                dtype = "fp32"
            return requested_device, dtype
        raise


def parse_auto_int(value: Optional[str], *, name: str = "--max-new-tokens") -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() == "auto":
        return None
    try:
        number = int(text)
    except ValueError as exc:
        from ..errors import UsageError

        raise UsageError(f"{name} must be 'auto' or a positive integer") from exc
    if number <= 0:
        from ..errors import UsageError

        raise UsageError(f"{name} must be a positive integer")
    return number
