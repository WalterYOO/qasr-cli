"""Error types and exit codes shared by every qasr command."""

from __future__ import annotations

from typing import Any, Dict, Optional

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MODEL = 3
EXIT_INPUT = 4
EXIT_QUALITY = 5
EXIT_PARTIAL = 6
EXIT_INTERRUPT = 130


class QasrError(Exception):
    """Base error carrying a stable ``kind`` and process exit code."""

    kind = "internal"
    exit_code = EXIT_MODEL
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details or {}
        if code is not None:
            self.exit_code = code

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": self.kind,
            "code": self.exit_code,
            "message": self.message,
            "retryable": bool(self.retryable),
            "details": self.details,
        }
        if self.hint:
            out["hint"] = self.hint
        return out


class UsageError(QasrError):
    kind = "usage"
    exit_code = EXIT_USAGE


class ConfigError(UsageError):
    kind = "config"


class InputError(QasrError):
    kind = "input"
    exit_code = EXIT_INPUT


class CapabilityError(QasrError):
    kind = "capability"
    exit_code = EXIT_INPUT


class ModelError(QasrError):
    kind = "model"
    exit_code = EXIT_MODEL


class RuntimeFailure(QasrError):
    kind = "runtime"
    exit_code = EXIT_MODEL


class QualityError(QasrError):
    kind = "quality"
    exit_code = EXIT_QUALITY


class PartialFailure(QasrError):
    kind = "partial"
    exit_code = EXIT_PARTIAL
