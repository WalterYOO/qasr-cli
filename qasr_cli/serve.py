"""``qasr serve``: thin argv wrapper around ``qwen-asr-serve``.

The module is import-safe without vLLM: it only resolves the model and builds
the child argv. Starting the server replaces the current process via
``os.execvp`` so qasr never owns the child or its logs.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .catalog import ResolvedModel, resolve_model, verify_model
from .config import Config
from .errors import ModelError

SERVE_BINARY = "qwen-asr-serve"
INSTALL_HINT = (
    'uv tool install --python 3.12 "qasr-cli[vllm]" '
    'or pip install "qwen-asr[vllm]==0.0.6"'
)


@dataclass
class ServePlan:
    binary: str
    argv: List[str]
    requested: str
    resolved: str
    source: str  # local | repo
    path: Optional[str] = None
    repo: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "ok": True,
            "command": "serve",
            "argv": list(self.argv),
            "binary": self.binary,
            "model": self.requested,
            "resolved": self.resolved,
            "source": self.source,
            "path": self.path,
            "repo": self.repo,
        }


def _local_complete(resolved: ResolvedModel) -> bool:
    if not resolved.present or resolved.path is None:
        return False
    return verify_model(resolved)["status"] == "complete"


def resolve_serve_model(value: Optional[str], config: Config) -> ResolvedModel:
    """Prefer a complete local directory; otherwise fall back to the repo id."""
    requested = str(value or config.get("model") or "1.7b")
    resolved = resolve_model(requested, config)
    if resolved.present and not _local_complete(resolved):
        # An incomplete local copy should not silently serve broken weights;
        # point the server at the repo id so it can download cleanly.
        if resolved.repo:
            return ResolvedModel(
                value=resolved.value,
                alias=resolved.alias,
                repo=resolved.repo,
                path=None,
                source=resolved.source,
                kind=resolved.kind,
            )
    return resolved


def build_plan(
    *,
    model: Optional[str],
    host: Optional[str],
    port: Optional[int],
    gpu_memory_utilization: Optional[float],
    passthrough: Sequence[str],
    config: Config,
    binary: Optional[str] = None,
) -> ServePlan:
    resolved = resolve_serve_model(model, config)
    if resolved.path is not None:
        model_arg = str(resolved.path)
        source = "local"
    elif resolved.repo:
        model_arg = resolved.repo
        source = "repo"
    else:
        model_arg = resolved.value
        source = "local"

    argv: List[str] = [binary or SERVE_BINARY, model_arg]
    if host:
        argv.extend(["--host", str(host)])
    if port is not None:
        argv.extend(["--port", str(port)])
    if gpu_memory_utilization is not None:
        argv.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    argv.extend(str(item) for item in passthrough)

    return ServePlan(
        binary=argv[0],
        argv=argv,
        requested=str(model or config.get("model") or "1.7b"),
        resolved=model_arg,
        source=source,
        path=str(resolved.path) if resolved.path else None,
        repo=resolved.repo,
    )


def launch(plan: ServePlan) -> None:
    """Replace this process with the server; never returns on success."""
    executable = shutil.which(plan.binary)
    if executable is None:
        raise ModelError(
            f"{SERVE_BINARY} is not installed",
            hint=INSTALL_HINT,
        )
    argv = [executable, *plan.argv[1:]]
    try:
        os.execvp(executable, argv)
    except OSError as exc:
        raise ModelError(
            f"failed to start {SERVE_BINARY}: {exc}",
            hint=INSTALL_HINT,
        ) from exc
