"""Lazy access to torch/qwen_asr and device policy.

Nothing in this module imports torch at import time. All third-party imports
happen inside functions, and ``--offline`` environment variables are set
*before* transformers/qwen_asr are imported.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import platform
import sys
from typing import Any, Dict, Iterator, Optional

from .errors import ModelError, RuntimeFailure

INSTALL_HINT = 'uv pip install "qasr-cli[local]"  (or uv tool install --python 3.12 "qasr-cli[local]")'


@contextlib.contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Redirect fd 1 to fd 2 so third-party banners never pollute results."""
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        saved_fd = os.dup(1)
    except OSError:
        yield
        return
    try:
        os.dup2(2, 1)
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def configure_offline(offline: bool, *, cache_dir: Optional[str] = None) -> None:
    """Set HF/transformers offline variables. Must run before the first import."""
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    if cache_dir:
        os.environ.setdefault("HF_HOME", cache_dir)


def import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeFailure(f"torch is not importable: {exc}", hint=INSTALL_HINT) from exc


def import_qwen_asr() -> Any:
    try:
        return importlib.import_module("qwen_asr")
    except Exception as exc:
        raise RuntimeFailure(f"qwen_asr is not importable: {exc}", hint=INSTALL_HINT) from exc


def package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in ("torch", "transformers", "qwen_asr", "vllm"):
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def detect_device() -> tuple[str, str]:
    """Best-effort ``(device, detail)`` for diagnostics; never raises.

    Used by ``qasr doctor`` so that module never imports torch itself.
    """
    try:
        torch_mod = importlib.import_module("torch")
    except Exception:
        return "unknown", "torch unavailable"
    try:
        if torch_mod.cuda.is_available():
            return f"cuda:{torch_mod.cuda.current_device()}", torch_mod.cuda.get_device_name(0)
        mps = getattr(torch_mod.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", f"Apple MPS ({platform.machine()})"
        return "cpu", f"CPU only ({platform.machine()})"
    except Exception as exc:  # pragma: no cover - defensive
        return "unknown", f"device probe failed: {exc}"


def resolve_device(requested: str = "auto", torch_mod: Any = None) -> str:
    """Map ``auto|cpu|mps|cuda|cuda:N`` to a concrete torch device string."""
    torch_mod = torch_mod or import_torch()
    requested = (requested or "auto").strip()
    if requested == "auto":
        if torch_mod.cuda.is_available():
            return "cuda"
        mps = getattr(torch_mod.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch_mod.cuda.is_available():
            raise RuntimeFailure(
                "--device cuda requested but CUDA is not available",
                hint="use --device auto (PyTorch build has no CUDA support here)",
            )
        return requested
    if requested == "mps":
        mps = getattr(torch_mod.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeFailure(
                "--device mps requested but MPS is not available",
                hint="use --device cpu (or run on Apple Silicon with a recent torch)",
            )
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise RuntimeFailure(f"unsupported --device value: {requested}", hint="auto|cpu|mps|cuda|cuda:N")


def resolve_dtype(requested: str = "auto", device: str = "auto", torch_mod: Any = None) -> str:
    """Return a canonical dtype name (fp32|fp16|bf16)."""
    requested = (requested or "auto").strip()
    if requested == "auto":
        if device.startswith("cuda"):
            return "bf16"
        if device == "mps":
            return "fp16"
        return "fp32"
    aliases = {
        "fp32": "fp32", "float32": "fp32",
        "fp16": "fp16", "float16": "fp16", "half": "fp16",
        "bf16": "bf16", "bfloat16": "bf16",
    }
    if requested not in aliases:
        raise RuntimeFailure(f"unsupported --dtype value: {requested}", hint="auto|fp32|fp16|bf16")
    return aliases[requested]


def torch_dtype(name: str, torch_mod: Any = None) -> Any:
    torch_mod = torch_mod or import_torch()
    mapping = {
        "fp32": torch_mod.float32,
        "fp16": torch_mod.float16,
        "bf16": torch_mod.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ModelError(f"unknown dtype: {name}") from exc
