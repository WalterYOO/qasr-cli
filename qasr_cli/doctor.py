"""Environment diagnostics (``qasr doctor``).

This module must not import torch/qwen_asr/vllm. Versions are read from
installed distribution metadata, and device probing is delegated to
:mod:`qasr_cli.runtime`, which owns lazy torch imports.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import SCHEMA_VERSION, __version__
from .catalog import FingerprintCache, resolve_model, verify_model
from .config import Config, qasr_home

EXPECTED_TRANSFORMERS = "4.57.6"
MIN_DISK_BYTES = 5 * 1024 * 1024 * 1024


def _check(
    name: str,
    status: str,
    value: Any = None,
    detail: str = "",
    hint: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": name, "status": status, "value": value, "detail": detail}
    if hint:
        out["hint"] = hint
    return out


def _dist_version(dist_name: str) -> Optional[str]:
    """Read an installed distribution version without importing the module."""
    try:
        from importlib.metadata import version as dist_version

        return dist_version(dist_name)
    except Exception:
        return None


def _check_python() -> Dict[str, Any]:
    info = sys.version_info
    ok = (3, 10) <= (info.major, info.minor) < (3, 13)
    return _check(
        "python",
        "ok" if ok else "fail",
        platform.python_version(),
        f"{sys.executable} ({info.major}.{info.minor})",
        None if ok else "qasr requires Python >=3.10,<3.13 (use --python 3.12)",
    )


def _check_torch() -> Dict[str, Any]:
    version = _dist_version("torch")
    if version is None:
        return _check(
            "torch",
            "fail",
            None,
            "not installed",
            'uv pip install "qasr-cli[local]"  (or uv tool install --python 3.12 "qasr-cli[local]")',
        )
    return _check("torch", "ok", version, "installed")


def _check_qwen_asr() -> Dict[str, Any]:
    version = _dist_version("qwen-asr")
    if version is None:
        return _check(
            "qwen-asr",
            "fail",
            None,
            "not installed",
            'uv pip install "qasr-cli[local]"  (installs qwen-asr==0.0.6)',
        )
    return _check("qwen-asr", "ok", version, "installed")


def _check_transformers() -> Dict[str, Any]:
    version = _dist_version("transformers")
    if version is None:
        return _check("transformers", "warn", None, "not installed",
                      'uv pip install "qasr-cli[local]"')
    if version != EXPECTED_TRANSFORMERS:
        return _check(
            "transformers",
            "warn",
            version,
            f"expected {EXPECTED_TRANSFORMERS} (qwen-asr 0.0.6 pins it)",
            f"uv pip install transformers=={EXPECTED_TRANSFORMERS}",
        )
    return _check("transformers", "ok", version, "matches qwen-asr pin")


def _check_vllm() -> Dict[str, Any]:
    version = _dist_version("vllm")
    if version is None:
        return _check("vllm", "ok", None, "not installed (optional; auto uses transformers)")
    return _check("vllm", "ok", version, "installed (Linux + CUDA only)")


def _check_backend(config: Optional[Config]) -> Dict[str, Any]:
    """Report the configured backend and what ``auto`` resolves to.

    Metadata only: this module must not import torch/vllm.
    """
    selected = str((config.get("backend") if config is not None else None) or "auto")
    server = config.get("server") if config is not None else None
    if server:
        resolved = "remote"
    elif selected != "auto":
        resolved = selected
    elif (
        sys.platform.startswith("linux")
        and _dist_version("torch") is not None
        and _dist_version("vllm") is not None
    ):
        resolved = "vllm"
    else:
        resolved = "transformers"
    hint = None
    status = "ok"
    if resolved == "vllm" and not sys.platform.startswith("linux"):
        status = "warn"
        hint = "vLLM requires Linux + CUDA; use --backend transformers"
    return _check(
        "backend",
        status,
        {"selected": selected, "resolved": resolved},
        f"selected {selected}, resolved {resolved}",
        hint,
    )


def _check_device() -> Dict[str, Any]:
    from .runtime import detect_device

    device, detail = detect_device()
    if device == "unknown":
        return _check("device", "warn", None, detail, "install the [local] extra to run inference")
    return _check("device", "ok", device, detail)


def _total_memory_bytes() -> Optional[int]:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


def _check_memory() -> Dict[str, Any]:
    total = _total_memory_bytes()
    if not total:
        return _check("memory", "warn", None, "could not determine system memory")
    gib = round(total / (1024 ** 3), 1)
    status = "ok" if gib >= 8 else "warn"
    hint = None if status == "ok" else "less than 8 GiB RAM; prefer --model 0.6b and --batch-size 1"
    return _check("memory", status, f"{gib} GiB", "system RAM", hint)


def _check_models(config: Optional[Config] = None) -> List[Dict[str, Any]]:
    cache = FingerprintCache()
    checks: List[Dict[str, Any]] = []
    for alias in ("1.7b", "0.6b", "aligner"):
        resolved = resolve_model(alias, config)
        verify = verify_model(resolved, cache=cache)
        status = verify["status"]
        if status == "complete":
            checks.append(_check(f"model:{alias}", "ok", str(resolved.path), "complete"))
        elif status == "missing":
            checks.append(
                _check(f"model:{alias}", "warn", None, "not downloaded",
                       f"qasr models pull {alias}")
            )
        else:
            checks.append(
                _check(f"model:{alias}", "fail", str(resolved.path),
                       "incomplete download", verify.get("hint"))
            )
    return checks


def _check_disk() -> Dict[str, Any]:
    try:
        home = qasr_home()
        home.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(home)
    except OSError as exc:
        return _check("disk", "warn", None, f"cannot stat QASR_HOME: {exc}")
    free_gib = round(usage.free / (1024 ** 3), 1)
    status = "ok" if usage.free >= MIN_DISK_BYTES else "warn"
    hint = None if status == "ok" else "less than 5 GiB free under QASR_HOME"
    return _check("disk", status, f"{free_gib} GiB free", str(home), hint)


def _check_soundfile() -> Dict[str, Any]:
    try:
        import soundfile as sf
    except Exception as exc:
        return _check("soundfile", "fail", None, f"not importable: {exc}",
                      'uv pip install "qasr-cli"')
    return _check("soundfile", "ok", getattr(sf, "__version__", "unknown"),
                  f"libsndfile {sf.__libsndfile_version__}")


def _check_ffmpeg() -> Dict[str, Any]:
    path = shutil.which("ffmpeg")
    if not path:
        return _check("ffmpeg", "warn", None, "not found on PATH",
                      "install ffmpeg to decode m4a/mp4/webm inputs")
    try:
        out = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=10)
        first = (out.stdout or "").splitlines()[0] if out.stdout else "ffmpeg"
    except (OSError, subprocess.SubprocessError):
        first = "ffmpeg"
    return _check("ffmpeg", "ok", first, path)


def _check_home_writable() -> Dict[str, Any]:
    home = qasr_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".qasr-doctor-", dir=home)
        os.close(fd)
        Path(name).unlink(missing_ok=True)
    except OSError as exc:
        return _check("qasr-home", "fail", str(home), f"not writable: {exc}",
                      "set QASR_HOME to a writable directory")
    return _check("qasr-home", "ok", str(home), "writable")


def _check_server(server: str) -> Dict[str, Any]:
    """Probe a remote OpenAI-compatible server (token stays out of the payload)."""
    from .remote import probe_server

    try:
        info = probe_server(server)
    except Exception as exc:  # invalid URL -> UsageError, never crash doctor
        from .remote import redact_url

        message = getattr(exc, "message", str(exc))
        return _check("server", "fail", {"url": redact_url(str(server))}, message,
                      getattr(exc, "hint", None))

    value = {
        "url": info["url"],
        "latency_ms": info["latency_ms"],
        "models": info["models"],
        "token_set": info["token_set"],
        "reachable": info["reachable"],
        "http_status": info["http_status"],
        "auth_rejected": info["auth_rejected"],
        "timed_out": info["timed_out"],
    }
    if not info["reachable"]:
        hint = ("check the host/port, the SSH tunnel, and the server process")
        return _check("server", "fail", value, str(info.get("error") or "unreachable"), hint)
    status = info.get("http_status") or 0
    if info["auth_rejected"]:
        hint = (
            "QASR_SERVE_TOKEN is set but the server rejected it"
            if info["token_set"]
            else "set QASR_SERVE_TOKEN in the environment"
        )
        return _check("server", "warn", value, f"HTTP {status} (auth rejected)", hint)
    if status >= 500:
        return _check("server", "fail", value, f"HTTP {status}", "server returned an error; check its logs")
    if status >= 400:
        return _check("server", "warn", value, f"HTTP {status}",
                      "the server may not expose /v1/models")
    if info.get("error"):
        return _check("server", "warn", value, f"{status}: {info['error']}")
    count = len(info["models"])
    latency = info["latency_ms"]
    detail = f"{count} model(s), {latency} ms"
    if info["token_set"]:
        detail += ", token set"
    return _check("server", "ok", value, detail)


def run_doctor(config: Optional[Config] = None, *, server: Optional[str] = None) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = [
        _check_python(),
        _check("qasr", "ok", __version__, "qasr-cli"),
        _check("platform", "ok", platform.platform(), platform.machine()),
        _check_torch(),
        _check_qwen_asr(),
        _check_transformers(),
        _check_vllm(),
        _check_backend(config),
        _check_device(),
        _check_memory(),
        _check_home_writable(),
        _check_disk(),
        _check_soundfile(),
        _check_ffmpeg(),
    ]
    checks.extend(_check_models(config))
    remote_url = server or (config.get("server") if config is not None else None)
    if remote_url:
        checks.append(_check_server(str(remote_url)))

    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        status = "fail"
    elif "warn" in statuses:
        status = "warn"
    else:
        status = "ok"
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": status != "fail",
        "command": "doctor",
        "status": status,
        "checks": checks,
    }


def format_doctor_text(payload: Dict[str, Any]) -> str:
    lines = [f"qasr doctor: {payload['status']}"]
    for check in payload["checks"]:
        lines.append(f"  [{check['status']:4}] {check['name']}: {check['detail']}")
        if check.get("hint"):
            lines.append(f"         hint: {check['hint']}")
    return "\n".join(lines)


__all__ = ["format_doctor_text", "run_doctor"]
