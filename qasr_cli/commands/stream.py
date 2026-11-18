"""``qasr stream``: vLLM-only streaming with cumulative snapshots (P2)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from .. import stream as core_stream
from ..audio import probe
from ..backends import build_vllm_backend
from ..catalog import require_local, resolve_model
from ..errors import (
    CapabilityError,
    InputError,
    QasrError,
    RuntimeFailure,
    UsageError,
)
from ..languages import resolve_forceable
from ..runtime import require_vllm, set_cuda_visible_device
from .common import (
    CommandContext,
    resolve_backend,
    resolve_device_dtype,
    selected_backend,
)

MIC_HINT = 'uv tool install "qasr-cli[vllm,mic]"'
_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _read_context_file(path: str) -> str:
    target = Path(path).expanduser()
    if not target.is_file():
        raise InputError(f"context file not found: {path}")
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError(f"cannot read context file: {path}") from exc
    parts = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return " ".join(parts)


def _context(ns: Any, cfg: Any) -> str:
    parts: list[str] = []
    configured = cfg.get("context") or []
    if isinstance(configured, str):
        configured = [configured]
    parts.extend(str(part) for part in configured)
    parts.extend(getattr(ns, "context", None) or [])
    merged: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = part.strip()
        if text and text not in seen:
            seen.add(text)
            merged.append(text)
    return " ".join(merged)


def _language(ns: Any, cfg: Any) -> Optional[str]:
    raw = getattr(ns, "language", None)
    if raw is None and cfg.source_of("language") != "default":
        raw = cfg.get("language")
    if raw is None or not str(raw).strip() or str(raw).strip().casefold() == "auto":
        return None
    return resolve_forceable(str(raw))


def _resolve_backend_for_stream(cfg: Any) -> None:
    requested = selected_backend(cfg)
    if requested in ("remote", "transformers"):
        raise CapabilityError(
            "streaming is only available with the local vLLM backend",
            hint=MIC_HINT + "; remote servers do not expose chunk rollback",
        )
    if requested == "vllm":
        try:
            require_vllm()
        except QasrError as exc:
            details = {"original_hint": exc.hint} if exc.hint else None
            raise CapabilityError(exc.message, hint=MIC_HINT, details=details) from exc
        return
    if requested != "auto":
        raise CapabilityError(
            f"backend '{requested}' cannot stream",
            hint=MIC_HINT,
        )
    resolved, _fallback = resolve_backend(cfg)
    if resolved != "vllm":
        raise CapabilityError(
            "streaming requires vLLM, which is not available on this host",
            hint=MIC_HINT,
        )


def _validate_numbers(ns: Any) -> tuple[int, int, int, int, int, float]:
    sample_rate = int(getattr(ns, "sample_rate", 16000))
    if sample_rate != core_stream.SAMPLE_RATE:
        raise UsageError("--sample-rate must be 16000 for streaming")
    channels = int(getattr(ns, "channels", 1))
    if channels not in (1, 2):
        raise UsageError("--channels must be 1 or 2")
    chunk_ms = int(getattr(ns, "chunk_ms", 2000))
    if not 100 <= chunk_ms <= 10000 or chunk_ms % 10 != 0:
        raise UsageError(
            "--chunk-ms must be between 100 and 10000 and a multiple of 10"
        )
    unfixed_chunks = int(getattr(ns, "unfixed_chunks", 2))
    if unfixed_chunks < 0:
        raise UsageError("--unfixed-chunks must be >= 0")
    unfixed_tokens = int(getattr(ns, "unfixed_tokens", 5))
    if unfixed_tokens < 0:
        raise UsageError("--unfixed-tokens must be >= 0")
    max_seconds = float(getattr(ns, "max_stream_seconds", 3600.0))
    if max_seconds <= 0:
        raise UsageError("--max-stream-seconds must be > 0")
    chunk_samples = int(round(chunk_ms * core_stream.SAMPLE_RATE / 1000))
    return (
        chunk_samples, channels, chunk_ms, unfixed_chunks, unfixed_tokens, max_seconds,
    )


def handle(ns: Any, ctx: CommandContext) -> int:
    cfg = ctx.config
    input_value = str(getattr(ns, "input", "") or "").strip()
    if not input_value:
        raise UsageError("--input is required", hint="mic, stdin, or a local audio path")
    if input_value == "-":
        raise UsageError(
            "'-' is not an audio source",
            hint="use --input stdin for a PCM pipe",
        )
    if _URL_RE.match(input_value):
        raise InputError(
            "stream does not accept URLs",
            hint="download the file locally or use qasr transcribe for URLs",
        )
    if getattr(ns, "input", None) == "mic" and getattr(ns, "realtime", False):
        raise UsageError("--realtime only applies to file input")
    if getattr(ns, "input", None) == "stdin" and getattr(ns, "realtime", False):
        raise UsageError("--realtime only applies to file input")
    if input_value != "stdin":
        for flag in ("pcm", "sample_rate", "channels"):
            if hasattr(ns, flag):
                raise UsageError(
                    f"--{flag.replace('_', '-')} is only valid with --input stdin"
                )
    if input_value == "stdin" and not getattr(ns, "pcm", None):
        raise UsageError(
            "--input stdin requires --pcm s16le",
            hint="stream raw 16 kHz little-endian PCM frames",
        )
    if ctx.json_mode:
        raise UsageError(
            "--json cannot be combined with stream",
            hint="machine output is controlled by --emit jsonl; errors use --json envelopes only",
        )
    if ctx.dry_run:
        raise UsageError(
            "--dry-run is not supported by stream",
            hint="stream runs continuously; use transcribe --dry-run to inspect a model plan",
        )

    (
        chunk_samples, channels, chunk_ms, unfixed_chunks, unfixed_tokens, max_seconds,
    ) = _validate_numbers(ns)

    language = _language(ns, cfg)
    context = _context(ns, cfg)
    if getattr(ns, "context_file", None):
        context = " ".join(
            part for part in (context, _read_context_file(ns.context_file)) if part
        )

    settings = core_stream.StreamSettings(
        chunk_samples=chunk_samples,
        chunk_ms=chunk_ms,
        channels=channels,
        input_kind="stdin" if input_value == "stdin" else ("mic" if input_value == "mic" else "file"),
        emit=getattr(ns, "emit", "text"),
        realtime=bool(getattr(ns, "realtime", False)),
        max_samples=int(round(max_seconds * core_stream.SAMPLE_RATE)),
        context=context,
        language=language,
        unfixed_chunks=unfixed_chunks,
        unfixed_tokens=unfixed_tokens,
    )

    if selected_backend(cfg) in ("auto", "vllm"):
        set_cuda_visible_device(str(cfg.get("device") or "auto"))
    _resolve_backend_for_stream(cfg)

    gpu_util = getattr(ns, "gpu_memory_utilization", None)
    if gpu_util is not None:
        gpu_util = float(gpu_util)
        if not 0.0 < gpu_util <= 1.0:
            raise UsageError("--gpu-memory-utilization must be in (0, 1]")

    model = resolve_model(cfg.get("model"), cfg)
    require_local(model, action="stream")
    model_path = model.path or Path(str(model.value))

    try:
        device, dtype = resolve_device_dtype(cfg)
    except QasrError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeFailure(f"cannot resolve device/dtype: {exc}") from exc

    source: Iterable[Any]
    if input_value == "stdin":
        buffer = getattr(sys.stdin, "buffer", sys.stdin)
        source = core_stream.iter_pcm_s16le(buffer, chunk_samples, channels)
    elif input_value == "mic":
        source = core_stream.iter_mic_chunks()
    else:
        target = Path(input_value).expanduser()
        if not target.is_file():
            raise InputError(f"audio file not found: {target}")
        info = probe(target)
        source = core_stream.iter_file_chunks(target, info, chunk_samples)

    backend = build_vllm_backend(
        model_path,
        device=device,
        dtype=dtype,
        gpu_memory_utilization=gpu_util,
        offline=bool(cfg.get("offline")),
    )
    try:
        backend.ensure_loaded()
    except QasrError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeFailure(f"failed to load vLLM model: {exc}") from exc

    return core_stream.run_stream(backend, source, settings)


__all__ = ["handle"]
