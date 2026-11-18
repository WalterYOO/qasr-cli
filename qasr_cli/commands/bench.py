"""``qasr bench``: synthetic or file-based throughput measurement."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from .. import SCHEMA_VERSION
from ..audio import AudioInfo, probe, write_wav_16k
from ..backends import TranscribeJob, TranscribeOptions
from ..catalog import require_local, resolve_model
from ..errors import EXIT_OK, UsageError
from ..inputs import InputItem, expand_inputs
from .common import (
    CommandContext,
    resolve_backend,
    resolve_device_dtype,
    selected_backend,
)


def _peak_memory_mb() -> float | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    import sys

    if sys.platform == "darwin":
        return round(usage / (1024 * 1024), 1)
    return round(usage / 1024, 1)


def _synthetic(duration: float, directory: Path) -> tuple[Path, AudioInfo]:
    import numpy as np

    samples = np.zeros(int(round(duration * 16000)), dtype="float32")
    path = write_wav_16k(directory / "synthetic.wav", samples)
    return path, AudioInfo(
        path=path,
        duration_s=duration,
        sample_rate=16000,
        channels=1,
        frames=int(len(samples)),
    )


def handle(ns: Any, ctx: CommandContext) -> int:
    cfg = ctx.config
    duration = float(getattr(ns, "duration", None) or 10.0)
    if duration <= 0:
        raise UsageError("--duration must be > 0")
    iterations = int(getattr(ns, "iterations", None) or 3)
    if not 1 <= iterations <= 100:
        raise UsageError("--iterations must be between 1 and 100")

    is_remote = selected_backend(cfg) == "remote"
    backend_choice, backend_fallback = resolve_backend(cfg)
    model = resolve_model(cfg.get("model"), cfg)
    model_id = model.repo or model.alias or str(model.value)
    if is_remote and cfg.source_of("model") == "default":
        model_id = ""

    if is_remote:
        device, dtype = "remote", "fp32"
    else:
        device, dtype = resolve_device_dtype(cfg, optional=ctx.dry_run)
    if backend_choice == "vllm" and not is_remote:
        from ..runtime import set_cuda_visible_device

        set_cuda_visible_device(str(cfg.get("device") or "auto"))

    inputs = list(getattr(ns, "inputs", None) or [])
    if ctx.dry_run:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "command": "bench",
            "plan": {
                "backend": backend_choice,
                "device": device,
                "dtype": dtype,
                "model": model_id,
                "iterations": iterations,
                "inputs": inputs or [f"synthetic:{duration:g}s"],
                "synthetic": not inputs,
                "duration_s": duration,
            },
        }
        ctx.emit(json.dumps(payload, ensure_ascii=False))
        return EXIT_OK

    work: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qasr-bench-") as raw_tmp:
        tmp = Path(raw_tmp)
        if inputs:
            expansion = expand_inputs(inputs, offline=bool(cfg.get("offline")))
            for item in expansion.items:
                info = probe(item.path)
                work.append({"item": item, "info": info, "synthetic": False})
        else:
            path, info = _synthetic(duration, tmp)
            item = InputItem(id="synthetic", source=str(path), path=path, origin="file")
            work.append({"item": item, "info": info, "synthetic": True})

        warnings: List[str] = []
        started = time.perf_counter()
        if is_remote:
            from ..remote import RemoteBackend, RemoteOptions

            backend = RemoteBackend(
                RemoteOptions(
                    base_url=str(cfg.get("server") or ""),
                    model=model_id or None,
                    timeout=float(cfg.get("timeout") or 300.0),
                    retries=int(cfg.get("retries") or 2),
                    log=ctx.log,
                )
            )
            backend.prepare()
            model_id = getattr(backend, "resolved_model_id", None) or model_id
            warnings.append("bench includes network time in RTF")
        else:
            require_local(model, action="bench")
            model_path = model.path or Path(str(model.value))
            if backend_choice == "vllm":
                from ..backends import build_vllm_backend

                backend = build_vllm_backend(
                    model_path,
                    device=device,
                    dtype=dtype,
                    offline=bool(cfg.get("offline")),
                )
            else:
                from ..backends import build_transformers_backend

                backend = build_transformers_backend(
                    model_path,
                    device=device,
                    dtype=dtype,
                    offline=bool(cfg.get("offline")),
                )
            ensure = getattr(backend, "ensure_loaded", None)
            if callable(ensure):
                ensure(need_align=False)
        load_s = round(time.perf_counter() - started, 3)

        elapsed = 0.0
        audio_s = 0.0
        for entry in work:
            for _ in range(iterations):
                begin = time.perf_counter()
                backend.transcribe(
                    [
                        TranscribeJob(
                            item=entry["item"],
                            info=entry["info"],
                            language=None,
                        )
                    ],
                    TranscribeOptions(timestamps=False),
                )
                elapsed += time.perf_counter() - begin
                audio_s += entry["info"].duration_s

    rtf = round(elapsed / audio_s, 4) if audio_s > 0 else None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": "bench",
        "backend": backend_choice,
        "backend_fallback": backend_fallback,
        "device": device,
        "dtype": dtype,
        "model": model_id,
        "load_s": load_s,
        "audio_s": round(audio_s, 3),
        "elapsed_s": round(elapsed, 3),
        "rtf": rtf,
        "iterations": iterations,
        "peak_memory_mb": _peak_memory_mb(),
        "synthetic": all(entry["synthetic"] for entry in work),
        "warnings": warnings,
    }
    if ctx.json_mode:
        ctx.emit(json.dumps(payload, ensure_ascii=False))
    else:
        print(
            f"{backend_choice}/{device}/{dtype} model={model_id} "
            f"load={load_s}s audio={audio_s:.2f}s elapsed={elapsed:.2f}s rtf={rtf}"
        )
    return EXIT_OK
