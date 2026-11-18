"""Forced alignment of a known transcript (``qasr align``)."""

from __future__ import annotations

import difflib
import json
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import SCHEMA_VERSION
from .audio import probe
from .backends import AlignJob, AlignOptions, build_transformers_backend
from .errors import EXIT_OK, InputError, QasrError, QualityError
from .inputs import InputItem, safe_output_path, sanitize_id
from .jobs import atomic_write_bytes, sha256_bytes
from .render import align_result_payload, render_format
from .subtitles import build_cues, cues_text

MAX_ALIGN_SECONDS = 180.0

LogFn = Callable[[str], None]


def _noop(_msg: str) -> None:
    return None


@dataclass
class AlignSettings:
    aligner_path: Path
    aligner_id: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    aligner_fingerprint: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "auto"
    language: str = "Chinese"
    granularity: str = "word"
    formats: List[str] = field(default_factory=lambda: ["json"])
    out_dir: Optional[Path] = None
    output: Optional[str] = None
    stdout_mode: str = "none"  # none | json | content
    dry_run: bool = False
    coverage_min: float = 0.5
    offline: bool = False


@dataclass
class AlignRunResult:
    stdout: str = ""
    exit_code: int = EXIT_OK
    payload: Optional[Dict[str, Any]] = None


def normalize_for_coverage(text: str) -> str:
    """NFKC, drop whitespace and punctuation, keep letters/numbers/symbols."""
    text = unicodedata.normalize("NFKC", text or "")
    out: List[str] = []
    for ch in text:
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("Z"):
            continue
        out.append(ch.casefold())
    return "".join(out)


def compute_coverage(reference: str, hypothesis: str) -> float:
    ref = normalize_for_coverage(reference)
    hyp = normalize_for_coverage(hypothesis)
    if not ref:
        return 1.0
    if not hyp:
        return 0.0
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return round(matched / len(ref), 4)


def apply_granularity(items: List[Any], granularity: str) -> List[Any]:
    if granularity != "char":
        return items
    out = []
    for item in items:
        text = item.text
        if len(text) <= 1:
            out.append(item)
            continue
        span = max(0.0, item.end - item.start)
        step = span / len(text)
        for index, ch in enumerate(text):
            out.append(
                type(item)(
                    text=ch,
                    start=item.start + step * index,
                    end=item.start + step * (index + 1),
                    kind="char",
                )
            )
    return out


def run(
    settings: AlignSettings,
    item: InputItem,
    text: str,
    *,
    log: Optional[LogFn] = None,
) -> AlignRunResult:
    log = log or _noop

    info = probe(item.path)
    if info.duration_s > MAX_ALIGN_SECONDS + 1e-6:
        raise InputError(
            f"audio is {info.duration_s:.1f}s, longer than the P0 alignment limit of "
            f"{int(MAX_ALIGN_SECONDS)}s",
            hint="P1 adds --window-seconds; split the audio for now",
            details={"duration_s": info.duration_s, "limit_s": MAX_ALIGN_SECONDS},
        )
    if not text.strip():
        raise InputError("reference text is empty")

    if settings.dry_run:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "command": "align",
            "plan": {
                "input": {"source": item.source, "duration_s": info.duration_s},
                "language": settings.language,
                "aligner": {"id": settings.aligner_id, "path": str(settings.aligner_path)},
                "device": settings.device,
                "dtype": settings.dtype,
                "formats": settings.formats,
                "granularity": settings.granularity,
                "coverage_min": settings.coverage_min,
                "outputs": _plan_outputs(settings, item),
            },
        }
        return AlignRunResult(stdout=json.dumps(payload, ensure_ascii=False), payload=payload)

    backend = build_transformers_backend(
        settings.aligner_path,
        aligner_path=None,
        device=settings.device,
        dtype=settings.dtype,
        attn_implementation=settings.attn_implementation,
        max_inference_batch_size=1,
        max_new_tokens=None,
        offline=settings.offline,
    )

    started = time.perf_counter()
    ensure = getattr(backend, "ensure_aligner", None)
    if callable(ensure):
        ensure()
    load_ms = int((time.perf_counter() - started) * 1000)

    job = AlignJob(item=item, info=info, text=text, language=settings.language)
    infer_started = time.perf_counter()
    try:
        results = backend.align([job], AlignOptions(granularity=settings.granularity))
    except QasrError:
        raise
    except Exception as exc:  # noqa: BLE001
        from .errors import RuntimeFailure

        raise RuntimeFailure(f"alignment failed: {exc}") from exc
    infer_ms = int((time.perf_counter() - infer_started) * 1000)

    result = results[0]
    result.items = apply_granularity(result.items, settings.granularity)
    hypothesis = "".join(seg.text for seg in result.items)
    coverage = compute_coverage(text, hypothesis)

    if settings.coverage_min > 0 and coverage < settings.coverage_min:
        raise QualityError(
            f"alignment coverage {coverage:.3f} is below --coverage-min {settings.coverage_min}",
            hint="check the transcript/audio pair, or lower --coverage-min (0 disables)",
            details={"coverage": coverage, "threshold": settings.coverage_min},
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": "align",
        "input": {"source": item.source, "duration_s": info.duration_s},
        "language": settings.language,
        "result": align_result_payload(result, coverage),
        "metrics": {"load_ms": load_ms, "infer_ms": infer_ms},
    }

    outputs = _write_outputs(settings, item, info, result, payload, coverage)
    payload["outputs"] = outputs

    stdout = ""
    if settings.stdout_mode == "json":
        stdout = json.dumps(payload, ensure_ascii=False)
    elif settings.stdout_mode == "content":
        if settings.formats == ["json"] or "json" in settings.formats:
            stdout = json.dumps(payload, ensure_ascii=False)
        else:
            fmt = settings.formats[0]
            if fmt in ("srt", "vtt"):
                stdout = render_format(
                    fmt, item=payload, result=None,
                    audio_duration_s=info.duration_s,
                    cues=build_cues(result.items, audio_duration_s=info.duration_s),
                )
            else:
                stdout = json.dumps(payload, ensure_ascii=False)

    log(f"aligned {len(result.items)} items, coverage {coverage:.3f}")
    return AlignRunResult(stdout=stdout, exit_code=EXIT_OK, payload=payload)


def _plan_outputs(settings: AlignSettings, item: InputItem) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if settings.output and settings.output != "-":
        out.append({"format": settings.formats[0], "path": settings.output})
    elif settings.out_dir is not None:
        base = sanitize_id(Path(item.source).stem)
        for fmt in settings.formats:
            out.append({"format": fmt, "path": str(Path(str(settings.out_dir)) / f"{base}.{fmt}")})
    return out


def _write_outputs(
    settings: AlignSettings,
    item: InputItem,
    info: Any,
    result: Any,
    payload: Dict[str, Any],
    coverage: float,
) -> List[Dict[str, Any]]:
    if not settings.out_dir and not (settings.output and settings.output != "-"):
        return []
    cues = build_cues(result.items, audio_duration_s=info.duration_s)
    targets: List[tuple[str, Path]] = []
    if settings.output and settings.output != "-":
        targets.append((settings.formats[0], Path(settings.output).expanduser()))
    else:
        assert settings.out_dir is not None
        base = sanitize_id(Path(item.source).stem)
        for fmt in settings.formats:
            targets.append((fmt, safe_output_path(settings.out_dir, f"{base}.{fmt}")))

    outputs: List[Dict[str, Any]] = []
    for fmt, target in targets:
        if fmt == "json":
            content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        elif fmt == "jsonl":
            content = json.dumps(payload, ensure_ascii=False) + "\n"
        elif fmt == "txt":
            content = cues_text(cues) + "\n"
        else:
            content = render_format(
                fmt, item=payload, result=None,
                audio_duration_s=info.duration_s, cues=cues,
            )
        data = content.encode("utf-8")
        atomic_write_bytes(target, data)
        display = settings.output if (settings.output and settings.output != "-") else (
            str(Path(str(settings.out_dir)) / target.name)
        )
        outputs.append({"format": fmt, "path": display, "sha256": sha256_bytes(data)})
    return outputs


__all__ = [
    "AlignSettings",
    "MAX_ALIGN_SECONDS",
    "apply_granularity",
    "compute_coverage",
    "normalize_for_coverage",
    "run",
]
