"""Forced alignment of a known transcript (``qasr align``)."""

from __future__ import annotations

import difflib
import json
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import SCHEMA_VERSION, long_audio
from .audio import AudioInfo, probe, read_window_16k, write_wav_16k
from .backends import (
    AlignJob,
    AlignOptions,
    AlignResult,
    Segment,
    TranscribeJob,
    TranscribeOptions,
    build_transformers_backend,
)
from .errors import EXIT_OK, CapabilityError, InputError, QasrError, QualityError, RuntimeFailure
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
    window_seconds: float = MAX_ALIGN_SECONDS
    window_overlap: float = long_audio.DEFAULT_OVERLAP_SECONDS
    language_source: str = "explicit"
    asr_path: Optional[Path] = None
    asr_id: str = ""
    asr_fingerprint: Optional[str] = None


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


def plan_align_windows(
    info: AudioInfo, settings: AlignSettings
) -> List[long_audio.Window]:
    """Plan windows; raises UsageError (exit 2) for invalid parameters."""
    return long_audio.plan_windows(
        info.frames,
        info.sample_rate,
        window_s=settings.window_seconds,
        overlap_s=settings.window_overlap,
    )


def _build_aligner_backend(settings: AlignSettings) -> Any:
    return build_transformers_backend(
        settings.aligner_path,
        aligner_path=None,
        device=settings.device,
        dtype=settings.dtype,
        attn_implementation=settings.attn_implementation,
        max_inference_batch_size=1,
        max_new_tokens=None,
        offline=settings.offline,
    )


def build_asr_backend(settings: AlignSettings) -> Any:
    """ASR backend used for window anchors and language identification."""
    if settings.asr_path is None:
        raise CapabilityError(
            "long-form alignment needs an ASR model for window anchors",
            hint="qasr models pull 0.6b  (or pass --model PATH)",
            details={"asr_id": settings.asr_id or None},
        )
    return build_transformers_backend(
        settings.asr_path,
        aligner_path=None,
        device=settings.device,
        dtype=settings.dtype,
        attn_implementation=settings.attn_implementation,
        max_inference_batch_size=1,
        max_new_tokens=None,
        offline=settings.offline,
    )


def _window_audio(
    item: InputItem, window: long_audio.Window, tmp_dir: Path
) -> tuple[InputItem, AudioInfo]:
    assert item.path is not None
    samples = read_window_16k(
        item.path, window.audio_start_sample, window.audio_end_sample
    )
    path = write_wav_16k(tmp_dir / f"window-{window.index}.wav", samples)
    duration = len(samples) / 16000.0
    return (
        InputItem(
            id=f"{item.id}#{window.index}",
            source=str(path),
            path=path,
            origin="file",
        ),
        AudioInfo(
            path=path,
            duration_s=duration,
            sample_rate=16000,
            channels=1,
            frames=int(len(samples)),
        ),
    )


def probe_language(
    asr_backend: Any, item: InputItem, *, seconds: float = 30.0
) -> Optional[str]:
    """Run ASR on the first ``seconds`` of audio to identify the language."""
    assert item.path is not None
    info = probe(item.path)
    end = min(info.frames, int(round(seconds * info.sample_rate)))
    if end <= 0:
        return None
    with tempfile.TemporaryDirectory(prefix="qasr-lid-") as raw_tmp:
        samples = read_window_16k(item.path, 0, end)
        if len(samples) == 0:
            return None
        path = write_wav_16k(Path(raw_tmp) / "probe.wav", samples)
        window_info = AudioInfo(
            path=path,
            duration_s=len(samples) / 16000.0,
            sample_rate=16000,
            channels=1,
            frames=int(len(samples)),
        )
        window_item = InputItem(
            id=f"{item.id}#lid", source=str(path), path=path, origin="file"
        )
        result = asr_backend.transcribe(
            [TranscribeJob(item=window_item, info=window_info)],
            TranscribeOptions(timestamps=False),
        )[0]
    return result.language


def _windowed_items(
    settings: AlignSettings,
    item: InputItem,
    info: AudioInfo,
    text: str,
    windows: Sequence[long_audio.Window],
    *,
    log: LogFn,
) -> tuple[List[Any], List[str]]:
    """Align each window against its own slice of the reference text."""
    asr = build_asr_backend(settings)
    ensure_asr = getattr(asr, "ensure_loaded", None)
    if callable(ensure_asr):
        ensure_asr()
    aligner = _build_aligner_backend(settings)
    ensure_aligner = getattr(aligner, "ensure_aligner", None)
    if callable(ensure_aligner):
        ensure_aligner()

    warnings: List[str] = []
    anchors: List[str] = []
    with tempfile.TemporaryDirectory(prefix="qasr-align-") as raw_tmp:
        tmp_dir = Path(raw_tmp)
        prepared: Dict[int, tuple[InputItem, AudioInfo]] = {}
        for window in windows:
            window_item, window_info = _window_audio(item, window, tmp_dir)
            prepared[window.index] = (window_item, window_info)
            try:
                anchor = asr.transcribe(
                    [TranscribeJob(item=window_item, info=window_info)],
                    TranscribeOptions(timestamps=False),
                )[0]
                anchors.append(anchor.text)
                warnings.extend(anchor.warnings)
            except QasrError as exc:
                warnings.append(f"window_asr_failed:{window.index}:{exc.message}")
                anchors.append("")

        spans = long_audio.attribute_reference(text, anchors)
        items: List[Any] = []
        failures: List[QasrError] = []
        attempted = 0
        for window, span in zip(windows, spans):
            if span.is_weak:
                warnings.append(
                    f"chunk_reference_match_low:{window.index}:{span.match_ratio:.2f}"
                )
            if not span.text.strip():
                continue
            attempted += 1
            window_item, window_info = prepared[window.index]
            core_start_rel = (
                window.core_start_sample - window.audio_start_sample
            ) / info.sample_rate
            core_end_rel = (
                window.core_end_sample - window.audio_start_sample
            ) / info.sample_rate
            offset = window.audio_start_sample / info.sample_rate
            try:
                aligned = aligner.align(
                    [
                        AlignJob(
                            item=window_item,
                            info=window_info,
                            text=span.text,
                            language=settings.language,
                        )
                    ],
                    AlignOptions(granularity="word"),
                )[0]
            except QasrError as exc:
                failures.append(exc)
                warnings.append(f"window_align_failed:{window.index}:{exc.message}")
                log(f"align: window {window.index} failed: {exc.message}")
                continue
            warnings.extend(aligned.warnings)
            for seg in aligned.items:
                midpoint = (seg.start + seg.end) / 2.0
                if not (core_start_rel <= midpoint < core_end_rel):
                    continue
                items.append(
                    Segment(
                        text=seg.text,
                        start=seg.start + offset,
                        end=seg.end + offset,
                        kind=seg.kind,
                    )
                )
        if attempted and len(failures) == attempted:
            raise failures[0]
    return items, warnings


def run(
    settings: AlignSettings,
    item: InputItem,
    text: str,
    *,
    log: Optional[LogFn] = None,
) -> AlignRunResult:
    log = log or _noop

    info = probe(item.path)
    if not text.strip():
        raise InputError("reference text is empty")

    windows = plan_align_windows(info, settings)

    if settings.dry_run:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "command": "align",
            "plan": {
                "input": {"source": item.source, "duration_s": info.duration_s},
                "language": settings.language,
                "language_source": settings.language_source,
                "aligner": {"id": settings.aligner_id, "path": str(settings.aligner_path)},
                "device": settings.device,
                "dtype": settings.dtype,
                "formats": settings.formats,
                "granularity": settings.granularity,
                "coverage_min": settings.coverage_min,
                "window_seconds": settings.window_seconds,
                "window_overlap": settings.window_overlap,
                "windows": [window.to_dict(info.sample_rate) for window in windows],
                "outputs": _plan_outputs(settings, item),
            },
        }
        return AlignRunResult(stdout=json.dumps(payload, ensure_ascii=False), payload=payload)

    started = time.perf_counter()
    warnings: List[str] = []
    plan_payload = {
        "window_seconds": settings.window_seconds,
        "window_overlap": settings.window_overlap,
        "count": len(windows),
        "windows": [window.to_dict(info.sample_rate) for window in windows],
    }

    if len(windows) <= 1:
        backend = _build_aligner_backend(settings)
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
            raise RuntimeFailure(f"alignment failed: {exc}") from exc
        infer_ms = int((time.perf_counter() - infer_started) * 1000)

        result = results[0]
        result.items = apply_granularity(result.items, settings.granularity)
    else:
        log(f"align: {len(windows)} windows of {settings.window_seconds:g}s")
        items, warnings = _windowed_items(settings, item, info, text, windows, log=log)
        infer_ms = int((time.perf_counter() - started) * 1000)
        load_ms = 0
        merged = long_audio.merge_items(items, duration_s=info.duration_s)
        merged = apply_granularity(merged, settings.granularity)
        result = AlignResult(items=merged, warnings=warnings)

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
        "language_source": settings.language_source,
        "result": align_result_payload(result, coverage),
        "metrics": {"load_ms": load_ms, "infer_ms": infer_ms},
        "window_plan": plan_payload,
    }
    if warnings:
        payload["warnings"] = sorted(set(warnings))

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
            if fmt in ("srt", "vtt", "tsv", "md"):
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
