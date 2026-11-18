"""``qasr align`` adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from .. import align as core
from ..catalog import FingerprintCache, require_local, resolve_aligner, resolve_model
from ..errors import CapabilityError, InputError, UsageError
from ..inputs import InputItem, sanitize_id
from ..languages import ALIGNABLE, detect_script_language, resolve_alignable
from ..render import FORMATS
from ..subtitles import cues_text, parse_srt
from .common import CommandContext, resolve_device_dtype, selected_backend


def _read_text(ns: Any) -> str:
    text = getattr(ns, "text", None)
    text_file = getattr(ns, "text_file", None)
    srt_file = getattr(ns, "srt", None)
    chosen = [value for value in (text, text_file, srt_file) if value]
    if len(chosen) != 1:
        raise UsageError("align needs exactly one of --text, --text-file, --srt")
    if text:
        return str(text)
    if text_file:
        target = Path(str(text_file)).expanduser()
        if not target.is_file():
            raise InputError(f"text file not found: {text_file}")
        return target.read_text(encoding="utf-8").strip()
    target = Path(str(srt_file)).expanduser()
    if not target.is_file():
        raise InputError(f"srt file not found: {srt_file}")
    return cues_text(parse_srt(target.read_text(encoding="utf-8")))


def _formats(ns: Any) -> List[str]:
    raw = getattr(ns, "format", None)
    if not raw:
        return ["json"]
    formats: List[str] = []
    for part in str(raw).split(","):
        fmt = part.strip()
        if not fmt:
            continue
        if fmt not in FORMATS:
            raise UsageError(f"unknown format: {fmt}", hint="choose from " + ", ".join(FORMATS))
        if fmt not in formats:
            formats.append(fmt)
    return formats or ["json"]


def handle(ns: Any, ctx: CommandContext) -> int:
    cfg = ctx.config
    if selected_backend(cfg) == "remote":
        raise CapabilityError(
            "remote backends do not provide an alignment endpoint",
            hint="run align locally with a forced aligner",
        )
    audio = getattr(ns, "audio", None)
    if not audio:
        raise UsageError("align requires an AUDIO argument")

    # Validate the audio up front: a missing/oversized file is an input error
    # (exit 4), never a runtime/model error.
    audio_path = Path(str(audio)).expanduser()
    if not audio_path.is_file():
        raise InputError(f"audio file not found: {audio_path}")
    from ..audio import probe

    info = probe(audio_path)

    text = _read_text(ns)
    if not text.strip():
        raise InputError("reference text is empty", hint="pass --text/--text-file/--srt")

    # ---- windowing ------------------------------------------------------
    window_seconds = getattr(ns, "window_seconds", None)
    if window_seconds is None:
        window_seconds = core.MAX_ALIGN_SECONDS
    window_seconds = float(window_seconds)
    if not 10.0 <= window_seconds <= core.MAX_ALIGN_SECONDS:
        raise UsageError(
            f"--window-seconds must be between 10 and {int(core.MAX_ALIGN_SECONDS)}",
            details={"window_seconds": window_seconds},
        )
    window_overlap = getattr(ns, "window_overlap", None)
    if window_overlap is None:
        window_overlap = 3.0
    window_overlap = float(window_overlap)
    if not 0.0 <= window_overlap <= 10.0 or window_overlap >= window_seconds:
        raise UsageError(
            "--window-overlap must satisfy 0 <= overlap <= 10 and overlap < window",
            details={"window_seconds": window_seconds, "window_overlap": window_overlap},
        )
    windows = core.long_audio.plan_windows(
        info.frames, info.sample_rate, window_s=window_seconds, overlap_s=window_overlap
    )

    raw_language = getattr(ns, "language", None)
    if raw_language is None and cfg.source_of("language") != "default":
        raw_language = cfg.get("language")
    if raw_language is None:
        raise UsageError(
            "align requires --language",
            hint="one of " + ", ".join(ALIGNABLE) + " (or 'auto' for script detection)",
        )

    language: Any = None
    language_source = "explicit"
    if str(raw_language).strip().casefold() == "auto":
        try:
            language = detect_script_language(text)
            language_source = "script"
            ctx.log(f"align: detected language {language} from script")
        except InputError:
            language = None
    else:
        language = resolve_alignable(str(raw_language))
    if language is not None and language not in ALIGNABLE:
        raise CapabilityError(f"language '{language}' cannot be aligned",
                              hint="alignable: " + ", ".join(ALIGNABLE))

    formats = _formats(ns)
    out_dir = Path(str(getattr(ns, "out_dir"))).expanduser() if getattr(ns, "out_dir", None) else None
    output = getattr(ns, "output", None)
    if out_dir is not None and output:
        raise UsageError("--out-dir and --output are mutually exclusive")
    if output and output != "-" and len(formats) > 1:
        raise UsageError("--output supports exactly one format", hint="use --out-dir")
    if len(formats) > 1 and out_dir is None and not output:
        raise UsageError("multiple formats require --out-dir",
                         hint="write files with -o DIR")

    if ctx.json_mode:
        stdout_mode = "json"
    elif output == "-" or (out_dir is None and not output):
        stdout_mode = "content"
    else:
        stdout_mode = "none"

    coverage_min = getattr(ns, "coverage_min", None)
    if coverage_min is None:
        coverage_min = 0.5
    coverage_min = float(coverage_min)
    if not 0.0 <= coverage_min <= 1.0:
        raise UsageError("--coverage-min must be between 0 and 1")

    item = InputItem(id=sanitize_id(Path(str(audio)).stem), source=str(audio),
                     path=Path(str(audio)).expanduser().resolve(), origin="file")

    aligner = resolve_aligner(cfg.get("aligner"), cfg, required=not ctx.dry_run)
    if aligner is None:
        raise CapabilityError("no aligner model available", hint="qasr models pull aligner")
    # Dry-run plans the resolved location/alias even when weights are absent.
    aligner_path = aligner.path or Path(aligner.display)
    if not ctx.dry_run and aligner.path is None:
        raise CapabilityError("no aligner model available", hint="qasr models pull aligner")

    device, dtype = resolve_device_dtype(cfg, optional=ctx.dry_run)

    fingerprint = None
    if not ctx.dry_run and aligner.present and aligner.path is not None:
        cache = FingerprintCache()
        fingerprint = cache.dir_fingerprint(aligner.path)
        cache.flush()

    # ASR is needed for window anchors (long audio) and for Latin-script LID.
    needs_asr = len(windows) > 1 or language is None
    asr_path = None
    asr_id = ""
    asr_fingerprint = None
    if needs_asr:
        asr_model = resolve_model(cfg.get("model"), cfg)
        asr_path = (
            require_local(asr_model, action="align")
            if not ctx.dry_run
            else (asr_model.path or Path(str(asr_model.value)))
        )
        asr_id = asr_model.repo or asr_model.alias or str(asr_path)
        if not ctx.dry_run and asr_model.present and asr_model.path is not None:
            cache = FingerprintCache()
            asr_fingerprint = cache.dir_fingerprint(asr_model.path)
            cache.flush()

    settings = core.AlignSettings(
        aligner_path=aligner_path,
        aligner_id=aligner.repo or "Qwen/Qwen3-ForcedAligner-0.6B",
        aligner_fingerprint=fingerprint,
        device=device,
        dtype=dtype,
        attn_implementation=str(cfg.get("attn_implementation") or "auto"),
        language=language or "auto",
        granularity=getattr(ns, "granularity", None) or "word",
        formats=formats,
        out_dir=out_dir,
        output=output,
        stdout_mode=stdout_mode,
        dry_run=ctx.dry_run,
        coverage_min=float(coverage_min),
        offline=bool(cfg.get("offline")),
        window_seconds=window_seconds,
        window_overlap=window_overlap,
        language_source=language_source,
        asr_path=asr_path,
        asr_id=asr_id,
        asr_fingerprint=asr_fingerprint,
    )

    if language is None:
        if ctx.dry_run:
            settings.language_source = "lid_probe"
        else:
            asr_backend = core.build_asr_backend(settings)
            ensure = getattr(asr_backend, "ensure_loaded", None)
            if callable(ensure):
                ensure()
            try:
                detected = core.probe_language(asr_backend, item)
            except Exception as exc:  # noqa: BLE001 - reported as input failure
                raise InputError(
                    "language identification probe failed",
                    hint="pass --language " + "|".join(ALIGNABLE),
                    details={"error": str(exc)},
                ) from exc
            if detected not in ALIGNABLE:
                raise InputError(
                    f"could not identify an alignable language (got {detected!r})",
                    hint="pass --language " + "|".join(ALIGNABLE),
                )
            language = detected
            settings.language = detected
            settings.language_source = "lid_probe"
            ctx.warn(f"align: identified language {detected} from audio (lid_probe)")

    result = core.run(settings, item, text, log=ctx.log)
    ctx.emit(result.stdout)
    if result.exit_code == 0 and settings.stdout_mode == "none":
        ctx.log("align: wrote " + ", ".join(formats))
    return result.exit_code
