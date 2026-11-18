"""``qasr transcribe`` adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .. import transcribe as core
from ..catalog import FingerprintCache, require_local, resolve_aligner, resolve_model
from ..errors import (
    CapabilityError,
    InputError,
    ModelError,
    QasrError,
    UsageError,
)
from ..inputs import InputItem, expand_inputs, resolve_manifest_audio
from ..jobs import ensure_output_unlocked, load_manifest
from ..languages import resolve_alignable, resolve_forceable
from .common import CommandContext, parse_auto_int, resolve_device_dtype

GRANULARITIES = ("sentence", "word", "char")


def _preflight(specs: List[core.ItemSpec], *, dry_run: bool) -> None:
    """Probe inputs up front so bad files fail before model/runtime checks.

    Failures stay on the spec (not in ``spec_errors``) so state/manifest runs
    can record a ``failed`` entry for them.
    """
    from ..audio import probe

    for spec in specs:
        try:
            if spec.item.path is None:
                raise InputError(f"no local file for input {spec.id}")
            if dry_run and spec.item.origin == "url" and not spec.item.path.exists():
                continue
            spec.info = probe(spec.item.path)
        except QasrError as exc:
            spec.error = exc


def _read_context_file(path: str) -> List[str]:
    target = Path(path).expanduser()
    if not target.is_file():
        raise InputError(f"context file not found: {path}")
    lines: List[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _split_language(raw: str, *, timestamps: bool) -> Optional[str]:
    if raw is None or not str(raw).strip():
        raw = "auto"
    raw = str(raw).strip()
    if raw.casefold() == "auto":
        if timestamps:
            raise CapabilityError(
                "timestamps require an explicit alignment language",
                hint="pass --language zh|en|yue|fr|de|it|ja|ko|pt|ru|es",
            )
        return None
    if timestamps:
        return resolve_alignable(raw)
    return resolve_forceable(raw)


def _normalize_formats(value: Any, default: Sequence[str]) -> List[str]:
    return core.parse_formats(value, default)


def handle(ns: Any, ctx: CommandContext) -> int:
    cfg = ctx.config

    if getattr(ns, "manifest", None) and getattr(ns, "inputs", None):
        raise UsageError("--manifest cannot be combined with positional inputs")

    timestamps_default = bool(getattr(ns, "timestamps", False))
    granularity_default = getattr(ns, "granularity", None) or "sentence"
    if granularity_default not in GRANULARITIES:
        raise UsageError(f"--granularity must be one of {GRANULARITIES}")

    context_parts: List[str] = list(getattr(ns, "context", None) or [])
    if getattr(ns, "context_file", None):
        context_parts.extend(_read_context_file(ns.context_file))
    context_default = " ".join(part.strip() for part in context_parts if part.strip())

    raw_language = getattr(ns, "language", None)
    if raw_language is None:
        raw_language = cfg.get("language") or "auto"
    language_default = _split_language(str(raw_language), timestamps=timestamps_default)

    # ---- inputs ---------------------------------------------------------
    specs: List[core.ItemSpec] = []
    spec_errors: List[Dict[str, Any]] = []

    if getattr(ns, "manifest", None):
        if getattr(ns, "out_dir", None) is None:
            raise UsageError("--manifest requires --out-dir",
                             hint="state and outputs are written under the output directory")
        records = load_manifest(ns.manifest)
        for record in records:
            try:
                item = resolve_manifest_audio(
                    record, offline=bool(cfg.get("offline")), dry_run=ctx.dry_run
                )
            except QasrError as exc:
                spec_errors.append({"id": record.get("id"), "source": record.get("audio")
                                    or "audio_base64", "error": exc})
                continue
            specs.append(_spec_from_record(record, item, context_default,
                                           language_default, timestamps_default,
                                           granularity_default, ns, cfg))
    else:
        inputs = list(getattr(ns, "inputs", None) or [])
        if not inputs:
            raise UsageError("no input given", hint="qasr transcribe INPUT... | --manifest FILE")
        expansion = expand_inputs(
            inputs,
            recursive=bool(getattr(ns, "recursive", False)),
            offline=bool(cfg.get("offline")),
            dry_run=ctx.dry_run,
        )
        spec_errors.extend(expansion.errors)
        formats = _normalize_formats(getattr(ns, "format", None),
                                     _default_formats_hint(ns))
        for item in expansion.items:
            specs.append(
                core.ItemSpec(
                    item=item,
                    language=language_default,
                    context=context_default,
                    formats=list(formats),
                    timestamps=timestamps_default,
                    granularity=granularity_default,
                    output_base=item.id,
                )
            )

    # ---- preflight: validate every input before touching the runtime ----
    _preflight(specs, dry_run=ctx.dry_run)

    if not specs and not spec_errors:
        raise UsageError("no input given")

    # ---- output formats / targets --------------------------------------
    out_dir = Path(str(getattr(ns, "out_dir"))).expanduser() if getattr(ns, "out_dir", None) else None
    output = getattr(ns, "output", None)
    if out_dir is not None and output:
        raise UsageError("--out-dir and --output are mutually exclusive")

    total_inputs = len(specs) + len(spec_errors)
    if output and total_inputs > 1:
        raise UsageError("--output accepts a single input only",
                         hint="use --out-dir for multiple inputs")
    if getattr(ns, "resume", False) and out_dir is None:
        raise UsageError("--resume requires --out-dir",
                         hint="resume state lives in <out-dir>/.qasr/state.jsonl")
    if out_dir is not None and not ctx.dry_run:
        ensure_output_unlocked(out_dir)

    if getattr(ns, "format", None):
        formats = _normalize_formats(ns.format, ["txt"])
        for spec in specs:
            spec.formats = list(formats)

    if output:
        for spec in specs:
            if len(spec.formats) > 1:
                raise UsageError("--output supports exactly one format",
                                 hint="use --out-dir for multiple formats")

    for spec in specs:
        if any(fmt in ("srt", "vtt") for fmt in spec.formats) and not spec.timestamps:
            raise CapabilityError(
                "srt/vtt output requires timestamps",
                hint="pass --timestamps with an explicit alignable --language",
            )
        if spec.timestamps and spec.language is None:
            raise CapabilityError(
                "timestamps require an explicit alignment language",
                hint="pass --language zh|en|yue|fr|de|it|ja|ko|pt|ru|es",
            )

    # ---- stdout mode ----------------------------------------------------
    total = len(specs) + len(spec_errors)
    if ctx.jsonl_mode:
        stdout_mode = "jsonl"
    elif ctx.json_mode:
        stdout_mode = "json"
    elif output == "-":
        if total > 1:
            raise UsageError("--output - accepts a single input only",
                             hint="use --jsonl for multiple inputs")
        selected = specs[0].formats if specs else []
        if "jsonl" in selected:
            stdout_mode = "jsonl"
        elif "json" in selected:
            stdout_mode = "json"
        elif any(fmt in ("srt", "vtt") for fmt in selected):
            stdout_mode = "subtitle"
        else:
            stdout_mode = "text"
    elif out_dir is not None or output:
        stdout_mode = "none"
    elif total == 1:
        selected = specs[0].formats if specs else ["txt"]
        if getattr(ns, "format", None) and selected[0] in ("srt", "vtt"):
            stdout_mode = "subtitle"
        elif getattr(ns, "format", None) and selected[0] == "jsonl":
            stdout_mode = "jsonl"
        elif getattr(ns, "format", None) and selected[0] == "json":
            stdout_mode = "json"
        else:
            stdout_mode = "text"
    else:
        raise UsageError(
            "multiple inputs require --out-dir (or --json/--jsonl)",
            hint="write files with -o DIR, or stream JSON with --jsonl",
        )

    if (ctx.json_mode or ctx.jsonl_mode) and getattr(ns, "format", None):
        chosen = _normalize_formats(ns.format, ["txt"])
        if any(fmt in ("json", "jsonl") for fmt in chosen):
            raise UsageError("--format json/jsonl cannot be combined with --json/--jsonl",
                             hint="pick one output channel")

    # ---- numeric options (validate before any runtime work) -------------
    batch_size = getattr(ns, "batch_size", None)
    if batch_size is None:
        batch_size = cfg.get("batch_size")
    if batch_size is None:
        batch_size = 1
    if int(batch_size) < 1:
        raise UsageError("--batch-size must be >= 1")
    max_inference = getattr(ns, "max_inference_batch_size", None)
    if max_inference is None:
        max_inference = cfg.get("max_inference_batch_size")
    if max_inference is None:
        max_inference = batch_size
    if int(max_inference) < 1:
        raise UsageError("--max-inference-batch-size must be >= 1")
    if getattr(ns, "jobs", None) and int(ns.jobs) > 1:
        ctx.warn("local mode forces --jobs 1 (use --batch-size for throughput)")

    max_new_tokens = parse_auto_int(getattr(ns, "max_new_tokens", None))

    # ---- model / aligner ------------------------------------------------
    # When every input already failed preflight, report those errors before
    # touching the runtime (a missing model must not mask them). Zero-sample
    # audio produces an empty result without inference, so it needs no model.
    needs_runtime = any(
        spec.error is None and (spec.info is None or not spec.info.is_empty)
        for spec in specs
    )
    model = resolve_model(cfg.get("model"), cfg)
    if not ctx.dry_run and needs_runtime:
        require_local(model, action="transcribe")
    model_path = model.path or Path(str(model.value))

    need_align = any(spec.timestamps for spec in specs)
    aligner = resolve_aligner(cfg.get("aligner"), cfg, required=need_align and not ctx.dry_run) if need_align else None
    aligner_path = aligner.path if aligner is not None else None

    backend_name = str(cfg.get("backend") or "auto")
    if backend_name == "auto":
        backend_name = "transformers"
    if backend_name != "transformers":
        raise ModelError(
            f"backend '{backend_name}' is not available in P0",
            hint="P0 supports the local transformers backend; vLLM/remote arrive in P1",
        )

    device, dtype = resolve_device_dtype(cfg, optional=ctx.dry_run or not needs_runtime)

    model_fingerprint = None
    aligner_fingerprint = None
    if not ctx.dry_run and needs_runtime:
        cache = FingerprintCache()
        if model.present and model.path is not None:
            model_fingerprint = cache.dir_fingerprint(model.path)
        if aligner is not None and aligner.present and aligner.path is not None:
            aligner_fingerprint = cache.dir_fingerprint(aligner.path)
        cache.flush()

    settings = core.TranscribeSettings(
        model_path=model_path,
        model_id=model.repo or model.alias or str(model_path),
        model_fingerprint=model_fingerprint,
        aligner_path=aligner_path,
        aligner_fingerprint=aligner_fingerprint,
        backend_name=backend_name,
        device=device,
        dtype=dtype,
        attn_implementation=str(cfg.get("attn_implementation") or "auto"),
        batch_size=int(batch_size),
        max_inference_batch_size=int(max_inference),
        max_new_tokens=max_new_tokens,
        formats=specs[0].formats if specs else ["txt"],
        out_dir=out_dir,
        output=output,
        stdout_mode=stdout_mode,
        dry_run=ctx.dry_run,
        resume=bool(getattr(ns, "resume", False)),
        overwrite=bool(getattr(ns, "overwrite", False)),
        continue_on_error=not bool(getattr(ns, "fail_fast", False)),
        fail_fast=bool(getattr(ns, "fail_fast", False)),
        offline=bool(cfg.get("offline")),
    )

    result = core.run(settings, specs, spec_errors, log=ctx.log, config=cfg)
    ctx.emit(result.stdout)
    return result.exit_code


def _spec_from_record(
    record: Dict[str, Any],
    item: InputItem,
    context_default: str,
    language_default: Optional[str],
    timestamps_default: bool,
    granularity_default: str,
    ns: Any,
    cfg: Any,
) -> core.ItemSpec:
    timestamps = bool(record.get("timestamps", timestamps_default))
    raw_language = record.get("language")
    if raw_language is None:
        language = language_default
    else:
        language = _split_language(str(raw_language), timestamps=timestamps)
    if timestamps and language is None:
        raise CapabilityError(
            "timestamps require an explicit alignment language",
            hint="set 'language' in the manifest record or pass --language",
        )
    formats = _normalize_formats(record.get("formats"), ["txt"]) if record.get("formats") \
        else _normalize_formats(getattr(ns, "format", None), ["txt"])
    granularity = record.get("granularity") or granularity_default
    if granularity not in GRANULARITIES:
        raise UsageError(f"invalid granularity: {granularity}")
    output_base = core.validate_output_base(record.get("output", ""), item.id)
    context = record.get("context", context_default) or ""
    return core.ItemSpec(
        item=item,
        language=language,
        context=str(context),
        formats=list(formats),
        timestamps=timestamps,
        granularity=granularity,
        output_base=output_base,
    )


def _default_formats_hint(ns: Any) -> List[str]:
    if getattr(ns, "format", None):
        return _normalize_formats(ns.format, ["txt"])
    out_dir = getattr(ns, "out_dir", None)
    output = getattr(ns, "output", None)
    return core.default_output_formats(
        Path(str(out_dir)) if out_dir else None,
        output,
    )
