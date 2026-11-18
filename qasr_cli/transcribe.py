"""Transcription orchestration: inputs -> micro-batches -> results -> outputs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import SCHEMA_VERSION
from .audio import AudioInfo, max_new_tokens_for, probe
from .backends import (
    TranscribeJob,
    TranscribeOptions,
    TranscribeResult,
    build_transformers_backend,
)
from .config import Config
from .errors import (
    EXIT_INPUT,
    EXIT_OK,
    EXIT_PARTIAL,
    CapabilityError,
    InputError,
    QasrError,
    RuntimeFailure,
    UsageError,
)
from .inputs import InputItem, safe_output_path, sanitize_rel_path
from .jobs import (
    MANIFEST_FILENAME,
    STATE_DIRNAME,
    STATE_FILENAME,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    OutputLock,
    append_jsonl,
    atomic_write_bytes,
    atomic_write_text,
    now_iso,
    read_state,
    recover_running,
    sha256_bytes,
    sha256_file,
    should_skip,
)
from .jobs import (
    job_key as compute_job_key,
)
from .render import (
    FORMATS,
    SUBTITLE_FORMATS,
    batch_payload,
    error_payload,
    item_error_payload,
    item_payload,
    render_format,
    result_payload,
    subtitle_cues,
)

LogFn = Callable[[str], None]


def _noop(_msg: str) -> None:
    return None


@dataclass
class ItemSpec:
    """One requested transcription before inference."""

    item: InputItem
    language: Optional[str] = None
    context: str = ""
    formats: List[str] = field(default_factory=list)
    timestamps: bool = False
    granularity: str = "sentence"
    output_base: str = ""
    info: Optional[AudioInfo] = None
    audio_sha256: str = ""
    job_key: str = ""
    error: Optional[QasrError] = None
    empty: bool = False
    skipped: bool = False
    started_at: Optional[str] = None
    result: Any = None
    payload: Optional[Dict[str, Any]] = None
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.item.id


@dataclass
class TranscribeSettings:
    model_path: Path
    model_id: str
    model_fingerprint: Optional[str] = None
    aligner_path: Optional[Path] = None
    aligner_fingerprint: Optional[str] = None
    backend_name: str = "transformers"
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "auto"
    batch_size: int = 1
    max_inference_batch_size: int = 1
    max_new_tokens: Optional[int] = None  # None -> auto
    formats: List[str] = field(default_factory=lambda: ["txt"])
    out_dir: Optional[Path] = None
    output: Optional[str] = None
    stdout_mode: str = "none"  # none | text | json | jsonl | subtitle
    dry_run: bool = False
    resume: bool = False
    overwrite: bool = False
    continue_on_error: bool = True
    fail_fast: bool = False
    offline: bool = False


@dataclass
class RunResult:
    stdout: str = ""
    exit_code: int = EXIT_OK
    payload: Optional[Dict[str, Any]] = None
    summary: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Spec construction helpers
# ---------------------------------------------------------------------------

def parse_formats(value: Any, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        raw = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise UsageError("--format must be a comma-separated list")
    out: List[str] = []
    for fmt in raw:
        if fmt not in FORMATS:
            raise UsageError(f"unknown format: {fmt}", hint="choose from " + ", ".join(FORMATS))
        if fmt not in out:
            out.append(fmt)
    return out or list(default)


def validate_output_base(value: str, fallback: str) -> str:
    value = (value or "").strip()
    if not value:
        return fallback
    if "\x00" in value:
        raise UsageError("output name contains NUL")
    if Path(value).is_absolute():
        raise UsageError(f"manifest output must be relative: {value}")
    if value.startswith(".") and ".." in Path(value).parts:
        raise UsageError(f"manifest output must not contain '..': {value}")
    if any(part == ".." for part in Path(value).parts):
        raise UsageError(f"manifest output must not contain '..': {value}")
    return sanitize_rel_path(value, fallback=fallback)


def check_output_conflicts(specs: Sequence[ItemSpec]) -> None:
    seen: Dict[str, str] = {}
    for spec in specs:
        for fmt in spec.formats:
            rel = f"{spec.output_base}.{fmt}"
            if rel in seen and seen[rel] != spec.id:
                raise UsageError(
                    f"two inputs map to the same output file: {rel}",
                    hint="use --manifest with explicit 'output' names",
                    details={"ids": [seen[rel], spec.id]},
                )
            seen[rel] = spec.id


def default_output_formats(out_dir: Optional[Path], output: Optional[str]) -> List[str]:
    if output and output != "-":
        suffix = Path(output).suffix.casefold().lstrip(".")
        if suffix in FORMATS:
            return [suffix]
    return ["txt"]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(
    settings: TranscribeSettings,
    specs: List[ItemSpec],
    spec_errors: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    log: Optional[LogFn] = None,
    config: Optional[Config] = None,
) -> RunResult:
    log = log or _noop
    spec_errors = list(spec_errors or [])
    total_requested = len(specs) + len(spec_errors)

    # ---- probe & hash ---------------------------------------------------
    for spec in specs:
        if spec.item.path is None:
            spec.error = InputError(f"no local file for input {spec.id}")
            continue
        if (
            settings.dry_run
            and spec.item.origin == "url"
            and not spec.item.path.exists()
        ):
            # Dry-run plans URL inputs without downloading them.
            continue
        try:
            spec.info = probe(spec.item.path)
            spec.audio_sha256 = sha256_file(spec.item.path)
        except QasrError as exc:
            spec.error = exc
        except OSError as exc:
            spec.error = InputError(f"cannot read {spec.item.path}: {exc}")

    for spec in specs:
        if spec.error is None and spec.info is not None and spec.info.is_empty:
            # Zero-sample audio: success with empty text, no model needed.
            spec.empty = True
            spec.result = TranscribeResult(
                text="",
                language=None,
                language_source="forced" if spec.language else "auto",
                segments=[],
                timestamp_source=None,
                warnings=["empty audio"],
            )

    valid = [spec for spec in specs if spec.error is None]

    # ---- max_new_tokens -------------------------------------------------
    max_duration = max((spec.info.duration_s for spec in valid if spec.info), default=0.0)
    max_tokens = settings.max_new_tokens
    if max_tokens is None:
        max_tokens = max_new_tokens_for(max_duration)

    check_output_conflicts(specs)

    # ---- dry run --------------------------------------------------------
    if settings.dry_run:
        plan_outputs: List[Dict[str, Any]] = []
        if settings.out_dir or settings.output:
            for spec in specs:
                if spec.error is not None:
                    continue
                for fmt in spec.formats:
                    plan_outputs.append(
                        {"id": spec.id, "format": fmt,
                         "path": _display_output_path(settings, spec, fmt)}
                    )
        plan = {
            "inputs": [
                {
                    "id": spec.id,
                    "source": spec.item.source,
                    "duration_s": spec.info.duration_s if spec.info else None,
                    "origin": spec.item.origin,
                }
                for spec in specs
            ],
            "model": {
                "id": settings.model_id,
                "path": str(settings.model_path),
                "backend": settings.backend_name,
            },
            "aligner": {"path": str(settings.aligner_path)} if settings.aligner_path else None,
            "backend": settings.backend_name,
            "device": settings.device,
            "dtype": settings.dtype,
            "timestamps": any(spec.timestamps for spec in specs),
            "formats": settings.formats,
            "max_new_tokens": max_tokens,
            "outputs": plan_outputs,
            "errors": [_error_record(err) for err in spec_errors],
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ok": not spec_errors and all(spec.error is None for spec in specs),
            "command": "transcribe",
            "plan": plan,
        }
        if not payload["ok"]:
            payload["errors"] = [_error_record(err) for err in spec_errors] + [
                _spec_error_record(spec) for spec in specs if spec.error is not None
            ]
        return RunResult(
            stdout=json.dumps(payload, ensure_ascii=False),
            exit_code=EXIT_OK if payload["ok"] else _first_error_code(specs, spec_errors),
            payload=payload,
        )

    # ---- output dir / state / lock --------------------------------------
    lock: Optional[OutputLock] = None
    state_path: Optional[Path] = None
    latest: Dict[str, Dict[str, Any]] = {}
    display_out_dir = str(settings.out_dir) if settings.out_dir else None
    if settings.out_dir is not None:
        settings.out_dir.mkdir(parents=True, exist_ok=True)
        lock = OutputLock(settings.out_dir)
        lock.acquire()
        state_path = settings.out_dir / STATE_DIRNAME / STATE_FILENAME
        latest = read_state(state_path)
        touched = recover_running(state_path, [spec.id for spec in specs])
        if touched:
            log(f"recovered {len(touched)} interrupted job(s) to pending")
        _write_manifest(settings.out_dir / STATE_DIRNAME / MANIFEST_FILENAME, specs)

    try:
        return _run_locked(
            settings, specs, spec_errors, total_requested, max_tokens,
            log=log, state_path=state_path, latest=latest,
            display_out_dir=display_out_dir,
        )
    finally:
        if lock is not None:
            lock.release()


def _run_locked(
    settings: TranscribeSettings,
    specs: List[ItemSpec],
    spec_errors: List[Dict[str, Any]],
    total_requested: int,
    max_tokens: int,
    *,
    log: LogFn,
    state_path: Optional[Path],
    latest: Dict[str, Dict[str, Any]],
    display_out_dir: Optional[str],
) -> RunResult:
    valid = [spec for spec in specs if spec.error is None and spec.info is not None]

    # ---- job keys & resume ---------------------------------------------
    for spec in valid:
        request = {
            "language": spec.language,
            "context": spec.context,
            "timestamps": spec.timestamps,
            "granularity": spec.granularity,
            "formats": sorted(spec.formats),
            "max_new_tokens": max_tokens,
            "model": settings.model_id,
        }
        output_rel = ",".join(f"{spec.output_base}.{fmt}" for fmt in spec.formats)
        spec.job_key = compute_job_key(
            request,
            audio_sha256=spec.audio_sha256,
            model_fingerprint=settings.model_fingerprint,
            aligner_fingerprint=settings.aligner_fingerprint,
            backend=settings.backend_name,
            output_rel=output_rel,
        )

    backend = None
    load_ms = 0

    pending: List[ItemSpec] = []
    for spec in valid:
        record = latest.get(spec.id)
        if settings.out_dir is not None and should_skip(
            record, spec.job_key, resume=settings.resume, overwrite=settings.overwrite
        ):
            recorded = {o.get("format") for o in record.get("outputs", []) if isinstance(o, dict)}
            if recorded.issuperset(set(spec.formats)):
                spec.skipped = True
                spec.outputs = list(record.get("outputs", []))
                spec.payload = item_payload(
                    command="transcribe",
                    input_info=_input_info(spec),
                    model_info=_model_info(settings),
                    result={"text": None, "skipped": True},
                    metrics={},
                    outputs=spec.outputs,
                )
                spec.payload["skipped"] = True
                if state_path is not None:
                    append_jsonl(
                        state_path,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "id": spec.id,
                            "job_key": spec.job_key,
                            "status": STATUS_SKIPPED,
                            "outputs": spec.outputs,
                            "error": None,
                            "started_at": None,
                            "finished_at": now_iso(),
                        },
                    )
                log(f"skip {spec.id} (resume: unchanged)")
                continue
        if spec.empty:
            continue
        pending.append(spec)

    # Empty inputs are complete without inference (and never load the model).
    for spec in valid:
        if spec.empty and not spec.skipped:
            _complete_item(
                spec, spec.result, settings, log,
                load_ms=0, infer_ms=0,
                state_path=state_path,
                display_out_dir=display_out_dir,
            )

    if pending:
        try:
            backend = build_transformers_backend(
                settings.model_path,
                aligner_path=settings.aligner_path,
                device=settings.device,
                dtype=settings.dtype,
                attn_implementation=settings.attn_implementation,
                max_inference_batch_size=settings.max_inference_batch_size,
                max_new_tokens=max_tokens,
                offline=settings.offline,
            )
            ensure = getattr(backend, "ensure_loaded", None)
            started = time.perf_counter()
            if callable(ensure):
                ensure(need_align=any(spec.timestamps for spec in pending))
            load_ms = int((time.perf_counter() - started) * 1000)
            log(f"model loaded in {load_ms} ms")
        except QasrError:
            raise
        except Exception as exc:
            raise RuntimeFailure(f"failed to load model: {exc}") from exc

    if state_path is not None:
        started_at = now_iso()
        for spec in pending:
            spec.started_at = started_at
            append_jsonl(
                state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": spec.id,
                    "job_key": spec.job_key,
                    "status": "running",
                    "outputs": [],
                    "error": None,
                    "started_at": started_at,
                    "finished_at": None,
                },
            )

    first_item = True
    batch_size = max(1, settings.batch_size)
    for start in range(0, len(pending), batch_size):
        group = pending[start:start + batch_size]
        if settings.fail_fast and any(spec.error for spec in specs):
            break
        group_opts = TranscribeOptions(
            timestamps=any(spec.timestamps for spec in group),
            granularity=group[0].granularity,
            max_new_tokens=max_tokens,
        )
        _process_group(
            group, backend, group_opts, settings, log,
            load_ms=load_ms if first_item else 0,
            state_path=state_path,
            display_out_dir=display_out_dir,
        )
        first_item = False
        if settings.fail_fast and any(spec.error for spec in group):
            break

    # ---- collect -------------------------------------------------------
    for spec in specs:
        if spec.error is not None:
            spec.payload = item_error_payload(
                command="transcribe",
                input_info=_input_info(spec),
                error=spec.error.to_dict(),
                model_info=_model_info(settings),
            )
            if state_path is not None:
                append_jsonl(
                    state_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "id": spec.id,
                        "job_key": spec.job_key or None,
                        "status": STATUS_FAILED,
                        "outputs": [],
                        "error": spec.error.to_dict(),
                        "started_at": spec.started_at,
                        "finished_at": now_iso(),
                    },
                )

    # ---- render stdout --------------------------------------------------
    failures = [
        {"id": spec.id, "source": spec.item.source, "error": spec.error.to_dict()}
        for spec in specs
        if spec.error is not None
    ] + [_error_record(err) for err in spec_errors]
    skipped_count = sum(1 for spec in specs if spec.skipped)

    stdout = ""
    payload: Optional[Dict[str, Any]] = None
    if settings.stdout_mode == "jsonl":
        lines = [json.dumps(spec.payload, ensure_ascii=False) for spec in specs if spec.payload]
        for entry in spec_errors:
            record = _error_record(entry)
            lines.append(
                json.dumps(
                    item_error_payload(
                        command="transcribe",
                        input_info={"id": record["id"], "source": record.get("source")},
                        error=record["error"],
                    ),
                    ensure_ascii=False,
                )
            )
        stdout = "\n".join(lines)
    elif settings.stdout_mode == "json":
        if total_requested == 1 and not specs and len(spec_errors) == 1:
            # A single unusable input is a failure of that input: use the
            # standard error envelope instead of a batch with one error.
            payload = error_payload("transcribe", _error_record(spec_errors[0])["error"])
        elif total_requested == 1 and len(specs) == 1 and not spec_errors:
            payload = specs[0].payload
        else:
            payload = batch_payload(
                [spec.payload for spec in specs if spec.error is None and spec.payload],
                failures,
                total=total_requested, skipped=skipped_count,
            )
        stdout = json.dumps(payload, ensure_ascii=False)
    elif settings.stdout_mode == "text" and total_requested == 1 and len(specs) == 1:
        spec = specs[0]
        if spec.error is None and spec.result is not None:
            stdout = spec.result.text
    elif settings.stdout_mode == "subtitle" and total_requested == 1 and len(specs) == 1:
        spec = specs[0]
        if spec.error is None and spec.result is not None:
            cues = subtitle_cues(spec.result, spec.info.duration_s if spec.info else None)
            stdout = render_format(
                spec.formats[0],
                item=spec.payload or {},
                result=spec.result,
                audio_duration_s=spec.info.duration_s if spec.info else None,
                cues=cues,
            ).rstrip("\n")

    if settings.stdout_mode == "json" and payload is None and total_requested == 1 and spec_errors:
        payload = batch_payload([], spec_errors, total=1)

    # ---- exit code ------------------------------------------------------
    ok_count = sum(1 for spec in specs if spec.error is None and not spec.skipped)
    completed_count = ok_count + skipped_count
    failed_count = len(failures)
    if failed_count == 0:
        exit_code = EXIT_OK
    elif completed_count > 0:
        exit_code = EXIT_PARTIAL
    else:
        exit_code = _first_error_code(specs, spec_errors)

    if settings.stdout_mode == "none":
        if failed_count:
            log(f"done: {ok_count} ok, {failed_count} failed, {skipped_count} skipped")
        else:
            log(f"done: {ok_count} ok, {skipped_count} skipped")

    return RunResult(
        stdout=stdout,
        exit_code=exit_code,
        payload=payload,
        summary={"total": total_requested, "ok": ok_count,
                 "failed": failed_count, "skipped": skipped_count},
    )


def _process_group(
    group: List[ItemSpec],
    backend: Any,
    opts: TranscribeOptions,
    settings: TranscribeSettings,
    log: LogFn,
    *,
    load_ms: int,
    state_path: Optional[Path],
    display_out_dir: Optional[str],
) -> None:
    jobs = [
        TranscribeJob(
            item=spec.item,
            info=spec.info,
            language=spec.language,
            context=spec.context,
        )
        for spec in group
    ]
    started = time.perf_counter()
    try:
        results = backend.transcribe(jobs, opts)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        per_item_ms = elapsed_ms // max(1, len(group))
        for index, (spec, result) in enumerate(zip(group, results)):
            _complete_item(
                spec, result, settings, log,
                load_ms=load_ms if index == 0 else 0,
                infer_ms=per_item_ms,
                state_path=state_path,
                display_out_dir=display_out_dir,
            )
    except Exception as exc:  # noqa: BLE001 - fall back to one-by-one
        if len(group) == 1:
            _fail_item(group[0], exc, settings, log, state_path=state_path,
                       display_out_dir=display_out_dir)
            return
        log(f"batch of {len(group)} failed ({exc}); retrying item by item")
        for spec in group:
            single_started = time.perf_counter()
            try:
                results = backend.transcribe(
                    [TranscribeJob(item=spec.item, info=spec.info,
                                   language=spec.language, context=spec.context)],
                    TranscribeOptions(timestamps=spec.timestamps,
                                      granularity=spec.granularity,
                                      max_new_tokens=opts.max_new_tokens),
                )
                elapsed = int((time.perf_counter() - single_started) * 1000)
                _complete_item(spec, results[0], settings, log, load_ms=0,
                               infer_ms=elapsed, state_path=state_path,
                               display_out_dir=display_out_dir)
            except Exception as inner:  # noqa: BLE001
                _fail_item(spec, inner, settings, log, state_path=state_path,
                           display_out_dir=display_out_dir)


def _complete_item(
    spec: ItemSpec,
    result: Any,
    settings: TranscribeSettings,
    log: LogFn,
    *,
    load_ms: int,
    infer_ms: int,
    state_path: Optional[Path],
    display_out_dir: Optional[str],
) -> None:
    duration = spec.info.duration_s if spec.info else 0.0
    wall_ms = load_ms + infer_ms
    rtf = (infer_ms / (duration * 1000)) if duration > 0 else None
    spec.result = result
    spec.metrics = {
        "load_ms": load_ms,
        "infer_ms": infer_ms,
        "wall_ms": wall_ms,
        "rtf": round(rtf, 4) if rtf is not None else None,
    }

    outputs: List[Dict[str, Any]] = []
    try:
        outputs = _write_outputs(spec, settings, display_out_dir)
    except QasrError as exc:
        spec.error = exc
        spec.result = None
        return

    spec.outputs = outputs
    spec.payload = item_payload(
        command="transcribe",
        input_info=_input_info(spec),
        model_info=_model_info(settings),
        result=result_payload(result),
        metrics=spec.metrics,
        outputs=outputs,
    )
    if state_path is not None:
        append_jsonl(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "id": spec.id,
                "job_key": spec.job_key,
                "status": STATUS_DONE,
                "outputs": outputs,
                "error": None,
                "started_at": spec.started_at,
                "finished_at": now_iso(),
            },
        )
    log(f"ok {spec.id} ({duration:.2f}s, rtf {rtf:.3f})" if rtf is not None else f"ok {spec.id}")


def _fail_item(
    spec: ItemSpec,
    exc: Exception,
    settings: TranscribeSettings,
    log: LogFn,
    *,
    state_path: Optional[Path],
    display_out_dir: Optional[str],
) -> None:
    if isinstance(exc, QasrError):
        error = exc
    else:
        error = RuntimeFailure(f"transcription failed for {spec.id}: {exc}")
    spec.error = error
    log(f"error {spec.id}: {error.message}")


def _write_outputs(
    spec: ItemSpec,
    settings: TranscribeSettings,
    display_out_dir: Optional[str],
) -> List[Dict[str, Any]]:
    if spec.result is None:
        return []
    if not settings.out_dir and not (settings.output and settings.output != "-"):
        return []

    cues = subtitle_cues(spec.result, spec.info.duration_s if spec.info else None)
    outputs: List[Dict[str, Any]] = []

    if settings.output and settings.output != "-":
        targets = [(spec.formats[0], Path(settings.output).expanduser())]
    else:
        targets = []
        assert settings.out_dir is not None
        for fmt in spec.formats:
            rel = f"{spec.output_base}.{fmt}"
            targets.append((fmt, safe_output_path(settings.out_dir, rel)))

    allow_overwrite = settings.overwrite or settings.resume
    for fmt, target in targets:
        if fmt in SUBTITLE_FORMATS and not spec.result.segments and not spec.empty:
            raise CapabilityError(
                f"cannot write {fmt} without timestamps",
                hint="pass --timestamps --language <alignable language>, or run 'qasr align'",
            )
        if target.exists() and not allow_overwrite:
            raise InputError(
                f"output already exists: {target}",
                hint="pass --overwrite to replace it, or --resume to skip unchanged jobs",
            )
        item = spec.payload or item_payload(
            command="transcribe",
            input_info=_input_info(spec),
            model_info=_model_info(settings),
            result=result_payload(spec.result),
            metrics=spec.metrics,
            outputs=[],
        )
        text = render_format(
            fmt, item=item, result=spec.result,
            audio_duration_s=spec.info.duration_s if spec.info else None,
            cues=cues,
        )
        data = text.encode("utf-8")
        atomic_write_bytes(target, data)
        display = _display_path(display_out_dir, target, settings)
        outputs.append({"format": fmt, "path": display, "sha256": sha256_bytes(data)})
    return outputs


def _display_path(display_out_dir: Optional[str], target: Path, settings: TranscribeSettings) -> str:
    if settings.output and settings.output != "-":
        return str(Path(settings.output))
    if display_out_dir:
        try:
            return str(Path(display_out_dir) / target.relative_to(Path(settings.out_dir)))
        except (ValueError, TypeError):
            return str(target)
    return str(target)


def _display_output_path(settings: TranscribeSettings, spec: ItemSpec, fmt: str) -> str:
    if settings.output and settings.output != "-":
        return settings.output
    if settings.out_dir is not None:
        return str(Path(str(settings.out_dir)) / f"{spec.output_base}.{fmt}")
    return f"{spec.output_base}.{fmt}"


def _write_manifest(path: Path, specs: Sequence[ItemSpec]) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "items": [
            {
                "id": spec.id,
                "source": spec.item.source,
                "language": spec.language,
                "context": spec.context,
                "timestamps": spec.timestamps,
                "granularity": spec.granularity,
                "formats": list(spec.formats),
                "output": spec.output_base,
            }
            for spec in specs
        ],
    }
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _input_info(spec: ItemSpec) -> Dict[str, Any]:
    return {
        "id": spec.id,
        "source": spec.item.source,
        "duration_s": spec.info.duration_s if spec.info else None,
        "sha256": spec.audio_sha256 or None,
    }


def _model_info(settings: TranscribeSettings) -> Dict[str, Any]:
    return {
        "id": settings.model_id,
        "backend": settings.backend_name,
        "device": settings.device,
        "dtype": settings.dtype,
        "fingerprint": settings.model_fingerprint,
    }


def _error_record(entry: Dict[str, Any]) -> Dict[str, Any]:
    error = entry.get("error")
    if isinstance(error, QasrError):
        error_dict = error.to_dict()
    elif isinstance(error, dict):
        error_dict = error
    else:
        error_dict = RuntimeFailure(str(error)).to_dict()
    return {"id": entry.get("id") or entry.get("source"), "source": entry.get("source"),
            "error": error_dict}


def _spec_error_record(spec: ItemSpec) -> Dict[str, Any]:
    error = spec.error or RuntimeFailure("unknown error")
    return {"id": spec.id, "source": spec.item.source, "error": error.to_dict()}


def _first_error_code(specs: Sequence[ItemSpec], spec_errors: Sequence[Dict[str, Any]]) -> int:
    for spec in specs:
        if spec.error is not None:
            return spec.error.exit_code
    for entry in spec_errors:
        error = entry.get("error")
        if isinstance(error, QasrError):
            return error.exit_code
        if isinstance(error, dict) and isinstance(error.get("code"), int):
            return int(error["code"])
    return EXIT_INPUT
