"""Transcription orchestration: inputs -> micro-batches -> results -> outputs."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import SCHEMA_VERSION, __version__, long_audio
from .audio import AudioInfo, max_new_tokens_for, probe, read_window_16k, write_wav_16k
from .backends import (
    AlignJob,
    AlignOptions,
    Segment,
    TranscribeJob,
    TranscribeOptions,
    TranscribeResult,
    _is_cjk_char,
    aggregate_sentences,
    build_transformers_backend,
    build_vllm_backend,
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
    canonical_json,
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

# Bump when chunk records gain new semantics that older caches must not mask
# (e.g. windows recovered from their own anchor when the reference is short).
CHUNK_LOGIC_VERSION = 3


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
    attempts: int = 1

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
    backend_fallback: bool = False
    gpu_memory_utilization: Optional[float] = None
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
    skip_existing: bool = False
    retry_failed: bool = False
    offline: bool = False
    max_chunk_seconds: float = 270.0
    jobs: int = 1
    remote_url: Optional[str] = None
    remote_timeout: float = 300.0
    remote_retries: int = 2


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


def _display_id(spec: ItemSpec) -> str:
    """The identifier as the user wrote it (sanitized only for filesystem use)."""
    return spec.item.raw_id or spec.id


def check_output_conflicts(specs: Sequence[ItemSpec]) -> None:
    """Reject duplicate *final* output paths, even when ids differ.

    Two raw ids such as ``a/b`` and ``a_b`` sanitize to the same file name;
    comparing ids would silently overwrite one result with the other. Compare
    the actual relative output path instead and report the raw ids.
    """
    seen: Dict[str, ItemSpec] = {}
    for spec in specs:
        if spec.error is not None:
            continue
        for fmt in spec.formats:
            rel = f"{spec.output_base}.{fmt}"
            other = seen.get(rel)
            if other is not None and other is not spec:
                raise UsageError(
                    f"two inputs map to the same output file: {rel}",
                    hint="use --manifest with explicit 'output' names",
                    details={
                        "ids": [_display_id(other), _display_id(spec)],
                        "sources": [other.item.source, spec.item.source],
                    },
                )
            seen[rel] = spec


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

    # Only file outputs can overwrite each other; stdout streams cannot.
    if settings.out_dir is not None or (settings.output and settings.output != "-"):
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
                    "windows": _planned_window_count(spec, settings),
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
            "backend_fallback": settings.backend_fallback,
            "device": settings.device,
            "dtype": settings.dtype,
            "timestamps": any(spec.timestamps for spec in specs),
            "formats": settings.formats,
            "max_new_tokens": max_tokens,
            "max_chunk_seconds": settings.max_chunk_seconds,
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
            # Chunk attribution/fallback is code semantics, not an input:
            # without it, --resume would keep outputs produced by an older
            # implementation even though the logic changed.
            "chunk_logic_version": CHUNK_LOGIC_VERSION,
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
        if settings.skip_existing and _outputs_complete(spec, settings):
            spec.skipped = True
            spec.payload = item_payload(
                command="transcribe",
                input_info=_input_info(spec),
                model_info=_model_info(settings),
                result={"text": None, "skipped": True},
                metrics={},
                outputs=[
                    {"format": fmt, "path": str(path)}
                    for fmt, path in zip(spec.formats, _target_paths(spec, settings))
                ],
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
                        "outputs": spec.payload["outputs"],
                        "error": None,
                        "started_at": None,
                        "finished_at": now_iso(),
                    },
                )
            continue
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

    # With --fail-fast, an already-failed input means nothing else will run:
    # do not spend minutes loading weights just to break out immediately.
    stop_early = settings.fail_fast and any(spec.error is not None for spec in specs)

    if pending and not stop_early:
        started = time.perf_counter()
        if settings.backend_name == "remote":
            from .remote import RemoteBackend, RemoteOptions

            backend = RemoteBackend(
                RemoteOptions(
                    base_url=str(settings.remote_url or ""),
                    model=settings.model_id,
                    timeout=settings.remote_timeout,
                    retries=settings.remote_retries,
                    log=log,
                )
            )
            ensure = getattr(backend, "ensure_loaded", None)
            if callable(ensure):
                ensure(need_align=any(spec.timestamps and not spec.empty for spec in pending))
            prepare = getattr(backend, "prepare", None)
            if callable(prepare):
                prepare()
            resolved = getattr(backend, "resolved_model_id", None)
            if resolved and not settings.model_id:
                settings.model_id = str(resolved)
            load_ms = int((time.perf_counter() - started) * 1000)
            log(f"remote backend ready ({load_ms} ms)")
        else:
            try:
                if settings.backend_name == "vllm":
                    backend = build_vllm_backend(
                        settings.model_path,
                        aligner_path=settings.aligner_path,
                        device=settings.device,
                        dtype=settings.dtype,
                        max_inference_batch_size=settings.max_inference_batch_size,
                        max_new_tokens=max_tokens,
                        gpu_memory_utilization=settings.gpu_memory_utilization,
                        offline=settings.offline,
                    )
                else:
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
                if callable(ensure):
                    ensure(need_align=any(spec.timestamps and not spec.empty for spec in pending))
                load_ms = int((time.perf_counter() - started) * 1000)
                log(f"model loaded in {load_ms} ms")
            except QasrError:
                raise
            except Exception as exc:
                label = "vLLM model" if settings.backend_name == "vllm" else "model"
                raise RuntimeFailure(f"failed to load {label}: {exc}") from exc

    if state_path is not None and not stop_early:
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

    if settings.backend_name == "remote":
        _run_remote_items(
            pending, backend, settings, log,
            max_tokens=max_tokens,
            state_path=state_path,
            display_out_dir=display_out_dir,
        )
    else:
        first_item = True
        batch_size = max(1, settings.batch_size)
        index = 0
        while index < len(pending):
            if settings.fail_fast and any(spec.error for spec in specs):
                break
            spec = pending[index]
            if _is_long(spec, settings):
                _process_long_item(
                    spec, backend, settings, log,
                    load_ms=load_ms if first_item else 0,
                    max_tokens=max_tokens,
                    state_path=state_path,
                    display_out_dir=display_out_dir,
                )
                first_item = False
                index += 1
                if settings.fail_fast and spec.error is not None:
                    break
                continue
            group = []
            while (
                index < len(pending)
                and len(group) < batch_size
                and not _is_long(pending[index], settings)
            ):
                group.append(pending[index])
                index += 1
            group_opts = TranscribeOptions(
                timestamps=any(item.timestamps for item in group),
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
            if settings.fail_fast and any(item.error for item in group):
                break

    # Fail-fast leaves later items unattempted: give them an explicit terminal
    # state instead of leaving a stale ``running`` record behind.
    if settings.fail_fast:
        for spec in specs:
            if spec.skipped or spec.error is not None or spec.result is not None:
                continue
            spec.skipped = True
            spec.started_at = None
            spec.payload = item_payload(
                command="transcribe",
                input_info=_input_info(spec),
                model_info=_model_info(settings),
                result={"text": None, "skipped": True},
                metrics={},
                outputs=[],
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
                        "outputs": [],
                        "error": None,
                        "started_at": None,
                        "finished_at": now_iso(),
                    },
                )

    # ---- collect -------------------------------------------------------
    for spec in specs:
        if spec.error is not None:
            spec.payload = item_error_payload(
                command="transcribe",
                input_info=_input_info(spec),
                error=spec.error.to_dict(),
                model_info=_model_info(settings),
            )
            spec.payload["attempts"] = spec.attempts
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
                        "attempts": spec.attempts,
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

    if settings.backend_fallback and isinstance(payload, dict):
        payload["backend_fallback"] = True

    # ---- exit code ------------------------------------------------------
    ok_count = sum(1 for spec in specs if spec.error is None and not spec.skipped)
    completed_count = ok_count + skipped_count
    failed_count = len(failures)
    if failed_count == 0:
        exit_code = EXIT_OK
    elif settings.fail_fast:
        # Fail-fast aborts on the first failure: report that error's code, not
        # the aggregate partial-failure code (unattempted items are skipped).
        exit_code = _first_error_code(specs, spec_errors)
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


def _attempt_item(spec: ItemSpec, settings: TranscribeSettings, fn: Any) -> Any:
    """Run ``fn`` once, retrying a retryable failure a single time."""
    started = time.perf_counter()
    try:
        return fn(), None, int((time.perf_counter() - started) * 1000)
    except Exception as exc:  # noqa: BLE001 - translated by the caller
        if (
            settings.retry_failed
            and spec.attempts < 2
            and isinstance(exc, QasrError)
            and exc.retryable
        ):
            spec.attempts += 1
            started = time.perf_counter()
            try:
                return fn(), None, int((time.perf_counter() - started) * 1000)
            except Exception as retried:  # noqa: BLE001
                return None, retried, int((time.perf_counter() - started) * 1000)
        return None, exc, int((time.perf_counter() - started) * 1000)


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
    def run_group() -> Any:
        return backend.transcribe(jobs, opts)

    if len(group) == 1:
        results, exc, elapsed_ms = _attempt_item(group[0], settings, run_group)
        if exc is None:
            result = results[0]
        if exc is not None:
            _fail_item(group[0], exc, settings, log, state_path=state_path,
                       display_out_dir=display_out_dir)
            return
        _complete_item(
            group[0], result, settings, log,
            load_ms=load_ms, infer_ms=elapsed_ms,
            state_path=state_path, display_out_dir=display_out_dir,
        )
        return

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
        log(f"batch of {len(group)} failed ({exc}); retrying item by item")
        for spec in group:
            def run_single(spec: ItemSpec = spec) -> Any:
                return backend.transcribe(
                    [TranscribeJob(item=spec.item, info=spec.info,
                                   language=spec.language, context=spec.context)],
                    TranscribeOptions(timestamps=spec.timestamps,
                                      granularity=spec.granularity,
                                      max_new_tokens=opts.max_new_tokens),
                )

            results, inner, elapsed = _attempt_item(spec, settings, run_single)
            if inner is None:
                result = results[0]
            if inner is not None:
                _fail_item(spec, inner, settings, log, state_path=state_path,
                           display_out_dir=display_out_dir)
                continue
            _complete_item(spec, result, settings, log, load_ms=0,
                           infer_ms=elapsed, state_path=state_path,
                           display_out_dir=display_out_dir)


def _is_long(spec: ItemSpec, settings: TranscribeSettings) -> bool:
    return bool(
        spec.info is not None
        and not spec.info.is_empty
        and settings.max_chunk_seconds
        and spec.info.duration_s > settings.max_chunk_seconds + 1e-6
    )


def _planned_window_count(spec: ItemSpec, settings: TranscribeSettings) -> int:
    if spec.info is None or spec.info.is_empty:
        return 0
    if not _is_long(spec, settings):
        return 1
    return len(
        long_audio.plan_windows(
            spec.info.frames,
            spec.info.sample_rate,
            window_s=settings.max_chunk_seconds,
            max_window_s=max(180.0, settings.max_chunk_seconds),
        )
    )


def _dedupe(messages: Sequence[str]) -> List[str]:
    out: List[str] = []
    for message in messages:
        if message and message not in out:
            out.append(message)
    return out


def _transcribe_spec(
    backend: Any,
    spec: ItemSpec,
    settings: TranscribeSettings,
    max_tokens: int,
) -> TranscribeResult:
    if _is_long(spec, settings):
        return _transcribe_long(backend, spec, settings, max_tokens)
    job = TranscribeJob(
        item=spec.item,
        info=spec.info,
        language=spec.language,
        context=spec.context,
    )
    results = backend.transcribe(
        [job],
        TranscribeOptions(
            timestamps=spec.timestamps,
            granularity=spec.granularity,
            max_new_tokens=max_tokens,
        ),
    )
    return results[0]


def _process_long_item(
    spec: ItemSpec,
    backend: Any,
    settings: TranscribeSettings,
    log: LogFn,
    *,
    load_ms: int,
    max_tokens: int,
    state_path: Optional[Path],
    display_out_dir: Optional[str],
    state_lock: Optional[Any] = None,
) -> None:
    def run_long() -> Any:
        return _transcribe_long(backend, spec, settings, max_tokens)

    result, exc, elapsed = _attempt_item(spec, settings, run_long)
    if exc is not None:
        _fail_item(spec, exc, settings, log, state_path=state_path,
                   display_out_dir=display_out_dir)
        return
    _complete_item(
        spec, result, settings, log,
        load_ms=load_ms,
        infer_ms=elapsed,
        state_path=state_path,
        display_out_dir=display_out_dir,
        state_lock=state_lock,
    )


def _chunk_dir(settings: TranscribeSettings, spec: ItemSpec) -> Optional[Path]:
    if settings.out_dir is None or not spec.job_key:
        return None
    return settings.out_dir / STATE_DIRNAME / "chunks" / spec.job_key


def _chunk_identity(
    spec: ItemSpec, window: Any, settings: TranscribeSettings
) -> str:
    payload = {
        "job_key": spec.job_key,
        "index": window.index,
        "audio_samples": [window.audio_start_sample, window.audio_end_sample],
        "core_samples": [window.core_start_sample, window.core_end_sample],
        "max_chunk_seconds": settings.max_chunk_seconds,
        "language": spec.language,
        "context": spec.context,
        "model": settings.model_id,
        "model_fingerprint": settings.model_fingerprint,
        "aligner_fingerprint": settings.aligner_fingerprint,
        "backend": settings.backend_name,
        "qasr_version": __version__,
        "chunk_logic_version": CHUNK_LOGIC_VERSION,
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _read_chunk(
    chunk_dir: Optional[Path], index: int, identity: str, *, resume: bool
) -> Optional[Dict[str, Any]]:
    if chunk_dir is None or not resume:
        return None
    path = chunk_dir / f"{index}.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("chunk_key") != identity:
        return None
    return record


def _write_chunk(
    chunk_dir: Optional[Path], index: int, identity: str, record: Dict[str, Any]
) -> None:
    if chunk_dir is None:
        return
    record = {**record, "chunk_key": identity, "index": index}
    atomic_write_text(
        chunk_dir / f"{index}.json",
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _window_input(
    spec: ItemSpec, window: Any, sr: int, tmp_dir: Path
) -> Optional[tuple[InputItem, AudioInfo]]:
    if spec.item.path is None:
        return None
    samples = read_window_16k(
        spec.item.path, window.audio_start_sample, window.audio_end_sample
    )
    if len(samples) == 0:
        return None
    path = write_wav_16k(tmp_dir / f"window-{window.index}.wav", samples)
    duration = len(samples) / 16000.0
    item = InputItem(
        id=f"{spec.id}#{window.index}",
        source=str(path),
        path=path,
        origin="file",
    )
    info = AudioInfo(
        path=path,
        duration_s=duration,
        sample_rate=16000,
        channels=1,
        frames=int(len(samples)),
    )
    return item, info


def _join_window_text(previous: str, current: str) -> str:
    """Join overlapping window text without repeating an exact suffix/prefix."""
    left = previous.rstrip()
    right = current.lstrip()
    if not left:
        return right
    if not right:
        return left
    limit = min(len(left), len(right), 400)
    overlap = 0
    for size in range(limit, 1, -1):
        if left[-size:] == right[:size]:
            overlap = size
            break
    if overlap:
        return left + right[overlap:]
    if _is_cjk_char(left[-1]) or _is_cjk_char(right[0]):
        return left + right
    return left + " " + right


def _transcribe_long(
    backend: Any,
    spec: ItemSpec,
    settings: TranscribeSettings,
    max_tokens: int,
) -> TranscribeResult:
    assert spec.info is not None
    info = spec.info
    windows = long_audio.plan_windows(
        info.frames,
        info.sample_rate,
        window_s=settings.max_chunk_seconds,
        max_window_s=max(180.0, settings.max_chunk_seconds),
    )
    chunk_dir = _chunk_dir(settings, spec)
    if chunk_dir is not None and settings.overwrite and chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    with tempfile.TemporaryDirectory(prefix="qasr-chunks-") as raw_tmp:
        tmp_dir = Path(raw_tmp)
        if spec.timestamps:
            return _long_timestamps(
                backend, spec, settings, windows, chunk_dir, tmp_dir, max_tokens
            )
        return _long_plain(
            backend, spec, settings, windows, chunk_dir, tmp_dir, max_tokens
        )


def _long_plain(
    backend: Any,
    spec: ItemSpec,
    settings: TranscribeSettings,
    windows: Sequence[Any],
    chunk_dir: Optional[Path],
    tmp_dir: Path,
    max_tokens: int,
) -> TranscribeResult:
    assert spec.info is not None
    texts: List[str] = []
    warnings: List[str] = []
    language: Optional[str] = None
    language_source = "forced" if spec.language else "auto"
    for window in windows:
        identity = _chunk_identity(spec, window, settings)
        record = _read_chunk(chunk_dir, window.index, identity, resume=settings.resume)
        if record is None:
            prepared = _window_input(spec, window, spec.info.sample_rate, tmp_dir)
            if prepared is None:
                continue
            item, window_info = prepared
            window_tokens = (
                settings.max_new_tokens
                if settings.max_new_tokens is not None
                else max_new_tokens_for(window_info.duration_s)
            )
            result = backend.transcribe(
                [TranscribeJob(item=item, info=window_info,
                               language=spec.language, context=spec.context)],
                TranscribeOptions(
                    timestamps=False,
                    granularity="sentence",
                    max_new_tokens=window_tokens,
                ),
            )[0]
            record = {
                "text": result.text,
                "language": result.language,
                "language_source": result.language_source,
                "warnings": list(result.warnings),
            }
            _write_chunk(chunk_dir, window.index, identity, record)
        warnings.extend(record.get("warnings") or [])
        if language is None and record.get("language"):
            language = record["language"]
            language_source = record.get("language_source") or language_source
        text = str(record.get("text") or "")
        if text.strip():
            texts.append(text)

    text = ""
    for piece in texts:
        text = _join_window_text(text, piece)
    if len(windows) > 1:
        warnings.append("chunk_boundary_text_uncertain")
    return TranscribeResult(
        text=text,
        language=language,
        language_source=language_source,
        segments=[],
        timestamp_source=None,
        warnings=_dedupe(warnings),
    )


def _long_timestamps(
    backend: Any,
    spec: ItemSpec,
    settings: TranscribeSettings,
    windows: Sequence[Any],
    chunk_dir: Optional[Path],
    tmp_dir: Path,
    max_tokens: int,
) -> TranscribeResult:
    assert spec.info is not None
    sr = spec.info.sample_rate
    canonical = backend.transcribe(
        [TranscribeJob(item=spec.item, info=spec.info,
                       language=spec.language, context=spec.context)],
        TranscribeOptions(timestamps=False, granularity="sentence",
                          max_new_tokens=max_tokens),
    )[0]
    if not canonical.text.strip():
        return canonical

    warnings: List[str] = list(canonical.warnings)
    language = spec.language or canonical.language or ""
    records: Dict[int, Dict[str, Any]] = {}
    anchors: List[str] = []
    window_inputs: Dict[int, tuple[InputItem, AudioInfo]] = {}
    for window in windows:
        identity = _chunk_identity(spec, window, settings)
        record = _read_chunk(chunk_dir, window.index, identity, resume=settings.resume)
        if record is None:
            prepared = _window_input(spec, window, sr, tmp_dir)
            if prepared is None:
                anchors.append("")
                continue
            item, window_info = prepared
            window_inputs[window.index] = prepared
            window_tokens = (
                settings.max_new_tokens
                if settings.max_new_tokens is not None
                else max_new_tokens_for(window_info.duration_s)
            )
            anchor = backend.transcribe(
                [TranscribeJob(item=item, info=window_info,
                               language=spec.language, context=spec.context)],
                TranscribeOptions(timestamps=False, granularity="sentence",
                                  max_new_tokens=window_tokens),
            )[0]
            record = {"anchor_text": anchor.text, "warnings": list(anchor.warnings)}
            records[window.index] = record
        anchors.append(str(record.get("anchor_text") or ""))

    spans = long_audio.attribute_reference(canonical.text, anchors)
    # The whole-file reference pass can be shorter than the audio (model or
    # token-budget truncation), leaving later windows with an empty span. The
    # window's own anchor is already a transcript of that exact window, so
    # fall back to aligning it instead of silently dropping the speech.
    effective_texts: List[str] = []
    fallback_indexes: List[int] = []
    for window, span in zip(windows, spans):
        text = span.text
        # A weak span covers so little of the window that aligning it would
        # leave most of the window's speech out; a span whose tail is missing
        # can still be >= 50% overall while dropping the window's ending.
        # Prefer the window's own ASR anchor in both cases, not just when the
        # span is empty.
        if (span.is_incomplete or not text.strip()) and anchors[window.index].strip():
            text = anchors[window.index]
            fallback_indexes.append(window.index)
        effective_texts.append(text)
    if fallback_indexes:
        warnings.extend(
            f"chunk_anchor_fallback:{index}" for index in fallback_indexes
        )
        warnings.append(f"chunk_anchor_fallback_count:{len(fallback_indexes)}")

    items: List[Any] = []
    for position, (window, span) in enumerate(zip(windows, spans)):
        identity = _chunk_identity(spec, window, settings)
        record = _read_chunk(chunk_dir, window.index, identity, resume=settings.resume)
        if record is not None and record.get("items") is not None:
            for raw_item in record["items"]:
                items.append(Segment(**raw_item))
            warnings.extend(record.get("warnings") or [])
            continue
        if span.is_weak:
            warnings.append(
                f"chunk_reference_match_low:{window.index}:{span.match_ratio:.2f}"
            )
        align_text = effective_texts[position]
        if not align_text.strip():
            _write_chunk(chunk_dir, window.index, identity,
                         {"anchor_text": anchors[window.index],
                          "items": [], "warnings": []})
            continue
        prepared = window_inputs.get(window.index)
        if prepared is None:
            prepared = _window_input(spec, window, sr, tmp_dir)
        if prepared is None:
            continue
        item, window_info = prepared
        core_start_rel = (window.core_start_sample - window.audio_start_sample) / sr
        core_end_rel = (window.core_end_sample - window.audio_start_sample) / sr
        offset = window.audio_start_sample / sr
        local: List[Segment] = []
        window_warnings: List[str] = []
        try:
            aligned = backend.align(
                [AlignJob(item=item, info=window_info, text=align_text,
                          language=language)],
                AlignOptions(granularity="word"),
            )[0]
        except QasrError as exc:
            window_warnings.append(
                f"chunk_align_failed:{window.index}:{exc.message}"
            )
            aligned = None
        if aligned is not None:
            for seg in aligned.items:
                midpoint = (seg.start + seg.end) / 2.0
                if not (core_start_rel <= midpoint < core_end_rel):
                    continue
                local.append(
                    Segment(
                        text=seg.text,
                        start=seg.start + offset,
                        end=seg.end + offset,
                        kind=seg.kind,
                    )
                )
            warnings.extend(aligned.warnings)
        items.extend(local)
        _write_chunk(
            chunk_dir,
            window.index,
            identity,
            {
                "anchor_text": anchors[window.index],
                "items": [seg.to_dict() for seg in local],
                "warnings": window_warnings,
            },
        )
        warnings.extend(window_warnings)

    if fallback_indexes:
        # canonical.text stops where the reference pass stopped; rebuild from
        # the per-window texts so the output text matches the recovered tail.
        text = ""
        for piece in effective_texts:
            text = _join_window_text(text, piece)
    else:
        text = canonical.text

    merged = long_audio.merge_items(items, duration_s=spec.info.duration_s)
    segments = (
        aggregate_sentences(merged) if spec.granularity == "sentence" else merged
    )
    return TranscribeResult(
        text=text,
        language=canonical.language,
        language_source=canonical.language_source,
        segments=segments,
        timestamp_source="forced_aligner" if merged else None,
        warnings=_dedupe(warnings),
        raw_segments=merged,
    )


def _run_remote_items(
    pending: List[ItemSpec],
    backend: Any,
    settings: TranscribeSettings,
    log: LogFn,
    *,
    max_tokens: int,
    state_path: Optional[Path],
    display_out_dir: Optional[str],
) -> None:
    """Run remote jobs concurrently while keeping state appends serialized.

    Results are reported in input order by the collection phase; the first
    error is likewise taken in input order. ``--fail-fast`` cancels queued
    futures, and the finalization pass marks anything unattempted as skipped so
    no ``running`` record is left behind.
    """
    import threading
    from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed

    state_lock = threading.Lock()
    workers = max(1, min(int(settings.jobs or 1), len(pending)))

    def work(spec: ItemSpec) -> Any:
        result, exc, elapsed = _attempt_item(
            spec, settings,
            lambda: _transcribe_spec(backend, spec, settings, max_tokens),
        )
        return spec, result, exc, elapsed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, spec): spec for spec in pending}
        for future in as_completed(futures):
            try:
                spec, result, exc, elapsed_ms = future.result()
            except CancelledError:
                continue
            if exc is not None:
                _fail_item(spec, exc, settings, log, state_path=state_path,
                           display_out_dir=display_out_dir)
                if settings.fail_fast:
                    for other in futures:
                        if other is not future:
                            other.cancel()
                continue
            _complete_item(
                spec, result, settings, log,
                load_ms=0,
                infer_ms=elapsed_ms,
                state_path=state_path,
                display_out_dir=display_out_dir,
                state_lock=state_lock,
            )


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
    state_lock: Optional[Any] = None,
) -> None:
    duration = spec.info.duration_s if spec.info else 0.0
    wall_ms = load_ms + infer_ms
    rtf = (infer_ms / (duration * 1000)) if duration > 0 else None
    if settings.backend_fallback and result is not None and "vllm unavailable" not in " ".join(
        getattr(result, "warnings", [])
    ):
        result.warnings.append(
            "vllm unavailable on this host; fell back to transformers"
        )
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
    if settings.backend_fallback:
        spec.payload["backend_fallback"] = True
    spec.payload["attempts"] = spec.attempts
    if state_path is not None:
        if state_lock is None:
            _append_done(state_path, spec, outputs)
        else:
            with state_lock:
                _append_done(state_path, spec, outputs)
    log(f"ok {spec.id} ({duration:.2f}s, rtf {rtf:.3f})" if rtf is not None else f"ok {spec.id}")


def _append_done(state_path: Path, spec: ItemSpec, outputs: List[Dict[str, Any]]) -> None:
    append_jsonl(
        state_path,
        {
            "schema_version": SCHEMA_VERSION,
            "id": spec.id,
            "job_key": spec.job_key,
            "status": STATUS_DONE,
            "outputs": outputs,
            "error": None,
            "attempts": spec.attempts,
            "started_at": spec.started_at,
            "finished_at": now_iso(),
        },
    )


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


def _target_paths(spec: ItemSpec, settings: TranscribeSettings) -> List[Path]:
    if settings.output and settings.output != "-":
        return [Path(settings.output).expanduser()]
    if settings.out_dir is None:
        return []
    return [
        safe_output_path(settings.out_dir, f"{spec.output_base}.{fmt}")
        for fmt in spec.formats
    ]


def _outputs_complete(spec: ItemSpec, settings: TranscribeSettings) -> bool:
    """True when every requested output exists and is non-empty."""
    paths = _target_paths(spec, settings)
    if not paths:
        return False
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def outputs_complete(spec: ItemSpec, settings: TranscribeSettings) -> bool:
    """Public check used by the command layer to skip model resolution."""
    return _outputs_complete(spec, settings)


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
