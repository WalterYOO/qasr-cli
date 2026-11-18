"""Backend protocol and the transformers implementation.

Third-party imports are lazy. ``cli.py`` and other pure modules import this
module only for type definitions, so the import itself stays torch-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .audio import AudioInfo
from .errors import CapabilityError, ModelError, RuntimeFailure
from .inputs import InputItem
from .runtime import (
    configure_offline,
    configure_vllm_logging,
    guard_stdout_at_exit,
    import_qwen_asr,
    import_torch,
    redirect_vllm_logging,
    require_vllm,
    resolve_device,
    resolve_dtype,
    set_cuda_visible_device,
    stdout_to_stderr,
    torch_dtype,
)

TEXT_TRANSFORM = "qwen-asr-repetition-fix-v1"

_LANG_TAG_RE = re.compile(r"^\s*language\s+([A-Za-z][A-Za-z ()]*?)\s*\n", re.IGNORECASE)
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x3040, 0x30FF),  # kana
    (0xAC00, 0xD7AF),  # hangul
)


@dataclass
class Capabilities:
    name: str
    batch: bool = True
    context: bool = True
    timestamps: bool = True
    align: bool = True
    streaming: bool = False


@dataclass
class TranscribeJob:
    item: InputItem
    info: AudioInfo
    language: Optional[str] = None  # canonical name; None = auto/LID
    context: str = ""


@dataclass
class AlignJob:
    item: InputItem
    info: AudioInfo
    text: str = ""
    language: str = ""


@dataclass
class TranscribeOptions:
    timestamps: bool = False
    granularity: str = "sentence"
    max_new_tokens: Optional[int] = None


@dataclass
class AlignOptions:
    granularity: str = "word"


@dataclass
class Segment:
    text: str
    start: float
    end: float
    kind: str = "word"

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "start": round(self.start, 3),
                "end": round(self.end, 3), "kind": self.kind}


@dataclass
class TranscribeResult:
    text: str
    language: Optional[str]
    language_source: str  # auto | forced | server
    segments: List[Segment] = field(default_factory=list)
    timestamp_source: Optional[str] = None
    text_transform: str = TEXT_TRANSFORM
    warnings: List[str] = field(default_factory=list)
    # Token/word level items before sentence aggregation (used for subtitles).
    raw_segments: List[Segment] = field(default_factory=list)


@dataclass
class AlignResult:
    items: List[Segment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class Backend(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...

    def transcribe(self, jobs: Sequence[TranscribeJob], opts: TranscribeOptions) -> List[TranscribeResult]: ...

    def align(self, jobs: Sequence[AlignJob], opts: AlignOptions) -> List[AlignResult]: ...

    def close(self) -> None: ...


def _is_cjk_char(text: str) -> bool:
    if len(text) != 1:
        return False
    code = ord(text)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def clean_text(text: str) -> str:
    """Remove residual ``<asr_text>`` tags and a leading ``language X`` line."""
    if not text:
        return ""
    cleaned = str(text).replace("<asr_text>", "")
    cleaned = _LANG_TAG_RE.sub("", cleaned, count=1)
    return cleaned.strip()


def _kind_for(text: str) -> str:
    return "char" if _is_cjk_char(text) else "word"


def _split_token(text: str, start: float, end: float) -> List[Segment]:
    """Split a multi-character token into characters with proportional timing."""
    chars = list(text)
    if len(chars) <= 1:
        return [Segment(text=text, start=start, end=end, kind="char")]
    span = max(0.0, end - start)
    step = span / len(chars) if chars else 0.0
    out: List[Segment] = []
    for i, ch in enumerate(chars):
        s = start + step * i
        e = start + step * (i + 1)
        out.append(Segment(text=ch, start=s, end=e, kind="char"))
    return out


SENTENCE_END = set("。！？!?；;…")
SENTENCE_PAUSE_MS = 700


def _join_segments(group: Sequence[Segment]) -> str:
    text = ""
    for seg in group:
        piece = seg.text
        if not piece:
            continue
        if text and not (_is_cjk_char(text[-1]) or _is_cjk_char(piece[0])):
            text += " "
        text += piece
    return text


def aggregate_sentences(segments: Sequence[Segment]) -> List[Segment]:
    """Group token items into sentence segments (punctuation or >=700ms pause)."""
    out: List[Segment] = []
    current: List[Segment] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = _join_segments(current).strip()
        if text:
            out.append(Segment(text=text, start=current[0].start,
                               end=max(seg.end for seg in current), kind="sentence"))
        current = []

    for seg in segments:
        if current and (seg.start - current[-1].end) * 1000.0 >= SENTENCE_PAUSE_MS:
            flush()
        current.append(seg)
        if seg.text and seg.text[-1] in SENTENCE_END:
            flush()
    flush()
    return out


def segments_from_timestamps(raw: Any, granularity: str = "sentence") -> List[Segment]:
    """Convert a ForcedAlignResult into qasr segments."""
    if raw is None:
        return []
    items = getattr(raw, "items", None)
    if items is None:
        return []
    segments: List[Segment] = []
    for item in items:
        text = str(getattr(item, "text", ""))
        if not text:
            continue
        start = float(getattr(item, "start_time", 0.0) or 0.0)
        end = float(getattr(item, "end_time", start) or start)
        if end < start:
            end = start
        if granularity == "char" and len(text) > 1:
            segments.extend(_split_token(text, start, end))
        else:
            segments.append(Segment(text=text, start=start, end=end, kind=_kind_for(text)))
    return segments


class _BaseTransformersModel:
    """Shared lazy-loading / device placement."""

    name = "transformers"

    def __init__(
        self,
        model_path: str | Path,
        *,
        aligner_path: Optional[str | Path] = None,
        device: str = "auto",
        dtype: str = "auto",
        attn_implementation: str = "auto",
        max_inference_batch_size: int = 1,
        max_new_tokens: Optional[int] = None,
        offline: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.aligner_path = Path(aligner_path) if aligner_path else None
        self.requested_device = device
        self.requested_dtype = dtype
        self.attn_implementation = attn_implementation
        self.max_inference_batch_size = max(1, int(max_inference_batch_size or 1))
        self.max_new_tokens = max_new_tokens
        self.offline = offline
        self.device: Optional[str] = None
        self.dtype: Optional[str] = None
        self._torch = None
        self._torch_dtype_obj = None

    def _prepare(self) -> None:
        if self._torch is not None:
            return
        configure_offline(self.offline)
        torch_mod = import_torch()
        self._torch = torch_mod
        self.device = resolve_device(self.requested_device, torch_mod)
        self.dtype = resolve_dtype(self.requested_dtype, self.device, torch_mod)
        self._torch_dtype_obj = torch_dtype(self.dtype, torch_mod)

    def _model_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "dtype": self._torch_dtype_obj,
            "local_files_only": self.offline,
            "low_cpu_mem_usage": True,
        }
        if self.attn_implementation and self.attn_implementation != "auto":
            kwargs["attn_implementation"] = self.attn_implementation
        return kwargs

    def capabilities(self) -> Capabilities:
        return Capabilities(name=self.name)


class TransformersBackend(_BaseTransformersModel):
    """Local transformers backend (CUDA / MPS / CPU)."""

    def __init__(self, *args: Any, aligner_path: Optional[str | Path] = None, **kwargs: Any) -> None:
        super().__init__(*args, aligner_path=aligner_path, **kwargs)
        self._asr: Any = None
        self._aligner: Any = None

    # -- loading ---------------------------------------------------------
    def ensure_loaded(self, *, need_align: bool = False) -> None:
        self._prepare()
        if self._asr is None:
            qwen_asr = import_qwen_asr()
            kwargs = self._model_kwargs()
            aligner_kwargs: Dict[str, Any] = {
                "dtype": self._torch_dtype_obj,
                "local_files_only": self.offline,
                "low_cpu_mem_usage": True,
            }
            with stdout_to_stderr():
                self._asr = qwen_asr.Qwen3ASRModel.from_pretrained(
                    str(self.model_path),
                    forced_aligner=str(self.aligner_path) if need_align and self.aligner_path else None,
                    forced_aligner_kwargs=aligner_kwargs if need_align and self.aligner_path else None,
                    max_inference_batch_size=self.max_inference_batch_size,
                    max_new_tokens=self.max_new_tokens or 512,
                    **kwargs,
                )
                self._asr.model.to(self.device)
                if getattr(self._asr, "forced_aligner", None) is not None:
                    self._asr.forced_aligner.model.to(self.device)
                    self._aligner = self._asr.forced_aligner
        elif need_align and self._aligner is None:
            raise ModelError(
                "timestamps were requested but the backend was loaded without an aligner",
                hint="re-run with --timestamps and a valid --aligner",
            )

    def ensure_aligner(self) -> None:
        if self._aligner is not None:
            return
        # For align-only runs the aligner model is passed as ``model_path``.
        aligner_source = self.aligner_path or self.model_path
        self._prepare()
        qwen_asr = import_qwen_asr()
        with stdout_to_stderr():
            self._aligner = qwen_asr.Qwen3ForcedAligner.from_pretrained(
                str(aligner_source),
                **self._model_kwargs(),
            )
            self._aligner.model.to(self.device)

    # -- inference -------------------------------------------------------
    def transcribe(self, jobs: Sequence[TranscribeJob], opts: TranscribeOptions) -> List[TranscribeResult]:
        if not jobs:
            return []
        need_align = bool(opts.timestamps)
        if need_align and self.aligner_path is None:
            raise CapabilityError(
                "timestamps require an aligner model",
                hint="qasr models pull aligner",
            )
        self.ensure_loaded(need_align=need_align)

        audios = [str(job.item.path) for job in jobs]
        contexts: List[str] = [job.context or "" for job in jobs]
        languages: List[Optional[str]] = [job.language for job in jobs]

        with stdout_to_stderr():
            raw_results = self._asr.transcribe(
                audio=audios,
                context=contexts,
                language=languages,
                return_time_stamps=need_align,
            )

        results: List[TranscribeResult] = []
        for job, raw in zip(jobs, raw_results):
            text = clean_text(getattr(raw, "text", "") or "")
            language = (getattr(raw, "language", "") or "").strip() or None
            source = "forced" if job.language else "auto"
            segments = segments_from_timestamps(
                getattr(raw, "time_stamps", None) if need_align else None,
                opts.granularity,
            )
            raw_segments = list(segments)
            if opts.granularity == "sentence" and segments:
                segments = aggregate_sentences(segments)
            warnings: List[str] = []
            if need_align and text and not segments:
                warnings.append("timestamps were requested but the aligner returned no items")
            results.append(
                TranscribeResult(
                    text=text,
                    language=language,
                    language_source=source,
                    segments=segments,
                    timestamp_source="forced_aligner" if segments else None,
                    warnings=warnings,
                    raw_segments=raw_segments,
                )
            )
        return results

    def align(self, jobs: Sequence[AlignJob], opts: AlignOptions) -> List[AlignResult]:
        if not jobs:
            return []
        self.ensure_aligner()
        audios = [str(job.item.path) for job in jobs]
        texts = [job.text for job in jobs]
        languages = [job.language for job in jobs]
        with stdout_to_stderr():
            raw_results = self._aligner.align(audio=audios, text=texts, language=languages)
        out: List[AlignResult] = []
        for raw in raw_results:
            items = segments_from_timestamps(raw, "word")
            out.append(AlignResult(items=items, warnings=[]))
        return out

    def close(self) -> None:
        self._asr = None
        self._aligner = None
        if self._torch is not None and self.device and self.device.startswith("cuda"):
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass


class VLLMBackend(_BaseTransformersModel):
    """Offline vLLM backend (Linux + CUDA). Loaded lazily via qwen_asr.LLM."""

    name = "vllm"

    def __init__(
        self,
        *args: Any,
        gpu_memory_utilization: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Device remapping must happen before torch initializes CUDA. Doing it
        # in __init__ is the earliest backend-level hook; the CLI also calls
        # this before its vLLM availability probe.
        if set_cuda_visible_device(self.requested_device) is not None:
            self.requested_device = "cuda:0"
        self.gpu_memory_utilization = gpu_memory_utilization
        self._asr: Any = None
        self._aligner: Any = None

    def _prepare(self) -> None:
        require_vllm()
        super()._prepare()
        if not self.device or not self.device.startswith("cuda"):
            raise RuntimeFailure(
                "the vLLM backend requires a CUDA device",
                hint="use --backend transformers for MPS/CPU",
            )
        # vLLM selects the device via CUDA_VISIBLE_DEVICES; indexed devices
        # were remapped in __init__ to the single visible GPU (cuda:0).

    def _engine_kwargs(self) -> Dict[str, Any]:
        dtype = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[str(self.dtype)]
        kwargs: Dict[str, Any] = {"dtype": dtype}
        if self.gpu_memory_utilization is not None:
            kwargs["gpu_memory_utilization"] = float(self.gpu_memory_utilization)
        return kwargs

    def _aligner_kwargs(self) -> Dict[str, Any]:
        return {
            "dtype": self._torch_dtype_obj,
            "local_files_only": self.offline,
            "low_cpu_mem_usage": True,
        }

    def ensure_loaded(self, *, need_align: bool = False) -> None:
        self._prepare()
        if self._asr is None:
            configure_offline(self.offline)
            configure_vllm_logging()
            with stdout_to_stderr():
                qwen_asr = import_qwen_asr()
            redirect_vllm_logging()
            guard_stdout_at_exit()
            use_aligner = need_align and self.aligner_path is not None
            with stdout_to_stderr():
                self._asr = qwen_asr.Qwen3ASRModel.LLM(
                    str(self.model_path),
                    forced_aligner=(
                        str(self.aligner_path) if use_aligner else None
                    ),
                    forced_aligner_kwargs=(
                        self._aligner_kwargs() if use_aligner else None
                    ),
                    max_inference_batch_size=self.max_inference_batch_size,
                    max_new_tokens=self.max_new_tokens or 512,
                    **self._engine_kwargs(),
                )
                self._aligner = getattr(self._asr, "forced_aligner", None)
        elif need_align and self._aligner is None:
            raise ModelError(
                "timestamps were requested but the vLLM backend was loaded without an aligner",
                hint="re-run with --timestamps and a valid --aligner",
            )

    def ensure_aligner(self) -> None:
        if self._aligner is not None:
            return
        self._prepare()
        with stdout_to_stderr():
            qwen_asr = import_qwen_asr()
        redirect_vllm_logging()
        aligner_source = self.aligner_path or self.model_path
        with stdout_to_stderr():
            self._aligner = qwen_asr.Qwen3ForcedAligner.from_pretrained(
                str(aligner_source),
                **self._aligner_kwargs(),
            )
            self._aligner.model.to(self.device)

    # ---- streaming (P2) -------------------------------------------------
    def init_streaming_state(
        self,
        *,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> Any:
        """Create one streaming state; streaming is vLLM-only and single-stream."""
        self.ensure_loaded()
        with stdout_to_stderr():
            return self._asr.init_streaming_state(
                context=context,
                language=language,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                chunk_size_sec=chunk_size_sec,
            )

    def streaming_transcribe(self, pcm16k: Any, state: Any) -> Any:
        with stdout_to_stderr():
            return self._asr.streaming_transcribe(pcm16k, state)

    def finish_streaming_transcribe(self, state: Any) -> Any:
        with stdout_to_stderr():
            return self._asr.finish_streaming_transcribe(state)

    def transcribe(self, jobs: Sequence[TranscribeJob], opts: TranscribeOptions) -> List[TranscribeResult]:
        if not jobs:
            return []
        need_align = bool(opts.timestamps)
        if need_align and self.aligner_path is None:
            raise CapabilityError(
                "timestamps require an aligner model",
                hint="qasr models pull aligner",
            )
        self.ensure_loaded(need_align=need_align)
        audios = [str(job.item.path) for job in jobs]
        contexts: List[str] = [job.context or "" for job in jobs]
        languages: List[Optional[str]] = [job.language for job in jobs]
        with stdout_to_stderr():
            raw_results = self._asr.transcribe(
                audio=audios,
                context=contexts,
                language=languages,
                return_time_stamps=need_align,
            )
        results: List[TranscribeResult] = []
        for job, raw in zip(jobs, raw_results):
            text = clean_text(getattr(raw, "text", "") or "")
            language = (getattr(raw, "language", "") or "").strip() or None
            source = "forced" if job.language else "auto"
            segments = segments_from_timestamps(
                getattr(raw, "time_stamps", None) if need_align else None,
                opts.granularity,
            )
            raw_segments = list(segments)
            if opts.granularity == "sentence" and segments:
                segments = aggregate_sentences(segments)
            warnings: List[str] = []
            if need_align and text and not segments:
                warnings.append("timestamps were requested but the aligner returned no items")
            results.append(
                TranscribeResult(
                    text=text,
                    language=language,
                    language_source=source,
                    segments=segments,
                    timestamp_source="forced_aligner" if segments else None,
                    warnings=warnings,
                    raw_segments=raw_segments,
                )
            )
        return results

    def align(self, jobs: Sequence[AlignJob], opts: AlignOptions) -> List[AlignResult]:
        if not jobs:
            return []
        self.ensure_aligner()
        with stdout_to_stderr():
            raw_results = self._aligner.align(
                audio=[str(job.item.path) for job in jobs],
                text=[job.text for job in jobs],
                language=[job.language for job in jobs],
            )
        return [
            AlignResult(items=segments_from_timestamps(raw, "word"), warnings=[])
            for raw in raw_results
        ]

    def close(self) -> None:
        self._asr = None
        self._aligner = None
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass


_CACHE: Dict[tuple, TransformersBackend] = {}
_VLLM_CACHE: Dict[tuple, VLLMBackend] = {}


def build_transformers_backend(
    model_path: str | Path,
    *,
    aligner_path: Optional[str | Path] = None,
    device: str = "auto",
    dtype: str = "auto",
    attn_implementation: str = "auto",
    max_inference_batch_size: int = 1,
    max_new_tokens: Optional[int] = None,
    offline: bool = False,
) -> TransformersBackend:
    """Process-wide cache keyed by every load-affecting parameter."""
    key = (
        str(Path(model_path).resolve()),
        str(Path(aligner_path).resolve()) if aligner_path else None,
        device,
        dtype,
        attn_implementation,
        int(max_inference_batch_size or 1),
        int(max_new_tokens) if max_new_tokens else None,
        bool(offline),
    )
    backend = _CACHE.get(key)
    if backend is None:
        backend = TransformersBackend(
            model_path,
            aligner_path=aligner_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            offline=offline,
        )
        _CACHE[key] = backend
    return backend


def build_vllm_backend(
    model_path: str | Path,
    *,
    aligner_path: Optional[str | Path] = None,
    device: str = "auto",
    dtype: str = "auto",
    max_inference_batch_size: int = 1,
    max_new_tokens: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    offline: bool = False,
) -> VLLMBackend:
    key = (
        str(Path(model_path).resolve()),
        str(Path(aligner_path).resolve()) if aligner_path else None,
        device,
        dtype,
        int(max_inference_batch_size or 1),
        int(max_new_tokens) if max_new_tokens else None,
        float(gpu_memory_utilization) if gpu_memory_utilization is not None else None,
        bool(offline),
    )
    backend = _VLLM_CACHE.get(key)
    if backend is None:
        backend = VLLMBackend(
            model_path,
            aligner_path=aligner_path,
            device=device,
            dtype=dtype,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            offline=offline,
        )
        _VLLM_CACHE[key] = backend
    return backend


def clear_backend_cache() -> None:
    for backend in list(_CACHE.values()) + list(_VLLM_CACHE.values()):
        backend.close()
    _CACHE.clear()
    _VLLM_CACHE.clear()
