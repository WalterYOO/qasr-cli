"""Streaming ASR state machine and audio sources (P2).

This module never imports torch/vllm/sounddevice at import time. The vLLM
backend and the optional microphone dependency are resolved lazily by the
command layer or inside the source generators.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, TextIO

from .audio import AudioInfo, read_window_16k
from .errors import (
    EXIT_INTERRUPT,
    EXIT_OK,
    EXIT_QUALITY,
    CapabilityError,
    InputError,
    ModelError,
    QasrError,
    QualityError,
    RuntimeFailure,
)

SAMPLE_RATE = 16000
MIC_BLOCK_MS = 100


@dataclass
class StreamSettings:
    chunk_samples: int
    chunk_ms: int
    channels: int = 1
    input_kind: str = "file"  # file | stdin | mic
    emit: str = "text"
    realtime: bool = False
    max_samples: int = 3600 * SAMPLE_RATE
    context: str = ""
    language: Optional[str] = None
    unfixed_chunks: int = 2
    unfixed_tokens: int = 5


def _now() -> float:
    return time.monotonic()


def _tail_warning(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def iter_pcm_s16le(
    stream: Any,
    chunk_samples: int,
    channels: int,
    *,
    warning: Callable[[str], None] = _tail_warning,
) -> Iterator[Any]:
    """Yield mono float32 chunks decoded from a 16-bit little-endian stream."""
    frame_bytes = 2 * int(channels)
    chunk_bytes = max(frame_bytes, int(chunk_samples) * frame_bytes)
    pending = bytearray()

    while True:
        data = stream.read(chunk_bytes)
        if not data:
            break
        pending.extend(data)
        if len(pending) < chunk_bytes:
            continue
        block = bytes(pending[:chunk_bytes])
        del pending[:chunk_bytes]
        yield _decode_pcm(block, channels)

    if pending:
        usable = len(pending) - (len(pending) % frame_bytes)
        if usable != len(pending):
            warning(
                f"discarding {len(pending) - usable} trailing byte(s) "
                "that do not form a complete PCM frame"
            )
        if usable:
            yield _decode_pcm(bytes(pending[:usable]), channels)


def _decode_pcm(data: bytes, channels: int) -> Any:
    import numpy as np

    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples.astype(np.float32, copy=False)


def iter_file_chunks(
    path: Path,
    info: AudioInfo,
    chunk_samples: int,
) -> Iterator[Any]:
    """Yield 16 kHz mono chunks without loading the whole file."""
    sent = 0
    while True:
        start_src = int(round(sent * info.sample_rate / SAMPLE_RATE))
        if start_src >= info.frames:
            break
        end_src = int(
            round((sent + chunk_samples) * info.sample_rate / SAMPLE_RATE)
        )
        end_src = min(max(end_src, start_src + 1), info.frames)
        try:
            data = read_window_16k(path, start_src, end_src)
        except QasrError:
            raise
        except Exception as exc:  # noqa: BLE001 - decoder-specific errors
            raise InputError(f"cannot decode audio '{path}': {exc}") from exc
        if data is None or len(data) == 0:
            break
        yield data
        sent += int(chunk_samples)


def _import_sounddevice() -> Any:
    import importlib

    try:
        return importlib.import_module("sounddevice")
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise CapabilityError(
            "microphone input requires the optional 'sounddevice' package",
            hint='uv tool install "qasr-cli[mic]"',
        ) from exc


def require_mic() -> None:
    """Fail fast when microphone support is missing, before loading a model."""
    _import_sounddevice()


def iter_mic_chunks(
    *,
    block_ms: int = MIC_BLOCK_MS,
) -> Iterator[Any]:
    """Yield 100 ms mono float32 blocks from the default microphone."""
    import numpy as np

    blocksize = max(1, int(round(block_ms * SAMPLE_RATE / 1000)))
    sd = _import_sounddevice()
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
        ) as stream:
            while True:
                data, _overflowed = stream.read(blocksize)
                samples = np.asarray(data, dtype=np.float32).reshape(-1)
                if samples.size:
                    yield samples
    except QasrError:
        raise
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - device backends vary
        detail = str(exc).strip().replace("\n", " ")
        if len(detail) > 200:
            detail = detail[:197] + "..."
        raise ModelError(
            f"microphone device error: {detail}" if detail else "microphone device error"
        ) from exc


class SnapshotWriter:
    """Write cumulative snapshots for both JSONL and human modes."""

    def __init__(
        self,
        emit: str,
        *,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self.emit = emit
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.seq = 0
        self.lines = 0
        self._tty = False
        try:
            self._tty = bool(self.stdout.isatty())
        except Exception:
            self._tty = False

    def snapshot(
        self,
        *,
        text: str,
        language: Optional[str],
        audio_s: float,
        elapsed_s: float,
        is_final: bool,
    ) -> None:
        if self.emit == "jsonl":
            payload = {
                "schema_version": 1,
                "ok": True,
                "command": "stream",
                "seq": self.seq,
                "audio_s": round(float(audio_s), 3),
                "elapsed_s": round(float(elapsed_s), 3),
                "language": language or None,
                "text": text or "",
                "is_final": bool(is_final),
            }
            print(json.dumps(payload, ensure_ascii=False), file=self.stdout, flush=True)
            self.seq += 1
            self.lines += 1
            return

        if self._tty:
            self.stdout.write(f"\r{text or ''}")
            if is_final:
                self.stdout.write("\n")
            self.stdout.flush()
        elif is_final:
            print(text or "", file=self.stdout, flush=True)

    def error(self, exc: QasrError) -> None:
        """Append a terminal error snapshot (used for cap and mid-stream errors)."""
        if self.emit != "jsonl":
            print(f"qasr: {exc.kind} error: {exc.message}", file=self.stderr)
            if exc.hint:
                print(f"hint: {exc.hint}", file=self.stderr)
            return
        error = exc.to_dict()
        payload = {
            "schema_version": 1,
            "ok": False,
            "command": "stream",
            "seq": self.seq,
            "is_final": True,
            "error": error,
        }
        print(json.dumps(payload, ensure_ascii=False), file=self.stdout, flush=True)
        self.seq += 1
        self.lines += 1


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _as_language(value: Any) -> Optional[str]:
    text = _as_text(value).strip()
    return text or None


def _error_from_exception(exc: BaseException, *, where: str) -> QasrError:
    if isinstance(exc, QasrError):
        return exc
    return RuntimeFailure(f"{where} failed: {exc}")


def run_stream(
    backend: Any,
    chunks: Iterable[Any],
    settings: StreamSettings,
    *,
    clock: Optional[Callable[[], float]] = None,
    sleep: Optional[Callable[[float], None]] = None,
    started_at: Optional[float] = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Drive one streaming session and emit snapshots.

    Returns the process exit code. Sources are only read between chunks; on
    Ctrl-C the tail buffered by the model is still flushed and a final
    snapshot is emitted before returning 130. ``started_at`` lets the command
    include model loading in ``elapsed_s``; when omitted the state machine
    starts its own timer.
    """
    import numpy as np

    clock = clock or _now
    sleep = sleep or time.sleep
    writer = SnapshotWriter(settings.emit, stdout=stdout, stderr=stderr)

    started = clock() if started_at is None else started_at
    audio_sent = 0
    interrupted = False
    cap_hit = False
    state = None
    last_chunk_id = 0

    try:
        try:
            state = backend.init_streaming_state(
                context=settings.context,
                language=settings.language,
                unfixed_chunk_num=settings.unfixed_chunks,
                unfixed_token_num=settings.unfixed_tokens,
                chunk_size_sec=settings.chunk_ms / 1000.0,
            )
            last_chunk_id = int(getattr(state, "chunk_id", 0) or 0)
            iterator = iter(chunks)
            try:
                current = next(iterator)
            except StopIteration:
                current = None
            while current is not None:
                try:
                    following = next(iterator)
                except StopIteration:
                    following = None
                if audio_sent >= settings.max_samples:
                    cap_hit = True
                    break
                samples = np.asarray(current, dtype=np.float32).reshape(-1)
                current = following
                if samples.size == 0:
                    continue
                complete_chunk = samples.size >= settings.chunk_samples
                if audio_sent + samples.size > settings.max_samples:
                    samples = samples[: settings.max_samples - audio_sent]
                    cap_hit = True
                if samples.size:
                    chunk_started = clock()
                    state = backend.streaming_transcribe(samples, state)
                    audio_sent += int(samples.size)
                    chunk_id = int(getattr(state, "chunk_id", 0) or 0)
                    if chunk_id > last_chunk_id:
                        last_chunk_id = chunk_id
                        writer.snapshot(
                            text=_as_text(getattr(state, "text", "")),
                            language=_as_language(getattr(state, "language", "")),
                            audio_s=audio_sent / SAMPLE_RATE,
                            elapsed_s=clock() - started,
                            is_final=False,
                        )
                    # Wall-clock pacing: only between complete chunks with more
                    # audio waiting; never pre-sleep, catch up, or sleep on EOF.
                    # The chunk's own processing time counts against the budget,
                    # so slow chunks sleep less instead of falling behind.
                    if (
                        settings.realtime
                        and settings.input_kind == "file"
                        and complete_chunk
                        and following is not None
                        and not cap_hit
                    ):
                        processing = clock() - chunk_started
                        sleep(max(0.0, settings.chunk_ms / 1000.0 - processing))
                if cap_hit:
                    break
        except KeyboardInterrupt:
            interrupted = True

        if cap_hit:
            error = QualityError(
                "stream exceeded --max-stream-seconds "
                f"({settings.max_samples / SAMPLE_RATE:g}s); this is a run limit, "
                "not transcription quality"
            )
            writer.error(error)
            return EXIT_QUALITY

        if state is not None:
            try:
                state = backend.finish_streaming_transcribe(state)
            except KeyboardInterrupt:
                interrupted = True
            except Exception as exc:  # noqa: BLE001
                raise _error_from_exception(exc, where="stream finish") from exc

        if state is not None:
            writer.snapshot(
                text=_as_text(getattr(state, "text", "")),
                language=_as_language(getattr(state, "language", "")),
                audio_s=audio_sent / SAMPLE_RATE,
                elapsed_s=clock() - started,
                is_final=True,
            )
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:  # noqa: BLE001 - normalize third-party errors
        error = _error_from_exception(exc, where="stream")
        if settings.emit == "jsonl" and writer.lines > 0:
            writer.error(error)
            return error.exit_code
        if error is exc:
            raise
        raise error from exc

    if interrupted:
        return EXIT_INTERRUPT
    return EXIT_OK


__all__ = [
    "MIC_BLOCK_MS",
    "SAMPLE_RATE",
    "SnapshotWriter",
    "StreamSettings",
    "iter_file_chunks",
    "iter_mic_chunks",
    "iter_pcm_s16le",
    "require_mic",
    "run_stream",
]
