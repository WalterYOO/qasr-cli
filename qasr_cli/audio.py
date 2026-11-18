"""Audio probing (duration/samplerate/channels). Must not import torch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .errors import InputError

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm", ".mp4")

# Extension -> hint when probing fails but the container is theoretically supported.
_CODEC_HINTS = {
    ".m4a": "m4a/mp4 need ffmpeg support; convert to wav or flac first",
    ".mp4": "m4a/mp4 need ffmpeg support; convert to wav or flac first",
    ".webm": "webm needs ffmpeg support; convert to wav or flac first",
    ".opus": "opus may require a newer libsndfile; convert to wav or flac first",
}


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration_s: float
    sample_rate: int
    channels: int
    frames: int
    format: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return self.frames <= 0 or self.duration_s <= 0.0


def probe(path: str | Path) -> AudioInfo:
    """Read audio metadata with soundfile; raise InputError on unreadable files."""
    target = Path(path)
    if not target.is_file():
        raise InputError(f"audio file not found: {target}")
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - base dependency
        raise InputError("soundfile is not installed", hint='uv pip install "qasr-cli"') from exc

    try:
        info = sf.info(str(target))
    except Exception as exc:  # soundfile raises various RuntimeError subclasses
        hint = _CODEC_HINTS.get(target.suffix.casefold())
        raise InputError(
            f"cannot read audio '{target}': {exc}",
            hint=hint,
            details={"path": str(target)},
        ) from exc

    frames = int(info.frames)
    rate = int(info.samplerate)
    duration = (frames / rate) if rate > 0 else 0.0
    if not math.isfinite(duration):
        duration = 0.0
    return AudioInfo(
        path=target.resolve(),
        duration_s=round(duration, 6),
        sample_rate=rate,
        channels=int(info.channels),
        frames=frames,
        format=info.format,
    )


def max_new_tokens_for(duration_s: float) -> int:
    """DESIGN §10: auto = min(16384, max(1024, ceil(duration_s*8)+256))."""
    return min(16384, max(1024, math.ceil(float(duration_s) * 8) + 256))


def read_window_16k(
    path: str | Path,
    start_sample: int,
    end_sample: int,
    *,
    block_frames: int = 1 << 20,
) -> Any:
    """Read ``[start_sample, end_sample)`` as mono float32 at 16 kHz.

    Uses soundfile block reads; resampling is a small linear interpolation so
    no extra dependency is needed. Imported lazily to keep the base CLI light.
    """
    import numpy as np
    import soundfile as sf

    target_sr = 16000
    start = max(0, int(start_sample))
    end = max(start, int(end_sample))
    with sf.SoundFile(str(path)) as handle:
        source_sr = int(handle.samplerate) or target_sr
        handle.seek(start)
        remaining = end - start
        chunks = []
        while remaining > 0:
            data = handle.read(min(block_frames, remaining), dtype="float32",
                               always_2d=True)
            if data.size == 0:
                break
            chunks.append(data.mean(axis=1))
            remaining -= len(data)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    mono = np.concatenate(chunks)
    if source_sr != target_sr and len(mono) > 0:
        duration = len(mono) / source_sr
        target_count = max(1, int(round(duration * target_sr)))
        source_positions = np.arange(len(mono), dtype=np.float64) / source_sr
        target_positions = np.arange(target_count, dtype=np.float64) / target_sr
        mono = np.interp(target_positions, source_positions, mono).astype(np.float32)
    return mono


def write_wav_16k(path: str | Path, samples: Any) -> Path:
    """Write mono float samples as a 16 kHz PCM16 WAV."""
    import soundfile as sf

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), samples, 16000, subtype="PCM_16")
    return target
