"""Audio probing (duration/samplerate/channels). Must not import torch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
            f"cannot read audio '{target.name}': {exc}",
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
