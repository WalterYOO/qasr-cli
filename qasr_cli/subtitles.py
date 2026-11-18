"""Cue construction and SRT/VTT rendering (pure Python).

All timings are integer milliseconds internally. Seconds are converted exactly
once, at the entry points, with half-up rounding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .backends import Segment

MAX_CUE_MS = 7000
MAX_CUE_WIDTH = 42
PAUSE_MS = 700
SENTENCE_END = set("。！？!?；;")
CJK_RANGES = (
    (0x1100, 0x11FF),
    (0x2E80, 0x303F),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFF60),
    (0xFFE0, 0xFFE6),
)


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


def to_ms(seconds: float) -> int:
    """Half-up conversion from seconds to integer milliseconds."""
    if seconds is None:
        return 0
    value = float(seconds) * 1000.0
    if not math.isfinite(value):
        return 0
    return int(math.floor(value + 0.5))


def is_wide_char(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in CJK_RANGES)


def display_width(text: str) -> int:
    return sum(2 if is_wide_char(ch) else 1 for ch in text)


def _normalize_segments(segments: Sequence[Segment], audio_duration_ms: Optional[int]) -> List[Segment]:
    out: List[Segment] = []
    prev_end = 0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start = max(0, to_ms(seg.start))
        end = max(start + 1, to_ms(seg.end))
        start = max(start, prev_end)
        if audio_duration_ms:
            end = min(end, audio_duration_ms)
            if start >= audio_duration_ms:
                if prev_end >= audio_duration_ms:
                    continue  # no room left in the audio; drop the token
                start = audio_duration_ms - 1
            end = min(max(end, start + 1), audio_duration_ms)
        else:
            end = max(start + 1, end)
        out.append(Segment(text=text, start=start, end=end, kind=seg.kind))
        prev_end = end
    return out


def join_tokens(tokens: Iterable[str]) -> str:
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    parts = [tokens[0]]
    for token in tokens[1:]:
        prev = parts[-1]
        if is_wide_char(prev[-1]) or is_wide_char(token[0]):
            parts.append(token)
        else:
            parts.append(" " + token)
    return "".join(parts)


def _split_wide_token(token: Segment, max_width: int) -> List[Segment]:
    """Split an over-wide CJK-ish token so cue width stays within the limit."""
    text = token.text
    if len(text) <= 1 or display_width(text) <= max_width:
        return [token]
    if not any(is_wide_char(ch) for ch in text):
        return [token]  # never split Latin words
    pieces: List[str] = []
    buf = ""
    width = 0
    for ch in text:
        w = 2 if is_wide_char(ch) else 1
        if buf and width + w > max_width:
            pieces.append(buf)
            buf = ""
            width = 0
        buf += ch
        width += w
    if buf:
        pieces.append(buf)
    total_chars = sum(len(p) for p in pieces) or 1
    span = max(0.0, token.end - token.start)
    out: List[Segment] = []
    char_cursor = 0
    for piece in pieces:
        start = token.start + span * (char_cursor / total_chars)
        char_cursor += len(piece)
        end = token.start + span * (char_cursor / total_chars)
        out.append(Segment(text=piece, start=start, end=end, kind=token.kind))
    return out


def build_cues(
    segments: Sequence[Segment],
    *,
    audio_duration_s: Optional[float] = None,
    max_cue_ms: int = MAX_CUE_MS,
    max_width: int = MAX_CUE_WIDTH,
    pause_ms: int = PAUSE_MS,
) -> List[Cue]:
    """Group token segments into subtitle cues.

    Break priority: sentence punctuation, then pauses >= 700 ms, while keeping
    every cue within 7 s and 42 display columns. CJK tokens count 2 columns;
    Latin words are never split.
    """
    duration_ms = to_ms(audio_duration_s) if audio_duration_s else None
    tokens: List[Segment] = []
    for token in _normalize_segments(segments, duration_ms):
        tokens.extend(_split_wide_token(token, max_width))
    cues: List[Cue] = []

    current: List[Segment] = []
    current_text = ""

    def flush() -> None:
        nonlocal current, current_text
        if not current:
            return
        start = current[0].start
        end = max(cur.end for cur in current)
        end = max(start + 1, end)
        if duration_ms:
            end = min(end, duration_ms)
            end = max(start + 1, end)
        cues.append(Cue(start_ms=start, end_ms=end, text=join_tokens(s.text for s in current)))
        current = []
        current_text = ""

    for token in tokens:
        candidate_text = join_tokens([*(s.text for s in current), token.text])
        candidate_span = token.end - (current[0].start if current else token.start)
        gap = token.start - current[-1].end if current else 0
        too_long = bool(current) and (candidate_span > max_cue_ms or display_width(candidate_text) > max_width)
        long_pause = bool(current) and gap >= pause_ms
        if too_long or long_pause:
            flush()
            candidate_text = token.text
            candidate_span = token.end - token.start
        current.append(token)
        current_text = candidate_text
        if token.text and token.text[-1] in SENTENCE_END:
            flush()
    flush()

    # Enforce monotonicity and a minimum 1 ms duration.
    fixed: List[Cue] = []
    prev_end = 0
    for cue in cues:
        start = max(cue.start_ms, prev_end)
        end = max(start + 1, cue.end_ms)
        if duration_ms:
            end = min(end, duration_ms)
            if start >= duration_ms:
                if prev_end >= duration_ms:
                    continue  # no room left; drop the cue
                start = duration_ms - 1
            end = min(max(end, start + 1), duration_ms)
        fixed.append(Cue(start_ms=start, end_ms=end, text=cue.text))
        prev_end = end
    return fixed


def _fmt_timestamp(ms: int, sep: str) -> str:
    if ms < 0:
        ms = 0
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{sep}{millis:03d}"


def render_srt(cues: Sequence[Cue]) -> str:
    lines: List[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.append(str(index))
        lines.append(f"{_fmt_timestamp(cue.start_ms, ',')} --> {_fmt_timestamp(cue.end_ms, ',')}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n" if cues else ""


def render_vtt(cues: Sequence[Cue]) -> str:
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.append(f"{_fmt_timestamp(cue.start_ms, '.')} --> {_fmt_timestamp(cue.end_ms, '.')}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def parse_srt(text: str) -> List[Cue]:
    """Parse SRT into cues (used by ``qasr align --srt``)."""
    cues: List[Cue] = []
    blocks = [b.strip() for b in text.replace("\r\n", "\n").split("\n\n") if b.strip()]
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        if "-->" not in lines[1] and "-->" in lines[0]:
            lines = lines[0:]
        time_line_idx = 0 if "-->" in lines[0] else 1
        if time_line_idx >= len(lines) or "-->" not in lines[time_line_idx]:
            continue
        left, right = lines[time_line_idx].split("-->", 1)
        cues.append(
            Cue(
                start_ms=_parse_ts(left.strip()),
                end_ms=_parse_ts(right.strip().split(" ")[0]),
                text=" ".join(lines[time_line_idx + 1:]).strip(),
            )
        )
    return cues


def _parse_ts(value: str) -> int:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3_600_000 + int(m) * 60_000 + to_ms(float(s))
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60_000 + to_ms(float(s))
        return to_ms(float(parts[0]))
    except (ValueError, TypeError):
        return 0


def cues_text(cues: Sequence[Cue]) -> str:
    return " ".join(cue.text.strip() for cue in cues if cue.text.strip())
