"""Long-audio window planning and text ownership (pure functions only).

No I/O, no model imports, no torch: this module is safe to import from the
base CLI and is exercised directly by unit tests.
"""

from __future__ import annotations

import difflib
import math
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Sequence

from .errors import UsageError

MAX_WINDOW_SECONDS = 180.0
DEFAULT_OVERLAP_SECONDS = 3.0
DEFAULT_SEARCH_SECONDS = 5.0
MIN_CORE_SLACK_SECONDS = 2.0
MERGE_EPSILON = 0.001
MIN_MATCH_RATIO = 0.5
# A window's text and its reference slice describe the same speech, so true
# matching blocks sit on one diagonal. Blocks off that diagonal are repeated
# phrases elsewhere in the reference; including them lets one window claim
# several windows' worth of text and starves every later window.
MATCH_BAND_RATIO = 0.4
MATCH_BAND_MIN = 60
CANDIDATE_MULTIPLIER = 2.0
CANDIDATE_PAD = 200
MAX_SPAN_MULTIPLIER = 2.0
MAX_SPAN_PAD = 200
# A reference that ends inside a window can still match >= MIN_MATCH_RATIO of
# the window's characters. Check the window's tail separately so that partial
# coverage is not mistaken for full coverage.
TAIL_START_RATIO = 0.8
MIN_TAIL_MATCH_RATIO = 0.2
# Below this many characters the tail slice is noise: a 4-char window "abcd"
# against reference "abc" would look like a missing tail even though it is a
# normal partial match. Only judge the tail when there is enough of it.
MIN_TAIL_CHARS = 8


@dataclass(frozen=True)
class Window:
    index: int
    audio_start_sample: int
    audio_end_sample: int
    core_start_sample: int
    core_end_sample: int

    def audio_seconds(self, sr: int) -> tuple[float, float]:
        return self.audio_start_sample / sr, self.audio_end_sample / sr

    def core_seconds(self, sr: int) -> tuple[float, float]:
        return self.core_start_sample / sr, self.core_end_sample / sr

    def to_dict(self, sr: int) -> dict:
        return {
            "index": self.index,
            "audio_start_s": round(self.audio_start_sample / sr, 3),
            "audio_end_s": round(self.audio_end_sample / sr, 3),
            "core_start_s": round(self.core_start_sample / sr, 3),
            "core_end_s": round(self.core_end_sample / sr, 3),
            "audio_start_sample": self.audio_start_sample,
            "audio_end_sample": self.audio_end_sample,
            "core_start_sample": self.core_start_sample,
            "core_end_sample": self.core_end_sample,
        }


@dataclass(frozen=True)
class TextSpan:
    window_index: int
    start_char: int
    end_char: int
    match_ratio: float
    text: str = ""
    # Fraction of the window text's final TAIL_START_RATIO..1.0 slice covered
    # by matching blocks. A reference that stops inside the window can still
    # reach MIN_MATCH_RATIO overall while this tail is empty.
    tail_match_ratio: float = 1.0

    @property
    def is_weak(self) -> bool:
        return self.match_ratio < MIN_MATCH_RATIO

    @property
    def is_incomplete(self) -> bool:
        return self.is_weak or self.tail_match_ratio < MIN_TAIL_MATCH_RATIO


def _check_params(
    window_s: float,
    overlap_s: float,
    search_s: float,
    max_window_s: float = MAX_WINDOW_SECONDS,
) -> None:
    if not math.isfinite(float(window_s)) or window_s <= 0 or window_s > max_window_s:
        raise UsageError(
            f"--window-seconds must be > 0 and <= {int(max_window_s)}",
            details={"window_seconds": window_s, "max": max_window_s},
        )
    if not math.isfinite(float(overlap_s)) or overlap_s < 0 or overlap_s >= window_s:
        raise UsageError(
            "--window-overlap must satisfy 0 <= overlap < window",
            details={"window_seconds": window_s, "overlap_seconds": overlap_s},
        )
    if not math.isfinite(float(search_s)) or search_s < 0:
        raise UsageError("--window-search must be >= 0")


def _lowest_energy_cut(
    start_sample: int,
    target_sample: int,
    sr: int,
    search_s: float,
    energy_frames: Optional[Sequence[float]],
    total_samples: int,
) -> int:
    if not energy_frames or search_s <= 0:
        return target_sample
    lower = max(start_sample + 1, target_sample - int(round(search_s * sr)))
    upper = target_sample
    if upper <= lower:
        return target_sample
    frames = len(energy_frames)
    if frames <= 0 or total_samples <= 0:
        return target_sample
    best_sample = target_sample
    best_value = math.inf
    for sample in range(lower, upper + 1):
        frame = min(frames - 1, int(sample * frames / total_samples))
        value = float(energy_frames[frame])
        if value < best_value:
            best_value = value
            best_sample = sample
    return best_sample


def plan_windows(
    total_samples: int,
    sr: int,
    window_s: float = MAX_WINDOW_SECONDS,
    overlap_s: float = DEFAULT_OVERLAP_SECONDS,
    search_s: float = DEFAULT_SEARCH_SECONDS,
    energy_frames: Optional[Sequence[float]] = None,
    max_window_s: float = MAX_WINDOW_SECONDS,
) -> List[Window]:
    """Plan contiguous core windows with a small audio lookback overlap.

    Sample positions are authoritative. Cores tile the input without gaps;
    audio windows may include ``overlap_s`` before the core but never exceed
    ``window_s``. ``max_window_s`` is the caller's hard limit: 180s for
    forced-alignment windows, up to 1200s for plain-text chunking.
    """
    _check_params(window_s, overlap_s, search_s, max_window_s)
    if not isinstance(sr, int) or sr <= 0:
        raise UsageError("sample rate must be a positive integer")
    total_samples = int(total_samples)
    if total_samples < 0:
        raise UsageError("total_samples must be >= 0")
    if total_samples == 0:
        return []

    window_samples = int(round(window_s * sr))
    # A window that fits within the limit stays a single window: splitting it
    # to leave room for the lookback overlap would create a short trailing
    # window and break the "one window per <= window_s input" contract.
    if total_samples <= window_samples:
        return [Window(0, 0, total_samples, 0, total_samples)]

    overlap_samples = int(round(overlap_s * sr))
    core_samples = window_samples - overlap_samples
    slack_samples = int(round(MIN_CORE_SLACK_SECONDS * sr))

    windows: List[Window] = []
    core_start = 0
    index = 0
    while core_start < total_samples:
        remaining = total_samples - core_start
        if remaining <= core_samples:
            core_end = total_samples
        else:
            target = core_start + core_samples
            lower = max(core_start + 1, target - slack_samples)
            core_end = _lowest_energy_cut(
                core_start, target, sr, search_s, energy_frames, total_samples
            )
            core_end = max(lower, min(core_end, target))
        audio_start = max(0, core_start - overlap_samples)
        audio_end = core_end
        if audio_end - audio_start > window_samples:
            # Energy cuts only move earlier, so this is defensive only.
            audio_start = audio_end - window_samples
        windows.append(
            Window(
                index=index,
                audio_start_sample=audio_start,
                audio_end_sample=audio_end,
                core_start_sample=core_start,
                core_end_sample=core_end,
            )
        )
        core_start = core_end
        index += 1
    return windows


def normalize_text(text: str) -> str:
    """NFKC, drop whitespace/punctuation, casefold (coverage-style)."""
    normalized = unicodedata.normalize("NFKC", text or "")
    out: List[str] = []
    for ch in normalized:
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("Z"):
            continue
        out.append(ch.casefold())
    return "".join(out)


def _normalized_with_offsets(text: str) -> tuple[str, List[int]]:
    """Return the normalized string plus normalized-index -> original-index."""
    text = unicodedata.normalize("NFKC", text or "")
    chars: List[str] = []
    offsets: List[int] = []
    for index, ch in enumerate(text):
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("Z"):
            continue
        chars.append(ch.casefold())
        offsets.append(index)
    return "".join(chars), offsets


def attribute_reference(
    ref_text: str, window_texts: Sequence[str]
) -> List[TextSpan]:
    """Monotonically assign reference character ranges to windows.

    Repeated phrases are consumed in order: a later window can never re-claim
    characters already assigned to an earlier window.
    """
    norm_ref, offsets = _normalized_with_offsets(ref_text)
    spans: List[TextSpan] = []
    cursor = 0
    for index, raw in enumerate(window_texts):
        window_norm = normalize_text(raw)
        if not window_norm or cursor >= len(norm_ref):
            spans.append(TextSpan(index, _offset_at(offsets, cursor, len(ref_text)),
                                  _offset_at(offsets, cursor, len(ref_text)), 0.0))
            continue
        # Match against a bounded candidate slice so a long remaining
        # reference does not let the longest block jump far ahead.
        window_len = len(window_norm)
        remaining = len(norm_ref) - cursor
        k = min(
            max(int(window_len * CANDIDATE_MULTIPLIER) + CANDIDATE_PAD, window_len),
            remaining,
        )
        matcher = difflib.SequenceMatcher(
            None, norm_ref[cursor:cursor + k], window_norm, autojunk=False
        )
        blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
        if not blocks:
            position = _offset_at(offsets, cursor, len(ref_text))
            spans.append(TextSpan(index, position, position, 0.0))
            continue
        # The window text and its true reference slice are the same speech, so
        # their matching blocks lie on one diagonal (a - b constant). Repeated
        # phrases elsewhere in the reference sit on other diagonals: including
        # them lets a window swallow later windows, so pick the single diagonal
        # that explains the most characters and ignore everything outside it.
        band = max(MATCH_BAND_MIN, int(window_len * MATCH_BAND_RATIO))
        best_chain: List[Any] = []
        best_score = -1
        for anchor in blocks:
            delta = anchor.a - anchor.b
            chain: List[Any] = []
            window_end = 0
            for block in blocks:
                if abs((block.a - block.b) - delta) > band:
                    continue
                # Repeated reference phrases map to the same window offset;
                # only the occurrence that advances the window belongs here.
                if block.b < window_end:
                    continue
                chain.append(block)
                window_end = block.b + block.size
            matched = sum(block.size for block in chain)
            # Prefer dense, nearby diagonals: a band that matches a few
            # characters and then jumps over a long passage explains the window
            # worse than a compact band, even when the jump lands on a longer
            # repeated tail. Windows tile the audio, so the true slice also
            # starts near the cursor; distance is weak evidence, not a veto.
            span = chain[-1].a + chain[-1].size - chain[0].a
            excess = max(0, span - matched)
            score = matched / (
                1.0 + (excess + chain[0].a) / max(1, window_len)
            )
            if score > best_score or (
                score == best_score
                and chain
                and best_chain
                and chain[0].a < best_chain[0].a
            ):
                best_score = score
                best_chain = chain
        if not best_chain:
            position = _offset_at(offsets, cursor, len(ref_text))
            spans.append(TextSpan(index, position, position, 0.0))
            continue
        # Own everything between the first and last in-band matching block.
        # Small substitutions/insertions inside that range belong to this
        # window; the cap keeps a pathological chain from spanning far more
        # text than the window itself.
        max_span = int(window_len * MAX_SPAN_MULTIPLIER) + MAX_SPAN_PAD
        ref_start_norm = cursor + best_chain[0].a
        ref_end_norm = cursor + best_chain[-1].a + best_chain[-1].size
        ref_end_norm = min(ref_end_norm, ref_start_norm + max_span)
        chain = [block for block in best_chain if cursor + block.a < ref_end_norm]
        if not chain:
            chain = best_chain[:1]
        ref_end_norm = min(ref_end_norm, cursor + chain[-1].a + chain[-1].size)
        matched = sum(block.size for block in chain)
        ratio = min(1.0, matched / max(1, window_len))
        tail_start = int(window_len * TAIL_START_RATIO)
        tail_len = window_len - tail_start
        tail_matched = 0
        if tail_len > 0:
            covered = set()
            for block in chain:
                covered.update(range(block.b, block.b + block.size))
            tail_matched = sum(
                1 for position in range(tail_start, window_len) if position in covered
            )
        tail_ratio = (
            tail_matched / tail_len if tail_len >= MIN_TAIL_CHARS else 1.0
        )
        start_char = offsets[ref_start_norm]
        end_char = offsets[ref_end_norm - 1] + 1
        spans.append(
            TextSpan(
                index,
                start_char,
                end_char,
                round(ratio, 4),
                ref_text[start_char:end_char],
                round(tail_ratio, 4),
            )
        )
        cursor = ref_end_norm
    return spans


def _offset_at(offsets: Sequence[int], norm_index: int, text_length: int) -> int:
    if norm_index < len(offsets):
        return offsets[norm_index]
    return int(text_length)


def merge_items(items: Sequence[Any], *, duration_s: float) -> List[Any]:
    """Sort, enforce monotonic starts and clamp/repair ends to ``duration_s``."""
    duration = max(0.0, float(duration_s or 0.0))
    ordered = sorted(
        (item for item in items if str(getattr(item, "text", "") or "").strip()),
        key=lambda item: float(getattr(item, "start", 0.0)),
    )
    out: List[Any] = []
    previous_end = 0.0
    for item in ordered:
        start = max(float(getattr(item, "start", 0.0)), previous_end)
        end = float(getattr(item, "end", start))
        if start >= duration:
            continue
        end = min(end, duration)
        if end <= start:
            end = min(start + MERGE_EPSILON, duration)
        if end <= start:
            continue
        out.append(replace(item, start=start, end=end))
        previous_end = end
    return out


def coverage(ref_text: str, items: Sequence[Any]) -> float:
    reference = normalize_text(ref_text)
    hypothesis = normalize_text("".join(str(getattr(item, "text", "")) for item in items))
    if not reference:
        return 1.0
    if not hypothesis:
        return 0.0
    matcher = difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return round(matched / len(reference), 4)
