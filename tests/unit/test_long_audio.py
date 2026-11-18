"""N4: long-audio pure functions."""

from __future__ import annotations

import pytest

from qasr_cli.backends import Segment
from qasr_cli.errors import UsageError
from qasr_cli.long_audio import (
    TextSpan,
    attribute_reference,
    coverage,
    merge_items,
    normalize_text,
    plan_windows,
)

SR = 16000


def test_181_seconds_plans_two_windows_with_contiguous_cores():
    windows = plan_windows(181 * SR, SR)
    assert len(windows) == 2
    first, last = windows
    assert first.core_start_sample == 0
    assert first.audio_start_sample == 0
    assert last.core_end_sample == 181 * SR
    assert first.core_end_sample == last.core_start_sample
    assert last.audio_start_sample == last.core_start_sample - 3 * SR
    for window in windows:
        assert window.audio_end_sample - window.audio_start_sample <= 180 * SR


def test_exactly_window_length_is_a_single_full_window():
    windows = plan_windows(180 * SR, SR)
    assert len(windows) == 1
    window = windows[0]
    assert window.audio_start_sample == 0
    assert window.audio_end_sample == 180 * SR
    assert window.core_start_sample == 0
    assert window.core_end_sample == 180 * SR


def test_exact_plain_chunk_limit_is_a_single_window():
    windows = plan_windows(270 * SR, SR, window_s=270, max_window_s=270)
    assert len(windows) == 1
    assert windows[0].core_end_sample == 270 * SR


def test_half_second_over_window_length_plans_two_windows():
    windows = plan_windows(int(180.5 * SR), SR)
    assert len(windows) == 2
    assert windows[0].core_end_sample == windows[1].core_start_sample
    assert windows[-1].core_end_sample == int(180.5 * SR)


def test_one_hour_window_count_and_bounds():
    total = 3600 * SR
    windows = plan_windows(total, SR)
    assert len(windows) == 21  # ceil(3600 / 177)
    assert windows[0].core_start_sample == 0
    assert windows[-1].core_end_sample == total
    for window in windows:
        core_len = window.core_end_sample - window.core_start_sample
        audio_len = window.audio_end_sample - window.audio_start_sample
        assert core_len > 0
        assert audio_len <= 180 * SR
        assert window.audio_start_sample >= 0
        assert window.audio_end_sample <= total
    for previous, current in zip(windows, windows[1:]):
        assert current.core_start_sample == previous.core_end_sample


def test_short_audio_is_one_window():
    windows = plan_windows(10 * SR, SR, window_s=180, overlap_s=3)
    assert len(windows) == 1
    assert windows[0].audio_start_sample == 0
    assert windows[0].audio_end_sample == 10 * SR


def test_zero_overlap_is_allowed():
    windows = plan_windows(400 * SR, SR, window_s=180, overlap_s=0)
    assert len(windows) == 3
    for window in windows:
        assert window.audio_start_sample == window.core_start_sample


def test_invalid_window_parameters_raise_usage():
    with pytest.raises(UsageError):
        plan_windows(SR, SR, window_s=181)
    with pytest.raises(UsageError):
        plan_windows(SR, SR, window_s=0)
    with pytest.raises(UsageError):
        plan_windows(SR, SR, window_s=180, overlap_s=180)
    with pytest.raises(UsageError):
        plan_windows(SR, SR, window_s=180, overlap_s=-1)
    with pytest.raises(UsageError):
        plan_windows(SR, SR, window_s=180, search_s=-1)


def test_energy_cut_moves_within_slack_and_keeps_bounds():
    total = 400 * SR
    frames = 4000
    energies = [10.0] * frames
    target = 177 * SR
    # Quiet frame 1.5s before the fixed cut.
    quiet_sample = target - int(1.5 * SR)
    energies[int(quiet_sample * frames / total)] = 0.01
    windows = plan_windows(total, SR, energy_frames=energies)
    first = windows[0]
    assert first.core_end_sample <= target
    assert first.core_end_sample >= target - 2 * SR
    assert first.core_end_sample == pytest.approx(quiet_sample, abs=SR // frames + 2)
    assert first.audio_end_sample - first.audio_start_sample <= 180 * SR


def test_empty_audio_has_no_windows():
    assert plan_windows(0, SR) == []


def test_attribute_reference_is_monotonic_for_repeated_phrases():
    text = "你好你好你好"
    spans = attribute_reference(text, ["你好", "你好", "你好"])
    assert len(spans) == 3
    assert all(span.match_ratio == 1.0 for span in spans)
    assert [span.start_char for span in spans] == [0, 2, 4]
    assert [span.end_char for span in spans] == [2, 4, 6]
    assert [span.text for span in spans] == ["你好", "你好", "你好"]


def test_attribute_reference_does_not_repeat_when_window_is_empty():
    spans = attribute_reference("abcdef", ["abc", "", "def"])
    assert spans[0].text == "abc"
    assert spans[1] == TextSpan(1, 3, 3, 0.0)
    assert spans[2].text == "def"


def test_attribute_reference_weak_match_is_flagged():
    spans = attribute_reference("hello world", ["completely different"])
    assert spans[0].is_weak
    assert spans[0].match_ratio < 0.5


def test_merge_items_is_strictly_monotonic_and_clamped():
    items = [
        Segment("a", 0.0, 1.0, "word"),
        Segment("b", 0.5, 2.0, "word"),
        Segment("c", 1.9, 1.95, "word"),
        Segment("d", 2.0, 3.5, "word"),
        Segment("", 2.1, 2.2, "word"),
        Segment("late", 4.0, 5.0, "word"),
    ]
    merged = merge_items(items, duration_s=3.0)
    assert [item.text for item in merged] == ["a", "b", "c", "d"]
    previous_end = 0.0
    for item in merged:
        assert item.start >= previous_end
        assert item.end > item.start
        assert item.end <= 3.0
        previous_end = item.end
    assert merged[1].start == 1.0
    assert merged[2].start == 2.0
    assert merged[2].end == 2.001
    assert merged[3].end == 3.0


def test_merge_items_drops_when_duration_is_insufficient():
    assert merge_items([Segment("x", 1.0, 2.0)], duration_s=0.0) == []


def test_normalize_and_coverage():
    assert normalize_text(" Hello, 世界！ ") == "hello世界"
    items = [Segment("你好", 0.0, 0.5), Segment("世界", 0.5, 1.0)]
    assert coverage("你好，世界！", items) == 1.0
    assert coverage("你好世界", []) == 0.0
    assert coverage("", []) == 1.0


def test_plain_chunking_can_exceed_alignment_window_cap():
    windows = plan_windows(600 * SR, SR, window_s=270, max_window_s=270)
    assert len(windows) == 3
    for window in windows:
        assert window.audio_end_sample - window.audio_start_sample <= 270 * SR
    with pytest.raises(UsageError):
        plan_windows(600 * SR, SR, window_s=270)


def test_attribute_reference_tolerates_small_differences():
    ref = "This is a LibriVox recording of a public domain book read by a volunteer"
    window = "this is a librivox recording of a public domain books read by volunteers"
    span = attribute_reference(ref, [window])[0]
    assert span.match_ratio > 0.8
    assert span.text.startswith("This is a LibriVox")
    assert span.text.endswith("volunteer")


def test_attribute_reference_keeps_middle_substitutions():
    ref = "alpha beta gamma delta epsilon zeta"
    window = "alpha beta omega delta epsilon"
    span = attribute_reference(ref, [window])[0]
    assert span.text == "alpha beta gamma delta epsilon"
    assert span.match_ratio > 0.8


def test_attribute_reference_ignores_leading_and_trailing_window_noise():
    ref = "alpha beta gamma delta"
    spans = attribute_reference(ref, ["uh alpha beta", "gamma delta um"])
    assert spans[0].text == "alpha beta"
    assert spans[1].text == "gamma delta"
    assert spans[0].match_ratio > 0.8
    assert spans[1].match_ratio > 0.8


def test_attribute_reference_covers_window_spanning_two_sentences():
    ref = "the quick brown fox jumps over the lazy dog"
    window = "the quick brown fox jumps over the lazy dog"
    span = attribute_reference(ref, [window])[0]
    assert span.start_char == 0
    assert span.end_char == len(ref)
    assert span.match_ratio == 1.0
