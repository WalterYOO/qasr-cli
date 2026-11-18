import pytest

from qasr_cli.backends import Segment
from qasr_cli.subtitles import (
    Cue,
    build_cues,
    display_width,
    join_tokens,
    parse_srt,
    render_srt,
    render_vtt,
    to_ms,
)


def seg(text, start, end, kind="word"):
    return Segment(text=text, start=start, end=end, kind=kind)


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.0004, 0), (0.0005, 1), (1.2345, 1235), (1.2344, 1234), (0.0, 0)],
)
def test_half_up_rounding(seconds, expected):
    assert to_ms(seconds) == expected


def test_display_width_cjk_counts_two():
    assert display_width("ab") == 2
    assert display_width("中文") == 4
    assert display_width("a中") == 3


def test_join_tokens_cjk_and_latin():
    assert join_tokens(["你", "好"]) == "你好"
    assert join_tokens(["hello", "world"]) == "hello world"
    assert join_tokens(["hello", "世界"]) == "hello世界"


def test_build_cues_splits_on_pause():
    segments = [seg("你好", 0.0, 0.5), seg("再见", 2.0, 2.5)]
    cues = build_cues(segments, audio_duration_s=3.0)
    assert len(cues) == 2
    assert cues[0].start_ms == 0
    assert cues[1].start_ms == 2000


def test_build_cues_splits_on_punctuation():
    segments = [seg("你好。", 0.0, 0.5), seg("再见", 0.6, 1.0)]
    cues = build_cues(segments, audio_duration_s=1.0)
    assert len(cues) == 2
    assert cues[0].text == "你好。"


def test_cue_max_duration_and_width():
    segments = [seg("字" * 30, 0.0, 1.0), seg("字" * 30, 1.0, 2.0)]
    cues = build_cues(segments, audio_duration_s=2.0)
    assert all(display_width(cue.text) <= 42 for cue in cues)
    assert all(cue.end_ms - cue.start_ms <= 7000 for cue in cues)


def test_cues_monotonic_and_min_duration():
    segments = [seg("a", 1.0, 1.0), seg("b", 0.5, 0.6)]
    cues = build_cues(segments, audio_duration_s=2.0)
    prev_end = 0
    for cue in cues:
        assert cue.start_ms >= prev_end
        assert cue.end_ms >= cue.start_ms + 1
        prev_end = cue.end_ms


def test_cues_capped_at_audio_duration():
    cues = build_cues([seg("hello", 0.0, 5.0)], audio_duration_s=2.0)
    assert cues[0].end_ms <= 2000


def test_render_srt_format():
    text = render_srt([Cue(0, 1500, "你好"), Cue(1_500, 2_000, "再见")])
    lines = text.strip().splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:01,500"
    assert lines[4] == "2"
    assert lines[5] == "00:00:01,500 --> 00:00:02,000"


def test_render_vtt_format():
    text = render_vtt([Cue(0, 1500, "你好")])
    assert text.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.500" in text


def test_parse_srt_roundtrip():
    original = render_srt([Cue(0, 1500, "你好"), Cue(1500, 2000, "再见")])
    cues = parse_srt(original)
    assert [cue.text for cue in cues] == ["你好", "再见"]
    assert cues[1].start_ms == 1500
