"""P2 S2: stdin PCM framing, downmix, residual bytes and EOF."""

from __future__ import annotations

import time

import pytest
from helpers import assert_exit, assert_no_traceback, parse_jsonl, pcm_s16le

pytestmark = pytest.mark.stream


def _stream(qasr, fake_stream, fake_model, *args, stdin=None, timeout=90):
    return qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm",
        *args,
        env=fake_stream.env(),
        stdin=stdin,
        timeout=timeout,
    )


def _jsonl(qasr, fake_stream, fake_model, *args, stdin=None):
    proc = _stream(qasr, fake_stream, fake_model, *args, stdin=stdin)
    assert_exit(proc, 0, context="stdin stream")
    values = parse_jsonl(proc)
    return proc, values


def test_empty_stdin_emits_one_final_snapshot(qasr, fake_stream, fake_model):
    proc, values = _jsonl(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--emit", "jsonl",
        stdin=b"",
    )
    assert len(values) == 1, values
    final = values[0]
    assert final["seq"] == 0
    assert final["is_final"] is True
    assert final["text"] == ""
    assert final.get("language") is None
    assert final["audio_s"] == pytest.approx(0.0)
    assert "FAKE-VLLM" not in proc.stdout


def test_one_second_mono_becomes_one_final_snapshot(
    qasr, fake_stream, fake_model, one_second_pcm
):
    proc, values = _jsonl(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--emit", "jsonl",
        stdin=one_second_pcm,
    )
    assert len(values) == 1, values
    assert values[0]["seq"] == 0
    assert values[0]["is_final"] is True
    assert values[0]["audio_s"] == pytest.approx(1.0, abs=0.02)
    feeds = fake_stream.events("feed")
    assert feeds and feeds[0]["n"] == 16_000
    finishes = fake_stream.events("finish")
    assert finishes and finishes[-1]["tail"] == 16_000


def test_five_seconds_with_two_second_chunks_has_cumulative_snapshots(
    qasr, fake_stream, fake_model
):
    pcm = pcm_s16le(5 * 16_000)
    _proc, values = _jsonl(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "2000", "--emit", "jsonl",
        stdin=pcm,
    )
    assert len(values) >= 3, values
    assert [item["seq"] for item in values] == list(range(len(values)))
    assert sum(1 for item in values if item["is_final"]) == 1
    assert values[-1]["is_final"] is True
    assert values[-1]["audio_s"] == pytest.approx(5.0, abs=0.02)
    audio = [item["audio_s"] for item in values]
    assert all(b >= a for a, b in zip(audio, audio[1:])), audio
    chunks = fake_stream.events("chunk")
    assert len(chunks) >= 2, chunks
    for before, after in zip(chunks, chunks[1:]):
        assert after["text"].startswith(before["text"]), (before, after)


def test_exact_chunk_multiple_still_emits_empty_tail_final(
    qasr, fake_stream, fake_model
):
    pcm = pcm_s16le(4 * 16_000)
    _proc, values = _jsonl(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "2000", "--emit", "jsonl",
        stdin=pcm,
    )
    assert sum(1 for item in values if item["is_final"]) == 1
    assert values[-1]["audio_s"] == pytest.approx(4.0, abs=0.02)
    finish = fake_stream.events("finish")[-1]
    assert finish["tail"] == 0
    assert len(values) >= 3, values
    assert values[-1]["text"] == values[-2]["text"]


def test_partial_tail_changes_final_snapshot(
    qasr, fake_stream, fake_model
):
    pcm = pcm_s16le(int(4.5 * 16_000))
    _proc, values = _jsonl(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "2000", "--emit", "jsonl",
        stdin=pcm,
    )
    assert values[-1]["audio_s"] == pytest.approx(4.5, abs=0.02)
    finish = fake_stream.events("finish")[-1]
    assert finish["tail"] == 8_000
    assert values[-1]["text"] != values[-2]["text"]
    # P2-Q10: the partial tail only shows up in the final snapshot; the last
    # non-final snapshot reports full chunks only.
    non_final = [item for item in values if not item["is_final"]]
    assert len(non_final) == 2, values
    assert non_final[-1]["audio_s"] == pytest.approx(4.0, abs=0.02)


def test_odd_byte_tail_is_dropped_with_warning(
    qasr, fake_stream, fake_model, one_second_pcm
):
    proc, values = _jsonl(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--emit", "jsonl",
        stdin=one_second_pcm + b"\x7f",
    )
    assert values[-1]["audio_s"] == pytest.approx(1.0, abs=0.02)
    assert "discard" in proc.stderr.lower() or "byte" in proc.stderr.lower()
    assert_no_traceback(proc)


def test_stereo_incomplete_frame_is_dropped_with_warning(
    qasr, fake_stream, fake_model
):
    pcm = pcm_s16le(16_000, channels=2, left=1000, right=3000)
    proc, values = _jsonl(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--channels", "2", "--emit", "jsonl",
        stdin=pcm + b"\x01\x02",
    )
    assert values[-1]["audio_s"] == pytest.approx(1.0, abs=0.02)
    assert "discard" in proc.stderr.lower() or "frame" in proc.stderr.lower()
    assert_no_traceback(proc)


def test_stereo_residual_uses_frame_bytes_not_single_sample(
    qasr, fake_stream, fake_model
):
    """P2-Q6: a stereo frame is 4 bytes; 6 trailing bytes drop 2, keep 1 frame."""
    pcm = pcm_s16le(16_000, channels=2, left=1000, right=3000) + b"\x01\x02\x03\x04\x05\x06"
    proc, values = _jsonl(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--channels", "2", "--emit", "jsonl",
        stdin=pcm,
    )
    # The 6-byte tail contains one complete frame (2 samples) plus 2 residual
    # bytes; only the residual 2 bytes are discarded.
    assert values[-1]["audio_s"] == pytest.approx(1.0 + 1 / 16_000, abs=0.01)
    assert "discard" in proc.stderr.lower() or "frame" in proc.stderr.lower()
    assert_no_traceback(proc)


def test_stereo_is_downmixed_by_frame_average(
    qasr, fake_stream, fake_model
):
    pcm = pcm_s16le(4_000, channels=2, left=1000, right=3000)
    _proc, values = _jsonl(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--channels", "2", "--chunk-ms", "100",
        "--emit", "jsonl",
        stdin=pcm,
    )
    assert values[-1]["audio_s"] == pytest.approx(0.25, abs=0.02)
    feeds = fake_stream.events("feed")
    assert feeds
    first = feeds[0]["first"][0]
    assert first == pytest.approx(2000 / 32768, abs=2e-4), first


def test_audio_s_is_rounded_to_three_decimals(
    qasr, fake_stream, fake_model
):
    """P2-Q10: audio_s is cumulative samples / 16000 rounded to 3 decimals."""
    _proc, values = _jsonl(
        qasr, fake_stream, fake_model,
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "100", "--emit", "jsonl",
        stdin=pcm_s16le(1_234),
    )
    for item in values:
        assert item["audio_s"] == round(item["audio_s"], 3), item


def test_split_stdin_writes_are_reassembled(
    qasr, fake_stream, fake_model
):
    pcm = pcm_s16le(16_000)
    proc = qasr.popen(
        "--model", fake_model,
        "stream", "--backend", "vllm",
        "--input", "stdin", "--pcm", "s16le",
        "--chunk-ms", "100", "--emit", "jsonl",
        env=fake_stream.env(),
        binary=True,
    )
    assert proc.stdin is not None
    for start in range(0, len(pcm), 4000):
        proc.stdin.write(pcm[start:start + 4000])
        proc.stdin.flush()
        time.sleep(0.02)
    stdout, stderr = proc.communicate(timeout=60)
    assert proc.returncode == 0, (stdout, stderr)
    values = parse_jsonl(
        type("P", (), {"stdout": stdout.decode(), "stderr": stderr.decode(),
                       "returncode": proc.returncode})()
    )
    assert values[-1]["is_final"] is True
    assert values[-1]["audio_s"] == pytest.approx(1.0, abs=0.02)
    assert fake_stream.events("finish")[-1]["audio_accum"] == 16_000


def test_file_input_uses_same_snapshot_contract(
    qasr, fake_stream, fake_model, five_second_wav
):
    proc = qasr.run(
        "--model", fake_model,
        "stream", "--backend", "vllm",
        "--input", five_second_wav,
        "--chunk-ms", "2000", "--emit", "jsonl",
        env=fake_stream.env(),
    )
    assert_exit(proc, 0, context="file stream")
    values = parse_jsonl(proc)
    assert values[-1]["is_final"] is True
    assert values[-1]["audio_s"] == pytest.approx(5.0, abs=0.05)
    assert sum(1 for item in values if item["is_final"]) == 1
