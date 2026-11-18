"""ADV-02/08/11/12: adversarial audio, duration edges, and no-runtime planning."""

from __future__ import annotations

import os

import pytest
from conftest import RUNTIME_MODULES
from helpers import assert_error, assert_exit, empty_wav, make_wav, parse_json, zero_byte


def test_directory_named_like_audio_is_rejected(qasr):
    path = qasr.work / "fake.wav"
    path.mkdir()
    (path / "inside.txt").write_text("not audio", encoding="utf-8")
    proc = qasr.run("--json", "transcribe", path)
    assert_exit(proc, 4, context="directory named .wav")
    assert_error(parse_json(proc), code=4)


def test_unreadable_file_is_input_error(qasr):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can read chmod 000 files")
    path = make_wav(qasr.work / "locked.wav", seconds=0.2)
    path.chmod(0)
    try:
        proc = qasr.run("--json", "transcribe", path)
    finally:
        path.chmod(0o600)
    assert_exit(proc, 4, context="unreadable audio")
    assert_error(parse_json(proc), code=4)


def test_empty_wav_succeeds_without_model(qasr):
    audio = empty_wav(qasr.work / "empty.wav")
    proc = qasr.run(
        "--json",
        "--model",
        "/nonexistent/model",
        "transcribe",
        audio,
    )
    # A valid 0-sample WAV has nothing to infer; it is a deliberate success
    # even when the model path is unusable.
    assert_exit(proc, 0, context="empty WAV short-circuit")
    payload = parse_json(proc)
    result = payload.get("result") or {}
    assert result.get("text") == "", payload
    assert result.get("language") is None, payload
    warnings = result.get("warnings") or payload.get("warnings") or []
    assert any("empty" in str(item).lower() for item in warnings), (
        f"empty audio should be reported in warnings: {payload!r}"
    )


def test_zero_byte_file_is_corrupt_not_empty_audio(qasr):
    audio = zero_byte(qasr.work / "zero.wav")
    proc = qasr.run(
        "--json",
        "--model",
        "/nonexistent/model",
        "transcribe",
        audio,
    )
    assert_exit(proc, 4, context="zero-byte WAV")
    assert_error(parse_json(proc), code=4)


def test_empty_wav_with_timestamps_needs_no_aligner(qasr):
    """Zero-sample audio has nothing to align, so a missing aligner must not fail it."""
    audio = empty_wav(qasr.work / "empty-ts.wav")
    proc = qasr.run(
        "--json",
        "--model",
        "/nonexistent/model",
        "--aligner",
        "/nonexistent/aligner",
        "transcribe",
        audio,
        "--timestamps",
        "--language",
        "zh",
    )
    assert_exit(proc, 0, context="empty WAV + timestamps")
    result = parse_json(proc).get("result") or {}
    assert result.get("text") == ""
    assert result.get("language") is None


@pytest.mark.parametrize("seconds,expected", [(180.0, 3), (180.5, 3)])
def test_align_180_second_boundary(qasr, seconds, expected):
    audio = make_wav(qasr.work / f"len-{seconds}.wav", seconds=seconds, silent=True)
    proc = qasr.run(
        "--json",
        "--model",
        "/nonexistent/model",
        "align",
        audio,
        "--text",
        "hello",
        "--language",
        "en",
        timeout=180,
    )
    assert_exit(proc, expected, context=f"align {seconds}s")
    # P1 removed the 180s hard cap; both lengths now reach model resolution
    # (exit 3 with the deliberately missing model). Exact-180s single-window
    # behaviour is asserted in review_tests_p1/test_n4_long_audio.py.


def test_dry_run_never_imports_runtime(qasr, import_blocker):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    out = qasr.work / "out"
    proc = qasr.run(
        "--json",
        "--dry-run",
        "transcribe",
        audio,
        "-f",
        "txt,srt",
        "--timestamps",
        "--language",
        "zh",
        "-o",
        out,
        env=import_blocker.env(),
    )
    assert_exit(proc, 0, context="dry-run without runtime")
    blocked = [name for name in import_blocker.attempts() if name.split(".")[0] in RUNTIME_MODULES]
    assert not blocked, f"dry-run imported runtime modules: {blocked[:10]}"


def test_dry_run_offline_never_imports_runtime(qasr, import_blocker):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = qasr.run(
        "--json",
        "--dry-run",
        "transcribe",
        audio,
        env=import_blocker.env(QASR_OFFLINE="1"),
    )
    assert_exit(proc, 0)
    blocked = [name for name in import_blocker.attempts() if name.split(".")[0] in RUNTIME_MODULES]
    assert not blocked, f"offline dry-run imported runtime modules: {blocked[:10]}"
