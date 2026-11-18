"""TS-01..09 and AL-01..15: timestamp gates, 180s boundary, align inputs."""

from __future__ import annotations

import pytest
from helpers import assert_error, assert_exit, make_wav, parse_json


def _transcribe(qasr, *args, timeout=120):
    return qasr.run("--json", "transcribe", *args, timeout=timeout)


def _align(qasr, *args, model=None, timeout=120):
    prefix = ["--json"]
    if model:
        prefix.extend(["--model", model])
    return qasr.run(*prefix, "align", *args, timeout=timeout)


def test_timestamps_default_auto_is_rejected(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _transcribe(qasr, audio, "--timestamps")
    assert_exit(proc, 4, context="--timestamps with default auto")
    error = assert_error(parse_json(proc), code=4)
    text = (error.get("message", "") + " " + str(error.get("hint", ""))).lower()
    assert "language" in text
    assert "zh" in text or "chinese" in text, f"hint should list alignment languages: {error!r}"


def test_timestamps_explicit_auto_is_rejected(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _transcribe(qasr, audio, "--timestamps", "--language", "auto")
    assert_exit(proc, 4, context="--timestamps --language auto")
    assert_error(parse_json(proc), code=4, kind="capability")


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_subtitle_formats_require_timestamps(qasr, fmt):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _transcribe(qasr, audio, "-f", fmt)
    assert_exit(proc, 4, context=f"-f {fmt} without timestamps")
    error = assert_error(parse_json(proc), code=4)
    text = (error.get("message", "") + " " + str(error.get("hint", ""))).lower()
    assert "timestamp" in text or "align" in text, f"unclear capability error: {error!r}"


def test_invalid_granularity_is_usage_error(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _transcribe(qasr, audio, "--timestamps", "--language", "zh", "--granularity", "token")
    assert_exit(proc, 2, context="invalid granularity")
    assert_error(parse_json(proc), code=2)


@pytest.mark.parametrize("language", ["zh", "en", "yue", "ja"])
def test_timestamps_with_alignable_language_plans(qasr, language):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = qasr.run(
        "--json",
        "--dry-run",
        "transcribe",
        audio,
        "--timestamps",
        "--language",
        language,
    )
    assert_exit(proc, 0, context=f"timestamps {language}")


def test_timestamps_with_tha_is_rejected(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = qasr.run(
        "--json",
        "--dry-run",
        "transcribe",
        audio,
        "--timestamps",
        "--language",
        "th",
    )
    assert_exit(proc, 4, context="timestamps with Thai")
    assert_error(parse_json(proc), code=4)


def test_align_requires_text_source(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _align(qasr, audio, "--language", "zh")
    assert_exit(proc, 2, context="align without text")


def test_align_text_sources_are_mutually_exclusive(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    text_file = qasr.work / "text.txt"
    text_file.write_text("你好", encoding="utf-8")
    proc = _align(qasr, audio, "--text", "你好", "--text-file", text_file, "--language", "zh")
    assert_exit(proc, 2, context="align text + text-file")


def test_align_requires_language(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _align(qasr, audio, "--text", "你好")
    assert_exit(proc, 2, context="align without language")


def test_align_rejects_unalignable_language(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _align(qasr, audio, "--text", "hello", "--language", "th")
    assert_exit(proc, 4, context="align Thai")
    assert_error(parse_json(proc), code=4)


def test_align_missing_text_file(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _align(qasr, audio, "--text-file", qasr.work / "missing.txt", "--language", "zh")
    assert_exit(proc, 4, context="align missing text file")
    assert_error(parse_json(proc), code=4)


def test_align_windows_audio_over_180s(qasr):
    audio = make_wav(qasr.work / "long.wav", seconds=181, silent=True)
    proc = qasr.run(
        "--json", "--dry-run", "align", audio,
        "--text", "hello", "--language", "en",
    )
    assert_exit(proc, 0, context="align 181s dry-run")
    plan = parse_json(proc).get("plan") or {}
    windows = plan.get("windows") or []
    # P1 windows anything over 180s instead of rejecting it.
    assert len(windows) >= 2, plan
    assert abs(float(windows[-1]["core_end_s"]) - 181.0) < 0.5, windows[-1]


def test_align_accepts_audio_under_180s(qasr):
    audio = make_wav(qasr.work / "long.wav", seconds=179.9, silent=True)
    proc = _align(
        qasr,
        audio,
        "--text",
        "hello",
        "--language",
        "en",
        model="/nonexistent/model",
        timeout=180,
    )
    # Passing the duration gate is proven by reaching model resolution.
    assert_exit(proc, 3, context="align 179.9s")


@pytest.mark.parametrize("value", ["1.5", "-0.1", "abc"])
def test_invalid_coverage_min(qasr, value):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _align(qasr, audio, "--text", "hello", "--language", "en", "--coverage-min", value)
    assert_exit(proc, 2, context=f"--coverage-min {value}")


def test_align_server_is_a_capability_error(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _align(
        qasr,
        audio,
        "--text",
        "hello",
        "--language",
        "en",
        "--server",
        "http://127.0.0.1:9",
    )
    assert_exit(proc, 4, context="align --server")
    payload = parse_json(proc)
    assert payload["error"]["kind"] == "capability", payload


def test_align_multiple_formats_require_out_dir(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = _align(qasr, audio, "--text", "hello", "--language", "en", "-f", "json,srt")
    assert_exit(proc, 2, context="align multi-format to stdout")


def test_align_malformed_srt(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    srt = qasr.work / "bad.srt"
    srt.write_text("1\nnot a timestamp\nhello\n", encoding="utf-8")
    proc = _align(qasr, audio, "--srt", srt, "--language", "en")
    assert_exit(proc, {2, 4}, context="malformed SRT")
    assert "Traceback" not in proc.stderr


def test_align_valid_text_file_reaches_model_resolution(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    text_file = qasr.work / "text.txt"
    text_file.write_text("hello world", encoding="utf-8")
    proc = _align(
        qasr,
        audio,
        "--text-file",
        text_file,
        "--language",
        "en",
        model="/nonexistent/model",
    )
    assert_exit(proc, 3, context="align valid text file")


def test_align_valid_srt_reaches_model_resolution(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=1.0)
    srt = qasr.work / "ok.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n", encoding="utf-8")
    proc = _align(
        qasr,
        audio,
        "--srt",
        srt,
        "--language",
        "en",
        model="/nonexistent/model",
    )
    assert_exit(proc, 3, context="align valid SRT")
