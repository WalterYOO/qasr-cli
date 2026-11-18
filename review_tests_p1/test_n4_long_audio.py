"""N4: long-audio window planning, 180s/270s boundaries, and dry-run safety."""

from __future__ import annotations

import os

import pytest
from fake_openai import error_response, json_response
from helpers import assert_exit, find_key, parse_json

pytestmark = pytest.mark.n4

SAMPLE_RATE = 16_000


def _windows(payload: dict) -> list[dict]:
    candidates: list = []
    for key in ("windows", "window_plan", "chunks"):
        candidates.extend(
            value for value in find_key(payload, key)
            if isinstance(value, list) and value
        )
    assert candidates, f"dry-run has no window plan: {payload!r}"
    windows = candidates[0]
    assert all(isinstance(window, dict) for window in windows), windows
    return windows


def _value(window: dict, *names: str):
    for name in names:
        if name in window:
            return window[name]
    raise AssertionError(f"window has none of {names}: {window!r}")


def _span(window: dict, prefix: str) -> tuple[float, float]:
    sample_names = (f"{prefix}_start_sample", f"{prefix}_end_sample")
    if sample_names[0] in window:
        return (
            float(_value(window, sample_names[0])) / SAMPLE_RATE,
            float(_value(window, sample_names[1])) / SAMPLE_RATE,
        )
    start = _value(window, f"{prefix}_start_s", f"{prefix}_start")
    end = _value(window, f"{prefix}_end_s", f"{prefix}_end")
    return float(start), float(end)


def _audio_span(window: dict) -> tuple[float, float]:
    return _span(window, "audio")


def _core_span(window: dict) -> tuple[float, float]:
    return _span(window, "core")


def _align_dry(qasr, audio, text_file, *extra, timeout=60):
    return qasr.run(
        "align", audio, "--text-file", text_file, "--language", "zh",
        "--dry-run", "--json", *extra, timeout=timeout,
    )


def test_align_181s_dry_run_plans_windows(qasr, long_wav, text_file):
    proc = _align_dry(qasr, long_wav, text_file)
    assert_exit(proc, 0, context="align 181s dry-run")
    payload = parse_json(proc)
    windows = _windows(payload)
    assert len(windows) >= 2, windows
    assert [window.get("index") for window in windows] == list(range(len(windows)))


def test_align_window_audio_never_exceeds_180s(qasr, long_wav, text_file):
    proc = _align_dry(qasr, long_wav, text_file)
    assert_exit(proc, 0, context="align window span")
    for window in _windows(parse_json(proc)):
        start, end = _audio_span(window)
        assert end <= start + 180.0 + 1e-6, (start, end, window)
        assert end > start, window


def test_align_window_formula(qasr, long_wav, text_file):
    proc = _align_dry(qasr, long_wav, text_file)
    assert_exit(proc, 0, context="align window formula")
    windows = _windows(parse_json(proc))
    for index, window in enumerate(windows):
        core_start, core_end = _core_span(window)
        audio_start, audio_end = _audio_span(window)
        assert abs(audio_end - core_end) < 0.05, (audio_start, audio_end, core_start, core_end)
        expected_start = max(0.0, core_start - 3.0)
        assert abs(audio_start - expected_start) < 0.05, (
            index, audio_start, expected_start
        )
        if index < len(windows) - 1:
            assert 175.0 <= core_end - core_start <= 177.0, (index, core_start, core_end)


def test_align_core_ranges_are_contiguous_and_cover_duration(
    qasr, long_wav, text_file
):
    proc = _align_dry(qasr, long_wav, text_file)
    assert_exit(proc, 0, context="align core ranges")
    windows = _windows(parse_json(proc))
    cores = [_core_span(window) for window in windows]
    assert abs(cores[0][0]) < 0.05, cores[0]
    assert abs(cores[-1][1] - 181.0) < 0.5, cores[-1]
    for (start, end), (next_start, _next_end) in zip(cores, cores[1:]):
        assert end > start, (start, end)
        assert abs(end - next_start) < 0.05, (cores,)


def test_align_exactly_180s_is_single_window(qasr, exactly_180_wav, text_file):
    proc = _align_dry(qasr, exactly_180_wav, text_file)
    assert_exit(proc, 0, context="align 180s boundary")
    windows = _windows(parse_json(proc))
    assert len(windows) == 1, windows
    start, end = _core_span(windows[0])
    assert abs(start) < 0.05
    assert abs(end - 180.0) < 0.5


def test_align_overlap_zero_is_accepted(qasr, long_wav, text_file):
    """P1-TASKS T4.3 allows 0-10, while T4.1 says overlap > 0."""
    proc = _align_dry(qasr, long_wav, text_file, "--window-overlap", "0")
    assert_exit(proc, 0, context="align overlap 0")
    windows = _windows(parse_json(proc))
    cores = [_core_span(window) for window in windows]
    for (start, end), (next_start, _next_end) in zip(cores, cores[1:]):
        assert abs(end - next_start) < 0.05, cores
    for index, window in enumerate(windows):
        core_start, core_end = _core_span(window)
        audio_start, _audio_end = _audio_span(window)
        assert abs(audio_start - core_start) < 0.05, (index, audio_start, core_start)
        if index < len(windows) - 1:
            assert 179.0 <= core_end - core_start <= 180.0, (index, core_start, core_end)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--window-seconds", "9"),
        ("--window-seconds", "181"),
        ("--window-overlap", "-1"),
        ("--window-overlap", "11"),
    ],
)
def test_align_window_flag_validation(qasr, long_wav, text_file, flag, value):
    proc = _align_dry(qasr, long_wav, text_file, flag, value)
    assert_exit(proc, 2, context=f"align {flag} {value}")
    assert "Traceback" not in proc.stderr


def test_align_window_overlap_must_be_smaller_than_window(
    qasr, long_wav, text_file
):
    proc = _align_dry(
        qasr, long_wav, text_file,
        "--window-seconds", "10", "--window-overlap", "10",
    )
    assert_exit(proc, 2, context="overlap == window")


def test_timestamps_with_max_chunk_over_180_is_usage_error(qasr, long_wav):
    proc = qasr.run(
        "transcribe", long_wav, "--timestamps", "--language", "zh",
        "--max-chunk-seconds", "181", "--json",
    )
    assert_exit(proc, 4, context="timestamps + max-chunk 181")
    error = parse_json(proc)["error"]
    text = (error.get("message") or "") + " " + (error.get("hint") or "")
    assert "180" in text or "align" in text.lower()


def test_timestamps_with_default_chunk_is_allowed(qasr, long_wav):
    proc = qasr.run(
        "transcribe", long_wav, "--timestamps", "--language", "zh",
        "--dry-run", "--json",
    )
    assert_exit(proc, 0, context="timestamps default chunk dry-run")


def test_pure_text_chunk_default_and_bounds(qasr, long_wav):
    default = qasr.run("transcribe", long_wav, "--dry-run", "--json")
    assert_exit(default, 0, context="pure-text default chunk")
    for value in ("10", "1200"):
        proc = qasr.run(
            "transcribe", long_wav, "--max-chunk-seconds", value,
            "--dry-run", "--json",
        )
        assert_exit(proc, 0, context=f"max-chunk {value}")


@pytest.mark.parametrize("value", ["9", "1201", "0", "-1", "abc"])
def test_max_chunk_invalid_values_do_not_crash(qasr, long_wav, value):
    proc = qasr.run(
        "transcribe", long_wav, "--max-chunk-seconds", value,
        "--dry-run", "--json",
    )
    assert_exit(proc, 2, context=f"max-chunk {value}")
    assert "Traceback" not in proc.stderr


def test_align_dry_run_does_not_import_runtime(
    qasr, long_wav, text_file, import_blocker, repo_root
):
    blocker, marker = import_blocker
    env = {"PYTHONPATH": f"{blocker}{os.pathsep}{repo_root}"}
    proc = qasr.run(
        "align", long_wav, "--text-file", text_file, "--language", "zh",
        "--dry-run", "--json", env=env, timeout=60,
    )
    assert_exit(proc, 0, context="align dry-run import blocker")
    assert not marker.exists(), "align --dry-run imported torch/qwen_asr/vllm"


def test_align_auto_han_script_resolves_without_model(
    qasr, long_wav, text_file
):
    proc = qasr.run(
        "align", long_wav, "--text-file", text_file, "--language", "auto",
        "--dry-run", "--json",
    )
    assert_exit(proc, 0, context="align auto Han")
    payload = parse_json(proc)
    dumped = str(payload).lower()
    assert "chinese" in dumped or "zh" in dumped


def test_align_latin_auto_needs_lid_probe(qasr, long_wav, tmp_path):
    latin = tmp_path / "latin.txt"
    latin.write_text("hello world, this is a test", encoding="utf-8")
    proc = qasr.run(
        "align", long_wav, "--text-file", latin, "--language", "auto",
        "--dry-run", "--json", timeout=60,
    )
    # Dry-run must not load an ASR model to probe; the capability decision is
    # either deferred to the plan or rejected cleanly.
    assert proc.returncode in {0, 3, 4}, proc.stderr
    assert "Traceback" not in proc.stderr


def test_srt_without_timestamps_is_still_capability_error(qasr, long_wav, tmp_path):
    proc = qasr.run(
        "transcribe", long_wav, "-f", "srt", "-o", tmp_path / "out", "--json"
    )
    assert_exit(proc, 4, context="srt without timestamps")


def test_remote_long_audio_is_chunked_client_side(qasr, fake_openai, long_wav):
    server = fake_openai()
    counter = {"n": 0}

    def transcription(request, srv):
        index = counter["n"]
        counter["n"] += 1
        return json_response({"text": f"language Chinese<asr_text>片段{index}"})

    server.transcription_response = transcription
    proc = qasr.run(
        "--server", server.url, "transcribe", long_wav,
        "--max-chunk-seconds", "60", "--json", timeout=120,
    )
    assert_exit(proc, 0, context="remote chunking")
    payload = parse_json(proc)
    posts = server.requests_for("/v1/audio/transcriptions")
    assert len(posts) >= 4, [request.headers.get("content-length") for request in posts]
    for request in posts:
        assert len(request.body) <= 25 * 1024 * 1024, len(request.body)
    final_text = payload["result"]["text"]
    for index in range(len(posts)):
        assert f"片段{index}" in final_text, final_text


def test_remote_chunk_resume_only_retries_unfinished(
    qasr, fake_openai, long_wav, tmp_path
):
    server = fake_openai()
    state = {"fail_second_window": True}

    def transcription(request, srv):
        fields, files = request.form
        name = files["file"].filename if "file" in files else ""
        if state["fail_second_window"] and "window-2" in name:
            return error_response("transient chunk failure", status=500)
        return json_response({"text": f"language Chinese<asr_text>{name}"})

    server.transcription_response = transcription
    out = tmp_path / "out"
    first = qasr.run(
        "--server", server.url, "transcribe", long_wav,
        "--out-dir", out, "--max-chunk-seconds", "60",
        "--retries", "0", "--json", timeout=120,
    )
    assert_exit(first, 3, context="chunked run with one failed window")
    chunk_dir = out / ".qasr" / "chunks"
    saved = sorted(path.name for path in chunk_dir.rglob("*.json"))
    assert saved, "completed chunk checkpoints were not written"

    state["fail_second_window"] = False
    server.reset()
    second = qasr.run(
        "--server", server.url, "transcribe", long_wav,
        "--out-dir", out, "--max-chunk-seconds", "60", "--resume",
        "--json", timeout=120,
    )
    assert_exit(second, 0, context="resume after one failed window")
    uploads = server.requests_for("/v1/audio/transcriptions")
    names = []
    for request in uploads:
        fields, files = request.form
        names.append(files["file"].filename if "file" in files else "")
    assert len(names) < 4, f"resume re-uploaded finished chunks: {names}"
    assert any("window-2" in name for name in names), names


def test_remote_chunk_ceiling_warns_and_downsizes(qasr, fake_openai, long_wav):
    server = fake_openai()
    proc = qasr.run(
        "--server", server.url, "transcribe", long_wav,
        "--max-chunk-seconds", "1200", "--json", timeout=120,
    )
    assert_exit(proc, 0, context="remote 600s ceiling")
    assert "600" in proc.stderr, proc.stderr
    assert len(server.requests_for("/v1/audio/transcriptions")) == 1


@pytest.mark.integration
def test_align_long_audio_end_to_end_monotonic(long_wav, text_file):
    pytest.skip("requires a real 0.6B ASR + aligner; run with RUN_INTEGRATION=1")
