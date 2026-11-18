"""Model-backed P0 acceptance checks.

Run with:
    RUN_INTEGRATION=1 QASR_TEST_MODEL_DIR=... QASR_TEST_AUDIO=... \
        pytest review_tests/test_integration_p0.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
import wave
from pathlib import Path

import pytest
from helpers import (
    assert_exit,
    copy_wav,
    display_width,
    empty_wav,
    parse_json,
    parse_srt,
    parse_vtt,
    write_manifest,
    zero_byte,
)

pytestmark = pytest.mark.integration

DEFAULT_ALIGN_TEXT = "甚至出现交易几乎停滞的情况。"


def _env() -> dict[str, str]:
    return {"QASR_OFFLINE": "1"}


def _run(qasr, env, command: str, *args, timeout: float = 900):
    return qasr.run("--json", *env.model_args, command, *args, env=_env(), timeout=timeout)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _result(payload: dict) -> dict:
    result = payload.get("result")
    assert isinstance(result, dict), f"missing result object: {payload!r}"
    return result


def _summary(payload: dict) -> dict:
    summary = payload.get("summary")
    assert isinstance(summary, dict), f"missing batch summary: {payload!r}"
    return summary


def _outputs(out: Path) -> list[Path]:
    return sorted(path for path in out.rglob("*") if path.is_file() and ".qasr" not in path.parts)


def _state_records(out: Path) -> list[dict]:
    state = out / ".qasr" / "state.jsonl"
    if not state.exists():
        return []
    records = []
    for line in state.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def test_transcribe_text_and_forced_language(qasr, integration_env):
    proc = _run(qasr, integration_env, "transcribe", integration_env.audio, "--language", "zh")
    assert_exit(proc, 0, context="integration transcribe zh")
    payload = parse_json(proc)
    result = _result(payload)
    text = result.get("text", "")
    assert isinstance(text, str) and text.strip(), f"empty transcription: {payload!r}"
    assert "<asr_text>" not in text
    assert not text.lstrip().lower().startswith("language "), f"metadata leaked into text: {text!r}"
    assert result.get("language") == "Chinese", result
    assert result.get("language_source") == "forced", result

    expected = os.environ.get("QASR_TEST_AUDIO_TEXT")
    if expected:
        assert expected in text, f"expected {expected!r} in {text!r}"


def test_plain_transcribe_does_not_need_aligner(qasr, integration_env):
    bogus = qasr.work / "bogus-aligner"
    bogus.mkdir()
    proc = qasr.run(
        "--json",
        "--model",
        str(integration_env.asr),
        "--aligner",
        str(bogus),
        "transcribe",
        str(integration_env.audio),
        "--language",
        "zh",
        env=_env(),
        timeout=900,
    )
    assert_exit(proc, 0, context="text-only transcribe must not load aligner")
    assert _result(parse_json(proc)).get("text", "").strip()


def test_timestamps_segments_are_clean_and_monotonic(qasr, integration_env):
    proc = _run(
        qasr,
        integration_env,
        "transcribe",
        integration_env.audio,
        "--language",
        "zh",
        "--timestamps",
    )
    assert_exit(proc, 0, context="integration timestamps")
    payload = parse_json(proc)
    result = _result(payload)
    assert result.get("timestamp_source") == "forced_aligner", result
    segments = result.get("segments")
    assert isinstance(segments, list) and segments, f"no segments: {result!r}"

    previous_start = -1.0
    for segment in segments:
        assert isinstance(segment, dict), segment
        start = float(segment["start"])
        end = float(segment["end"])
        assert start <= end + 1e-3, f"reversed segment: {segment!r}"
        assert start + 1e-3 >= previous_start, f"non-monotonic segments: {segments!r}"
        previous_start = start
        assert segment.get("kind") in {"sentence", "word", "char"}, segment
        assert str(segment.get("text", "")).strip(), segment
    assert "<asr_text>" not in result.get("text", "")


def test_subtitles_are_valid(qasr, integration_env):
    out = qasr.work / "subs"
    out.mkdir()
    proc = qasr.run(
        "--json",
        *integration_env.model_args,
        "transcribe",
        str(integration_env.audio),
        "--language",
        "zh",
        "--timestamps",
        "-f",
        "srt,vtt",
        "-o",
        str(out),
        env=_env(),
        timeout=900,
    )
    assert_exit(proc, 0, context="integration srt/vtt")
    payload = parse_json(proc)
    duration_s = None
    input_info = payload.get("input")
    if isinstance(input_info, dict):
        duration_s = input_info.get("duration_s")
    if duration_s is None:
        try:
            duration_s = _duration(integration_env.audio)
        except Exception:
            duration_s = None

    srt_files = sorted(out.rglob("*.srt"))
    vtt_files = sorted(out.rglob("*.vtt"))
    assert len(srt_files) == 1, f"expected one SRT, got {srt_files}"
    assert len(vtt_files) == 1, f"expected one VTT, got {vtt_files}"

    for cues, label in (
        (parse_srt(srt_files[0].read_text(encoding="utf-8")), "SRT"),
        (parse_vtt(vtt_files[0].read_text(encoding="utf-8")), "VTT"),
    ):
        assert cues, f"{label} has no cues"
        previous_end = 0
        for start, end, text in cues:
            assert end >= start + 1, f"{label} cue shorter than 1ms: {(start, end)}"
            assert start >= previous_end, f"{label} cues overlap: {(previous_end, start)}"
            if duration_s is not None:
                assert end <= int(duration_s * 1000) + 1000, f"{label} cue exceeds audio: {end}"
            assert end - start <= 7000, f"{label} cue longer than 7s: {end - start}"
            for line in text.splitlines():
                assert display_width(line) <= 42, f"{label} line too wide ({display_width(line)}): {line!r}"
            previous_end = end

def test_align_known_transcript(qasr, integration_env):
    text = os.environ.get("QASR_TEST_ALIGN_TEXT", DEFAULT_ALIGN_TEXT)
    proc = _run(
        qasr,
        integration_env,
        "align",
        integration_env.audio,
        "--text",
        text,
        "--language",
        "zh",
    )
    assert_exit(proc, 0, context="integration align")
    payload = parse_json(proc)
    result = _result(payload)
    items = result.get("items")
    assert isinstance(items, list) and items, f"no alignment items: {result!r}"

    previous_end = 0.0
    for item in items:
        start = float(item["start"])
        end = float(item["end"])
        assert start <= end + 1e-3, item
        assert start + 1e-3 >= previous_end, f"alignment overlaps: {items!r}"
        previous_end = end

    coverage = result.get("coverage")
    assert isinstance(coverage, (int, float)) and coverage >= 0.5, result

    using_default_audio = Path(os.environ.get("QASR_TEST_AUDIO", "/tmp/qasr_asr_zh.wav")) == integration_env.audio
    if using_default_audio and text == DEFAULT_ALIGN_TEXT:
        assert len(items) == 13, f"expected 13 Chinese character items, got {len(items)}: {items!r}"


def test_align_coverage_full_when_aligner_force_aligns_extra_text(qasr, integration_env):
    """A real forced aligner aligns every reference character, so coverage stays 1.0.

    The ``--coverage-min`` failure path is covered by the repo's fake-aligner
    tests; with a real aligner it is unreachable for ordinary text because the
    DESIGN text-matching definition counts every alignable character.
    """
    text = DEFAULT_ALIGN_TEXT + "这是完全无关的额外文本内容。"
    proc = _run(
        qasr,
        integration_env,
        "align",
        integration_env.audio,
        "--text",
        text,
        "--language",
        "zh",
        "--coverage-min",
        "0.9",
    )
    assert_exit(proc, 0, context="real aligner coverage")
    result = _result(parse_json(proc))
    assert result.get("coverage") == pytest.approx(1.0), result
    assert len(result.get("items", [])) > 13, result


def test_empty_audio_succeeds_with_null_language(qasr, integration_env):
    audio = empty_wav(qasr.work / "empty.wav")
    proc = _run(qasr, integration_env, "transcribe", audio)
    assert_exit(proc, 0, context="empty audio")
    result = _result(parse_json(proc))
    assert result.get("text") == "", result
    assert result.get("language") is None, result


def test_batch_isolates_bad_audio(qasr, integration_env):
    good = copy_wav(integration_env.audio, qasr.work / "good.wav")
    bad = zero_byte(qasr.work / "bad.wav")
    out = qasr.work / "batch"
    proc = _run(
        qasr,
        integration_env,
        "transcribe",
        good,
        bad,
        "--language",
        "zh",
        "-o",
        out,
    )
    assert_exit(proc, 6, context="mixed good/bad batch")
    payload = parse_json(proc)
    summary = _summary(payload)
    assert summary.get("total") == 2 and summary.get("ok") == 1 and summary.get("failed") == 1, summary
    ok_items = [item for item in payload.get("items", []) if item.get("ok")]
    assert len(ok_items) == 1, payload
    assert _result(ok_items[0]).get("text", "").strip()
    assert _outputs(out), "successful item produced no output file"


def _resume_manifest(qasr, integration_env, count: int = 2, context: str = "A") -> Path:
    records = []
    for index in range(count):
        path = copy_wav(integration_env.audio, qasr.work / f"in-{index}.wav")
        records.append(
            {
                "id": f"j{index}",
                "audio": str(path),
                "language": "zh",
                "context": context,
                "formats": ["txt", "json"],
            }
        )
    return write_manifest(qasr.work / "resume.jsonl", records)


def test_resume_skips_completed_items(qasr, integration_env):
    manifest = _resume_manifest(qasr, integration_env)
    out = qasr.work / "resume-out"
    args = ("transcribe", "--manifest", str(manifest), "-o", str(out), "--resume")

    first = _run(qasr, integration_env, *args)
    assert_exit(first, 0, context="resume first run")
    assert _summary(parse_json(first)).get("ok") == 2
    before = {path: _sha(path) for path in _outputs(out)}
    assert before

    second = _run(qasr, integration_env, *args)
    assert_exit(second, 0, context="resume second run")
    summary = _summary(parse_json(second))
    assert summary.get("skipped") == 2, f"resume re-ran completed items: {summary}"
    assert summary.get("ok") == 0, summary

    after = {path: _sha(path) for path in _outputs(out)}
    assert after == before, "resume changed output bytes for completed items"


def test_resume_reruns_when_request_changes(qasr, integration_env):
    out = qasr.work / "changed-out"
    manifest_a = _resume_manifest(qasr, integration_env, count=1, context="A")
    first = _run(qasr, integration_env, "transcribe", "--manifest", str(manifest_a), "-o", str(out), "--resume")
    assert_exit(first, 0)
    first_records = [r for r in _state_records(out) if r.get("id") == "j0" and r.get("status") == "done"]
    assert first_records, "first run recorded no done state"
    first_key = first_records[-1].get("job_key")
    assert first_key, first_records[-1]

    manifest_b = write_manifest(
        qasr.work / "resume.jsonl",
        [
            {
                "id": "j0",
                "audio": str(copy_wav(integration_env.audio, qasr.work / "in-0.wav")),
                "language": "zh",
                "context": "B",
                "formats": ["txt", "json"],
            }
        ],
    )
    second = _run(qasr, integration_env, "transcribe", "--manifest", str(manifest_b), "-o", str(out), "--resume")
    assert_exit(second, 0)
    payload = parse_json(second)
    # A one-record manifest yields a single-item payload, not a batch envelope.
    assert payload.get("ok") is True, payload
    assert _result(payload).get("skipped") is not True, payload
    records = [r for r in _state_records(out) if r.get("id") == "j0"]
    assert len(records) >= 2, f"context change produced no new state record: {records}"
    assert records[-1].get("job_key") != first_key, "context change reused the stale job_key"
    assert records[-1].get("status") == "done", records[-1]


def test_resume_reruns_when_output_is_tampered(qasr, integration_env):
    manifest = _resume_manifest(qasr, integration_env, count=1)
    out = qasr.work / "tamper-out"
    first = _run(qasr, integration_env, "transcribe", "--manifest", str(manifest), "-o", str(out), "--resume")
    assert_exit(first, 0)
    outputs = _outputs(out)
    assert outputs
    target = next((path for path in outputs if path.suffix == ".json"), outputs[0])
    target.write_text(target.read_text(encoding="utf-8") + "\nTAMPERED\n", encoding="utf-8")
    tampered_sha = _sha(target)

    second = _run(qasr, integration_env, "transcribe", "--manifest", str(manifest), "-o", str(out), "--resume")
    assert_exit(second, 0)
    payload = parse_json(second)
    # A one-record manifest yields a single-item payload, not a batch envelope.
    assert payload.get("ok") is True, payload
    assert _result(payload).get("skipped") is not True, payload
    assert "TAMPERED" not in target.read_text(encoding="utf-8")
    assert _sha(target) != tampered_sha, "tampered output was not rewritten"
    json.loads(target.read_text(encoding="utf-8"))


def test_model_stdout_remains_pure_json(qasr, integration_env):
    proc = _run(qasr, integration_env, "transcribe", integration_env.audio, "--language", "zh")
    assert_exit(proc, 0)
    stripped = proc.stdout.strip()
    assert stripped.startswith("{") and stripped.endswith("}"), proc.stdout[:500]
    parse_json(proc)
