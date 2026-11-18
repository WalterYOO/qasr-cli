"""TR-01..20: input expansion, preflight validation, and dry-run planning."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import (
    assert_error,
    assert_exit,
    empty_wav,
    make_wav,
    parse_json,
    text_as_wav,
    truncated_wav,
    zero_byte,
)


def _plan(payload: dict) -> dict:
    plan = payload.get("plan")
    assert isinstance(plan, dict), f"dry-run has no plan object: {payload!r}"
    return plan


def _plan_inputs(payload: dict) -> list:
    inputs = _plan(payload).get("inputs")
    assert isinstance(inputs, list), f"plan.inputs is not a list: {payload!r}"
    return inputs


def _source(item):
    if isinstance(item, dict):
        return item.get("source") or item.get("path") or item.get("audio")
    return item


def _basenames(payload: dict) -> list[str]:
    return [Path(str(_source(item))).name for item in _plan_inputs(payload)]


def _run_json(qasr, *args, timeout=60):
    return qasr.run("--json", *args, timeout=timeout)


def test_missing_file_is_input_error(qasr):
    proc = _run_json(qasr, "transcribe", "missing.wav")
    assert_exit(proc, 4, context="missing input")
    assert_error(parse_json(proc), code=4)


@pytest.mark.parametrize("builder", [zero_byte, truncated_wav, text_as_wav])
def test_corrupt_audio_is_input_error(qasr, builder):
    audio = builder(qasr.work / "corrupt.wav")
    proc = _run_json(qasr, "transcribe", audio)
    assert_exit(proc, 4, context=f"corrupt audio via {builder.__name__}")
    assert_error(parse_json(proc), code=4)
    assert "Traceback" not in proc.stderr, f"corrupt audio leaked a traceback:\n{proc.stderr}"


def test_unsupported_extension_is_input_error(qasr):
    path = qasr.work / "note.txt"
    path.write_text("hello", encoding="utf-8")
    proc = _run_json(qasr, "transcribe", path)
    assert_exit(proc, 4, context="unsupported extension")
    assert_error(parse_json(proc), code=4)


def test_multiple_inputs_with_json_emit_batch_envelope(qasr):
    # Empty WAVs short-circuit without a model, so this stays model-free while
    # still exercising the multi-input + --json output channel.
    a = empty_wav(qasr.work / "empty-a.wav")
    b = empty_wav(qasr.work / "empty-b.wav")
    proc = _run_json(qasr, "transcribe", a, b)
    assert_exit(proc, 0, context="multiple inputs + --json")
    payload = parse_json(proc)
    summary = payload.get("summary")
    assert isinstance(summary, dict), f"multi-input --json must be a batch envelope: {payload!r}"
    assert summary.get("total") == 2 and summary.get("ok") == 2, summary
    items = payload.get("items")
    assert isinstance(items, list) and len(items) == 2, payload
    for item in items:
        result = item.get("result", {})
        assert result.get("text") == ""
        assert result.get("language") is None


def test_multiple_inputs_without_output_channel_are_usage_error(qasr):
    a = empty_wav(qasr.work / "empty-a.wav")
    b = empty_wav(qasr.work / "empty-b.wav")
    proc = qasr.run("transcribe", a, b)
    assert_exit(proc, 2, context="multiple inputs without any output channel")
    text = (proc.stdout + proc.stderr).lower()
    assert "out-dir" in text or "output" in text or "json" in text, text[:2000]


def test_output_flag_rejects_multiple_inputs(qasr):
    a = make_wav(qasr.work / "a.wav", seconds=0.2)
    b = make_wav(qasr.work / "b.wav", seconds=0.2)
    proc = _run_json(qasr, "transcribe", a, b, "--output", qasr.work / "one.txt")
    assert_exit(proc, 2, context="--output with two inputs")
    assert_error(parse_json(proc), code=2)


@pytest.mark.parametrize(
    "args",
    [
        ("-f", "txt,json"),
        ("-f", "json"),
    ],
)
def test_invalid_format_combinations(qasr, args):
    audio = make_wav(qasr.work / "a.wav", seconds=0.2)
    proc = qasr.run("--json", "transcribe", audio, *args)
    assert_exit(proc, 2, context=f"format combination {args}")
    assert_error(parse_json(proc), code=2, kind="usage")


def test_tsv_format_is_accepted(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.2)
    # --format tsv/md is unconditionally mutually exclusive with --json/--jsonl,
    # so request the dry-run plan through the default (dry-run always prints JSON).
    proc = qasr.run("--dry-run", "transcribe", audio, "-f", "tsv")
    assert_exit(proc, 0, context="-f tsv")
    plan = parse_json(proc).get("plan") or {}
    assert plan.get("formats") == ["tsv"], plan


def test_jobs_greater_than_one_warns_and_forces_one(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.2)
    proc = qasr.run("--json", "--dry-run", "transcribe", audio, "--jobs", "2")
    assert_exit(proc, 0, context="local --jobs 2")
    assert "job" in proc.stderr.lower(), (
        "local --jobs>1 must warn and force 1; stderr was:\n" + proc.stderr
    )


def test_url_dry_run_does_not_download(qasr):
    proc = qasr.run(
        "--json",
        "--dry-run",
        "transcribe",
        "https://example.invalid/x.wav",
        timeout=30,
    )
    assert_exit(proc, 0, context="URL dry-run")
    payload = parse_json(proc)
    assert len(_plan_inputs(payload)) == 1
    cache = qasr.home / "cache" / "url"
    assert not cache.exists() or not any(cache.iterdir()), (
        f"dry-run downloaded a URL into {cache}"
    )


def test_offline_rejects_uncached_url(qasr):
    proc = qasr.run(
        "--offline",
        "--json",
        "transcribe",
        "https://example.invalid/x.wav",
        timeout=60,
    )
    assert_exit(proc, 4, context="offline URL")
    error = assert_error(parse_json(proc), code=4)
    text = (error.get("message", "") + " " + str(error.get("hint", ""))).lower()
    assert "offline" in text or "cache" in text, f"unclear offline error: {error!r}"


@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_batch_size(qasr, value):
    audio = make_wav(qasr.work / "a.wav", seconds=0.2)
    proc = qasr.run("--json", "transcribe", audio, "--batch-size", value)
    assert_exit(proc, 2, context=f"--batch-size {value}")


@pytest.mark.parametrize("value", ["0", "-1", "nope"])
def test_invalid_max_new_tokens(qasr, value):
    audio = make_wav(qasr.work / "a.wav", seconds=0.2)
    proc = qasr.run("--json", "transcribe", audio, "--max-new-tokens", value)
    assert_exit(proc, 2, context=f"--max-new-tokens {value}")


def test_max_new_tokens_auto_plans(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.2)
    proc = qasr.run("--json", "--dry-run", "transcribe", audio, "--max-new-tokens", "auto")
    assert_exit(proc, 0)
    assert parse_json(proc).get("ok") is True


def test_duplicate_inputs_are_deduplicated(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.2)
    out = qasr.work / "out"
    proc = qasr.run("--json", "--dry-run", "transcribe", audio, audio, "-o", out)
    assert_exit(proc, 0, context="duplicate input")
    assert len(_plan_inputs(parse_json(proc))) == 1


def test_literal_glob_is_expanded(qasr):
    folder = qasr.work / "glob"
    make_wav(folder / "a.wav", seconds=0.2)
    make_wav(folder / "b.wav", seconds=0.2)
    proc = qasr.run(
        "--json",
        "--dry-run",
        "transcribe",
        str(folder / "*.wav"),
        "-o",
        qasr.work / "out",
    )
    assert_exit(proc, 0, context="literal glob")
    assert _basenames(parse_json(proc)) == ["a.wav", "b.wav"]


def test_directory_expansion_is_sorted(qasr):
    folder = qasr.work / "sorted"
    for name in ("z.wav", "a.wav", "m.wav"):
        make_wav(folder / name, seconds=0.2)
    proc = qasr.run("--json", "--dry-run", "transcribe", folder, "-o", qasr.work / "out")
    assert_exit(proc, 0)
    assert _basenames(parse_json(proc)) == ["a.wav", "m.wav", "z.wav"]


def test_recursive_toggles_subdirectories(qasr):
    root = qasr.work / "tree"
    make_wav(root / "a.wav", seconds=0.2)
    make_wav(root / "sub" / "b.wav", seconds=0.2)
    out = qasr.work / "out"

    flat = qasr.run("--json", "--dry-run", "transcribe", root, "-o", out)
    assert_exit(flat, 0)
    assert _basenames(parse_json(flat)) == ["a.wav"]

    deep = qasr.run("--json", "--dry-run", "transcribe", root, "--recursive", "-o", out)
    assert_exit(deep, 0)
    assert _basenames(parse_json(deep)) == ["a.wav", "b.wav"]


def test_symlink_loop_terminates_and_deduplicates(qasr):
    root = qasr.work / "loop"
    make_wav(root / "a.wav", seconds=0.2)
    make_wav(root / "sub" / "b.wav", seconds=0.2)
    (root / "sub" / "back").symlink_to(root, target_is_directory=True)

    proc = qasr.run(
        "--json",
        "--dry-run",
        "transcribe",
        root,
        "--recursive",
        "-o",
        qasr.work / "out",
        timeout=25,
    )
    assert_exit(proc, 0, context="symlink loop")
    sources = [str(_source(item)) for item in _plan_inputs(parse_json(proc))]
    assert len(sources) == 2, f"symlink loop changed input count: {sources!r}"
    assert len(set(sources)) == len(sources), f"symlink loop produced duplicates: {sources!r}"


def test_dry_run_plan_has_expected_shape(qasr):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    out = qasr.work / "out"
    proc = qasr.run(
        "--json",
        "--dry-run",
        "transcribe",
        audio,
        "-f",
        "txt,srt,vtt",
        "--timestamps",
        "--language",
        "zh",
        "-o",
        out,
    )
    assert_exit(proc, 0, context="dry-run full plan")
    plan = _plan(parse_json(proc))
    for key in ("inputs", "model", "backend", "device", "dtype", "outputs"):
        assert key in plan, f"plan is missing {key!r}: {plan!r}"
    assert isinstance(plan["model"], dict)
    assert plan["backend"] in {"transformers", "vllm", "remote", "auto"}
    assert plan["device"] in {"cpu", "mps", "cuda", "auto"} or str(plan["device"]).startswith("cuda")
    outputs = plan["outputs"]
    assert isinstance(outputs, list) and len(outputs) == 3, f"expected 3 planned outputs: {outputs!r}"
    suffixes = {Path(str(item.get("path", item) if isinstance(item, dict) else item)).suffix for item in outputs}
    assert suffixes == {".txt", ".srt", ".vtt"}, f"unexpected planned outputs: {outputs!r}"
