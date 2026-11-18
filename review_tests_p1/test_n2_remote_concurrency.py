"""N2: remote --jobs concurrency, ordering, fail-fast, state integrity."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from fake_openai import error_response, json_response
from helpers import assert_exit, parse_json, parse_jsonl, read_state


def _latest_by_id(records: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for record in records:
        item_id = record.get("id")
        if isinstance(item_id, str):
            latest[item_id] = record
    return latest


def _copies(tmp_path: Path, source: Path, count: int) -> list[Path]:
    paths = []
    for index in range(1, count + 1):
        target = tmp_path / f"f{index}.wav"
        shutil.copyfile(source, target)
        paths.append(target)
    return paths


def _delayed_ok(request, srv):
    fields, files = request.form
    name = files["file"].filename if "file" in files else ""
    match = re.search(r"f(\d+)", name)
    index = int(match.group(1)) if match else 1
    delay = max(0.05, 0.6 - index * 0.08)
    return json_response(
        {"text": f"language Chinese<asr_text>{name}"}, delay=delay
    )


def test_remote_jobs_default_is_parallel(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai()
    server.handler = _delayed_ok
    inputs = _copies(tmp_path, short_wav, 4)
    proc = qasr.run("--server", server.url, "transcribe", *inputs, "--json")
    assert_exit(proc, 0, context="remote default jobs")
    assert server.max_active > 1, server.max_active


def test_jobs_4_bounds_concurrency_and_preserves_order(
    qasr, fake_openai, tmp_path, short_wav
):
    server = fake_openai()
    server.handler = _delayed_ok
    inputs = _copies(tmp_path, short_wav, 6)
    proc = qasr.run(
        "--server", server.url, "transcribe", *inputs,
        "--jobs", "4", "--json", timeout=60,
    )
    assert_exit(proc, 0, context="remote jobs 4")
    payload = parse_json(proc)
    ids = [item["input"]["id"] for item in payload["items"]]
    assert ids == [f"f{index}" for index in range(1, 7)], ids
    assert 1 < server.max_active <= 4, server.max_active


def test_jobs_1_is_strictly_serial(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai()
    server.handler = _delayed_ok
    inputs = _copies(tmp_path, short_wav, 3)
    proc = qasr.run(
        "--server", server.url, "transcribe", *inputs,
        "--jobs", "1", "--json", timeout=60,
    )
    assert_exit(proc, 0, context="remote jobs 1")
    assert server.max_active == 1, server.max_active


def test_jsonl_order_matches_input_order(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai()
    server.handler = _delayed_ok
    inputs = _copies(tmp_path, short_wav, 4)
    proc = qasr.run(
        "--server", server.url, "transcribe", *inputs,
        "--jobs", "4", "--jsonl", timeout=60,
    )
    assert_exit(proc, 0, context="remote jsonl order")
    records = parse_jsonl(proc)
    assert [record["input"]["id"] for record in records] == [
        f"f{index}" for index in range(1, 5)
    ]


def test_jobs_range_validation(qasr, short_wav):
    for value in ("0", "33", "-1", "abc"):
        proc = qasr.run("transcribe", short_wav, "--jobs", value, "--json")
        assert_exit(proc, 2, context=f"--jobs {value}")
        assert "Traceback" not in proc.stderr


def test_local_jobs_gt_1_warns_and_forces_one(qasr, empty_wavs):
    proc = qasr.run(
        "transcribe", *empty_wavs, "--jobs", "4", "--json"
    )
    assert_exit(proc, 0, context="local jobs warning")
    payload = parse_json(proc)
    assert payload["summary"]["ok"] == 2
    assert re.search(r"jobs[^\n]*1", proc.stderr, re.IGNORECASE), proc.stderr


def test_fail_fast_leaves_no_running_state_and_unlocks(
    qasr, fake_openai, tmp_path, short_wav, empty_wavs
):
    server = fake_openai()

    def handler(request, srv):
        fields, files = request.form
        name = files["file"].filename if "file" in files else ""
        if name.startswith("fail"):
            return error_response("bad input", status=400)
        return json_response(
            {"text": f"language Chinese<asr_text>{name}"}, delay=1.0
        )

    server.handler = handler
    fail = tmp_path / "fail.wav"
    shutil.copyfile(short_wav, fail)
    rest = [tmp_path / f"slow{index}.wav" for index in range(1, 5)]
    for path in rest:
        shutil.copyfile(short_wav, path)
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", fail, *rest,
        "--jobs", "3", "--fail-fast", "--out-dir", out, "--json", timeout=60,
    )
    # T2.3 (final): --fail-fast exits with the first error's code (4 here),
    # not the generic partial-batch code 6.
    assert_exit(proc, 4, context="remote fail-fast")
    payload = parse_json(proc)
    summary = payload["summary"]
    assert summary["total"] == 5
    assert summary["failed"] >= 1
    assert summary["ok"] + summary["failed"] + summary["skipped"] == 5
    records = read_state(out)
    assert records, "fail-fast batch must still write state"
    latest = _latest_by_id(records)
    assert set(latest) == {"fail", "slow1", "slow2", "slow3", "slow4"}, latest
    still_running = {
        item_id: record for item_id, record in latest.items()
        if record.get("status") == "running"
    }
    assert not still_running, still_running
    assert latest["fail"]["status"] == "failed"
    assert any(record.get("status") == "skipped" for record in latest.values()), latest

    rerun = qasr.run(
        "transcribe", *empty_wavs, "--out-dir", out, "--json", timeout=30
    )
    assert_exit(rerun, 0, context="lock released after fail-fast")


def test_concurrent_state_lines_are_valid_json(
    qasr, fake_openai, tmp_path, short_wav
):
    server = fake_openai()
    server.handler = _delayed_ok
    inputs = _copies(tmp_path, short_wav, 10)
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", *inputs,
        "--jobs", "4", "--out-dir", out, "--json", timeout=60,
    )
    assert_exit(proc, 0, context="concurrent state")
    records = read_state(out)
    assert records, "state.jsonl missing"
    ids = {record.get("id") for record in records}
    assert ids == {f"f{index}" for index in range(1, 11)}, ids
    latest = _latest_by_id(records)
    assert all(record.get("status") == "done" for record in latest.values()), latest
    for record in records:
        assert record.get("schema_version") == 1
        json.dumps(record)


def test_remote_batch_size_warns_about_double_concurrency(
    qasr, fake_openai, tmp_path, short_wav
):
    server = fake_openai()
    inputs = _copies(tmp_path, short_wav, 2)
    proc = qasr.run(
        "--server", server.url, "transcribe", *inputs,
        "--jobs", "2", "--batch-size", "2", "--json", timeout=60,
    )
    assert_exit(proc, 0, context="remote batch-size")
    assert re.search(r"batch[- ]size", proc.stderr, re.IGNORECASE), proc.stderr


def test_remote_resume_skips_completed_items_without_uploading(
    qasr, fake_openai, tmp_path, short_wav
):
    server = fake_openai()
    server.handler = _delayed_ok
    inputs = _copies(tmp_path, short_wav, 2)
    out = tmp_path / "out"
    first = qasr.run(
        "--server", server.url, "transcribe", *inputs,
        "--out-dir", out, "--json", timeout=60,
    )
    assert_exit(first, 0, context="remote resume first run")
    assert len(server.requests_for("/v1/audio/transcriptions")) == 2

    second = qasr.run(
        "--server", server.url, "transcribe", *inputs,
        "--out-dir", out, "--resume", "--json", timeout=60,
    )
    assert_exit(second, 0, context="remote resume second run")
    payload = parse_json(second)
    assert payload["summary"]["skipped"] == 2, payload["summary"]
    assert len(server.requests_for("/v1/audio/transcriptions")) == 2
    latest = _latest_by_id(read_state(out))
    assert all(record.get("status") == "skipped" for record in latest.values()), latest
