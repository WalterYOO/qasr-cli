"""INT-RES-04/05 and INT-LOCK-01: crash recovery, signals, and concurrency."""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import pytest
from helpers import assert_exit, copy_wav, parse_json, write_manifest, zero_byte

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _run(qasr, env, *args, timeout=900):
    return qasr.run("--json", *env.model_args, *args, env={"QASR_OFFLINE": "1"}, timeout=timeout)


def _manifest(qasr, env, count: int) -> tuple[Path, Path]:
    records = []
    for index in range(count):
        path = copy_wav(env.audio, qasr.work / f"kill-{index}.wav")
        records.append(
            {
                "id": f"k{index}",
                "audio": str(path),
                "language": "zh",
                "formats": ["txt"],
            }
        )
    manifest = write_manifest(qasr.work / "kill.jsonl", records)
    out = qasr.work / "kill-out"
    return manifest, out


def _wait_until(predicate, timeout: float, interval: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


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


def _done_ids(out: Path) -> set[str]:
    return {
        str(record.get("id"))
        for record in _state_records(out)
        if record.get("status") == "done"
    }


def _outputs_for(out: Path, job_id: str) -> list[Path]:
    return [
        path
        for path in out.rglob("*")
        if path.is_file() and ".qasr" not in path.parts and path.stem.startswith(job_id)
    ]


def test_second_process_is_blocked_by_lock(qasr, integration_env):
    manifest, out = _manifest(qasr, integration_env, count=8)
    args = ("transcribe", "--manifest", str(manifest), "-o", str(out), "--resume")

    first = qasr.popen("--json", *integration_env.model_args, *args, env={"QASR_OFFLINE": "1"})
    try:
        appeared = _wait_until(lambda: (out / ".qasr" / "lock").exists(), timeout=90)
        if not appeared:
            if first.poll() is not None:
                pytest.skip("first run finished before the lock became observable")
            pytest.fail("first run did not create .qasr/lock within 90s")

        second = _run(qasr, integration_env, *args, timeout=120)
        assert_exit(second, 2, context="second process on the same out-dir")
        text = (second.stdout + second.stderr).lower()
        assert "lock" in text or "another" in text, text[:2000]
    finally:
        try:
            first.wait(timeout=900)
        except Exception:
            first.kill()
            first.wait()
    assert first.returncode == 0, first.stderr.read() if first.stderr else "first run failed"


def test_fail_fast_leaves_no_running_state(qasr, integration_env):
    """A fail-fast abort must settle every item, not leave work stuck in running."""
    good = copy_wav(integration_env.audio, qasr.work / "good.wav")
    bad = zero_byte(qasr.work / "bad.wav")
    manifest = write_manifest(
        qasr.work / "fail-fast.jsonl",
        [
            {"id": "bad", "audio": str(bad), "formats": ["txt"]},
            {"id": "good", "audio": str(good), "formats": ["txt"]},
        ],
    )
    out = qasr.work / "fail-fast-out"
    proc = _run(
        qasr,
        integration_env,
        "transcribe",
        "--manifest",
        str(manifest),
        "-o",
        str(out),
        "--fail-fast",
        timeout=900,
    )
    assert_exit(proc, 6, context="fail-fast partial failure")

    latest: dict[str, dict] = {}
    for record in _state_records(out):
        item_id = record.get("id")
        if isinstance(item_id, str):
            latest[item_id] = record

    assert set(latest) == {"bad", "good"}, f"missing state records: {latest}"
    assert latest["bad"].get("status") == "failed", latest
    assert latest["good"].get("status") in {"done", "skipped", "failed"}, (
        f"fail-fast left an unprocessed item non-terminal: {latest['good']}"
    )
    running = [record for record in latest.values() if record.get("status") == "running"]
    assert not running, f"fail-fast left running state: {running}"

    payload = parse_json(proc)
    summary = payload.get("summary") or {}
    actual_done = sum(1 for record in latest.values() if record.get("status") == "done")
    assert summary.get("ok") == actual_done, (
        f"summary.ok={summary.get('ok')} but state has {actual_done} done item(s): {summary}"
    )


def test_kill_9_then_resume_recovers(qasr, integration_env):
    manifest, out = _manifest(qasr, integration_env, count=10)
    args = ("transcribe", "--manifest", str(manifest), "-o", str(out), "--resume")

    first = qasr.popen("--json", *integration_env.model_args, *args, env={"QASR_OFFLINE": "1"})
    try:
        got_done = _wait_until(lambda: bool(_done_ids(out)), timeout=240)
        if not got_done:
            if first.poll() is not None:
                pytest.skip("run finished before kill -9 could be injected")
            pytest.fail("no completed item appeared in state.jsonl within 240s")

        done = _done_ids(out)
        if len(done) >= 10:
            pytest.skip("run completed before kill -9 could be injected")
        job_id = sorted(done)[0]
        outputs = _outputs_for(out, job_id)
        assert outputs, f"no output for completed job {job_id}"
        mtimes = {path: path.stat().st_mtime_ns for path in outputs}
    finally:
        first.kill()
        first.wait(timeout=60)

    resumed = _run(qasr, integration_env, *args, timeout=900)
    assert_exit(resumed, 0, context="resume after kill -9")
    payload = parse_json(resumed)
    summary = payload.get("summary", {})
    assert summary.get("total") == 10, summary
    assert summary.get("ok", 0) + summary.get("skipped", 0) == 10, summary
    assert summary.get("skipped", 0) >= 1, f"resume re-ran already-completed work: {summary}"

    for path, mtime in mtimes.items():
        assert path.exists(), f"completed output disappeared: {path}"
        assert path.stat().st_mtime_ns == mtime, f"completed output was rewritten: {path}"

    # state.jsonl is append-only: historical "running" lines are expected.
    # Only the latest record per id describes the settled state.
    latest: dict[str, dict] = {}
    for record in _state_records(out):
        item_id = record.get("id")
        if isinstance(item_id, str):
            latest[item_id] = record
    active = [record for record in latest.values() if record.get("status") == "running"]
    assert not active, f"resume left running records: {active}"
    assert all(
        record.get("status") in {"done", "skipped"} for record in latest.values()
    ), f"resume left non-terminal state: {latest}"


def test_sigint_exits_130_and_resume_continues(qasr, integration_env):
    manifest, out = _manifest(qasr, integration_env, count=12)
    args = ("transcribe", "--manifest", str(manifest), "-o", str(out), "--resume")

    first = qasr.popen("--json", *integration_env.model_args, *args, env={"QASR_OFFLINE": "1"})
    started = _wait_until(
        lambda: (out / ".qasr" / "lock").exists() or bool(_state_records(out)),
        timeout=90,
    )
    if not started and first.poll() is not None:
        pytest.skip("run finished before SIGINT could be injected")
    assert started, "run never started"

    # Give the scheduler a moment to enter inference, then interrupt.
    time.sleep(2.0)
    if first.poll() is not None:
        pytest.skip("run finished before SIGINT could be injected")
    first.send_signal(signal.SIGINT)
    try:
        first.wait(timeout=120)
    except Exception:
        first.kill()
        first.wait()
        pytest.fail("process ignored SIGINT for 120s")

    assert first.returncode == 130, (
        f"SIGINT must exit 130, got {first.returncode}\n"
        f"stderr:\n{first.stderr.read() if first.stderr else ''}"
    )

    resumed = _run(qasr, integration_env, *args, timeout=900)
    assert_exit(resumed, 0, context="resume after SIGINT")
    summary = parse_json(resumed).get("summary", {})
    assert summary.get("total") == 12, summary
    assert summary.get("ok", 0) + summary.get("skipped", 0) == 12, summary
