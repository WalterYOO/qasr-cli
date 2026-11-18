import json
import os

import pytest

from qasr_cli import jobs
from qasr_cli.errors import UsageError


def test_atomic_write_and_fsync(tmp_path):
    target = tmp_path / "out" / "file.txt"
    jobs.atomic_write_text(target, "hello")
    assert target.read_text() == "hello"
    assert not list(target.parent.glob("*.tmp"))


def test_job_key_deterministic():
    kwargs = dict(
        request={"language": "Chinese", "timestamps": True},
        audio_sha256="a" * 64,
        model_fingerprint="sha256:m",
        aligner_fingerprint="sha256:a",
        backend="transformers",
        output_rel="clip.srt",
    )
    first = jobs.job_key(**kwargs)
    second = jobs.job_key(**kwargs)
    assert first == second
    changed = jobs.job_key(**{**kwargs, "audio_sha256": "b" * 64})
    assert changed != first


def test_state_roundtrip_and_latest_wins(tmp_path):
    state = tmp_path / "state.jsonl"
    jobs.append_jsonl(state, {"schema_version": 1, "id": "a", "status": "running"})
    jobs.append_jsonl(state, {"schema_version": 1, "id": "a", "status": "done"})
    latest = jobs.read_state(state)
    assert latest["a"]["status"] == "done"


def test_recover_running_marks_pending(tmp_path):
    state = tmp_path / "state.jsonl"
    jobs.append_jsonl(state, {"schema_version": 1, "id": "a", "status": "running",
                              "job_key": "k", "outputs": []})
    touched = jobs.recover_running(state, ["a"])
    assert touched == ["a"]
    assert jobs.read_state(state)["a"]["status"] == "pending"


def test_lock_blocks_live_pid(tmp_path):
    lock = jobs.OutputLock(tmp_path)
    lock.acquire()
    second = jobs.OutputLock(tmp_path)
    with pytest.raises(UsageError, match="locked"):
        second.acquire()
    lock.release()
    second.acquire()
    second.release()


def test_stale_lock_is_replaced(tmp_path):
    lock_path = tmp_path / ".qasr" / "lock"
    lock_path.parent.mkdir()
    lock_path.write_text(json.dumps({"pid": 999999999, "host": os.uname().nodename}))
    lock = jobs.OutputLock(tmp_path)
    lock.acquire()
    assert json.loads(lock_path.read_text())["pid"] == os.getpid()
    lock.release()


def test_outputs_match(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hi")
    digest = jobs.sha256_file(target)
    assert jobs.outputs_match([{"path": str(target), "sha256": digest}])
    target.write_text("changed")
    assert not jobs.outputs_match([{"path": str(target), "sha256": digest}])


def test_should_skip():
    record = {"status": "done", "job_key": "k", "outputs": []}
    assert jobs.should_skip(record, "k", resume=True, overwrite=False) is False  # no outputs
    assert jobs.should_skip(record, "other", resume=True, overwrite=False) is False
    assert jobs.should_skip(record, "k", resume=False, overwrite=False) is False
    assert jobs.should_skip(record, "k", resume=True, overwrite=True) is False


def test_manifest_strict_schema(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps({"audio": "a.wav", "unknown": 1}) + "\n", encoding="utf-8")
    with pytest.raises(UsageError, match="unknown field"):
        jobs.load_manifest(path)


def test_manifest_audio_xor_base64(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps({"audio": "a.wav", "audio_base64": "data:audio/wav;base64,AA=="}) + "\n")
    with pytest.raises(UsageError, match="exactly one"):
        jobs.load_manifest(path)


def test_manifest_duplicate_ids(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(
        json.dumps({"id": "x", "audio": "a.wav"}) + "\n"
        + json.dumps({"id": "x", "audio": "b.wav"}) + "\n"
    )
    with pytest.raises(UsageError, match="duplicate id"):
        jobs.load_manifest(path)


def test_manifest_rejects_unsafe_output(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps({"audio": "a.wav", "output": "../escape"}) + "\n")
    with pytest.raises(UsageError, match="\\.\\."):
        jobs.load_manifest(path)


def test_manifest_default_id(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps({"audio": "a.wav"}) + "\n")
    records = jobs.load_manifest(path)
    assert len(records[0]["id"]) == 12
