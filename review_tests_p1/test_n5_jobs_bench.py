"""N5: jobs index/show and bench contracts."""

from __future__ import annotations

import shutil
import time

import pytest
from fake_openai import json_response
from helpers import assert_exit, find_key, parse_json, read_job_index, validate_schema

pytestmark = pytest.mark.n5


def _seed_job(qasr, fake_openai, tmp_path, short_wav, name="job"):
    server = fake_openai()
    files = []
    for index in range(2):
        target = tmp_path / f"{name}{index}.wav"
        shutil.copyfile(short_wav, target)
        files.append(target)
    out = tmp_path / f"out-{name}"
    proc = qasr.run(
        "--server", server.url, "transcribe", *files,
        "--out-dir", out, "--json",
    )
    assert_exit(proc, 0, context=f"seed {name}")
    jobs = read_job_index(qasr.home)
    matches = [job for job in jobs if str(out) in str(job.get("out_dir"))]
    assert len(matches) == 1, jobs
    return matches[0], out


def test_jobs_index_is_written(qasr, fake_openai, tmp_path, short_wav):
    job, out = _seed_job(qasr, fake_openai, tmp_path, short_wav)
    assert job.get("job_id")
    assert job.get("out_dir")
    assert job.get("created_at")
    assert job.get("updated_at")
    assert job.get("command")
    assert job.get("total") == 2
    assert str(out) in str(job["out_dir"])


def test_jobs_list_json_schema(qasr, fake_openai, tmp_path, short_wav):
    job, _out = _seed_job(qasr, fake_openai, tmp_path, short_wav)
    proc = qasr.run("jobs", "list", "--json")
    assert_exit(proc, 0, context="jobs list")
    payload = parse_json(proc)
    validate_schema(payload, "jobs-list.json")
    assert payload["command"] == "jobs"
    entries = payload.get("jobs")
    assert isinstance(entries, list) and entries
    assert entries[0]["job_id"] == job["job_id"]


def test_jobs_show_by_id_and_out_dir(qasr, fake_openai, tmp_path, short_wav):
    job, out = _seed_job(qasr, fake_openai, tmp_path, short_wav)
    for target in (job["job_id"], out):
        proc = qasr.run("jobs", "show", target, "--json")
        assert_exit(proc, 0, context=f"jobs show {target}")
        payload = parse_json(proc)
        validate_schema(payload, "jobs-show.json")
        job_obj = payload.get("job") or payload
        assert str(job_obj.get("out_dir")) in str(out) or str(out) in str(job_obj.get("out_dir"))
        running = [value for value in find_key(payload, "running") if isinstance(value, int)]
        assert not running or max(running) == 0, running
        done = [value for value in find_key(payload, "done") if isinstance(value, int)]
        ok = [value for value in find_key(payload, "ok") if isinstance(value, int)]
        assert 2 in done or 2 in ok, payload


def test_jobs_show_unknown_is_exit_4(qasr):
    for value in ("nope", "../../etc", "/tmp"):
        proc = qasr.run("jobs", "show", value, "--json")
        assert_exit(proc, 4, context=f"jobs show {value}")
        assert "Traceback" not in proc.stderr


def test_jobs_list_limit_and_order(qasr, fake_openai, tmp_path, short_wav):
    first, _out = _seed_job(qasr, fake_openai, tmp_path, short_wav, name="first")
    time.sleep(1.1)
    second, _out2 = _seed_job(qasr, fake_openai, tmp_path, short_wav, name="second")
    proc = qasr.run("jobs", "list", "--json")
    assert_exit(proc, 0, context="jobs list order")
    entries = parse_json(proc)["jobs"]
    assert [entry["job_id"] for entry in entries[:2]] == [second["job_id"], first["job_id"]]
    limited = qasr.run("jobs", "list", "--json", "--limit", "1")
    assert_exit(limited, 0, context="jobs list --limit")
    assert len(parse_json(limited)["jobs"]) == 1


def test_jobs_list_limit_validation(qasr):
    for value in ("0", "-1", "10001", "abc"):
        proc = qasr.run("jobs", "list", "--json", "--limit", value)
        assert_exit(proc, 2, context=f"jobs --limit {value}")


def test_bench_dry_run_has_plan(qasr, tmp_path):
    proc = qasr.run(
        "bench", "--duration", "2", "--iterations", "3", "--dry-run", "--json"
    )
    assert_exit(proc, 0, context="bench dry-run")
    payload = parse_json(proc)
    plan = payload["plan"]
    assert plan.get("duration_s") in {2, 2.0} or plan.get("duration") == 2
    assert plan.get("iterations") == 3
    assert plan.get("synthetic") is True


def test_bench_remote_metrics(qasr, fake_openai):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>bench"}
    )
    proc = qasr.run(
        "--server", server.url, "bench",
        "--duration", "1", "--iterations", "1", "--json", timeout=60,
    )
    assert_exit(proc, 0, context="bench remote")
    payload = parse_json(proc)
    validate_schema(payload, "bench.json")
    body = payload.get("result") or payload
    assert body["iterations"] == 1
    assert body["backend"] == "remote"
    assert 0.5 <= float(body["audio_s"]) <= 1.5
    assert float(body["elapsed_s"]) >= 0
    assert float(body["rtf"]) >= 0
    assert "synthetic" in (proc.stderr + proc.stdout).lower()


def test_bench_remote_warns_about_network(qasr, fake_openai):
    server = fake_openai()
    proc = qasr.run(
        "--server", server.url, "bench",
        "--duration", "1", "--iterations", "1", "--json", timeout=60,
    )
    assert_exit(proc, 0, context="bench remote warning")
    assert "network" in proc.stderr.lower() or "网络" in proc.stderr


def test_bench_iterations_validation(qasr):
    for value in ("-1", "101", "abc"):
        proc = qasr.run("bench", "--iterations", value, "--dry-run", "--json")
        assert_exit(proc, 2, context=f"bench --iterations {value}")


@pytest.mark.parametrize(
    "args",
    [
        ("--iterations", "0"),
        ("--duration", "0"),
    ],
)
def test_bench_zero_values_are_rejected(qasr, args):
    proc = qasr.run("bench", *args, "--dry-run", "--json")
    assert_exit(proc, 2, context=f"bench {args}")


def test_bench_remote_token_is_not_leaked(qasr, fake_openai):
    token = "bench-secret-token-value"
    server = fake_openai(auth_token=token, require_auth=True)
    proc = qasr.run(
        "--server", server.url, "bench",
        "--duration", "1", "--iterations", "1", "--json", timeout=30,
        env={"QASR_SERVE_TOKEN": token},
    )
    assert proc.returncode in {0, 3, 4}, proc
    assert token not in proc.stdout
    assert token not in proc.stderr


@pytest.mark.gpu
def test_bench_vllm_peak_memory():
    pytest.skip("requires CUDA/vLLM; gpu06 evidence is external")
