"""N3: vLLM backend availability, auto selection, and safe failure."""

from __future__ import annotations

import sys

import pytest
from helpers import assert_exit, find_check, parse_json

pytestmark = pytest.mark.n3


def test_explicit_vllm_on_macos_exits_3_without_traceback(qasr, short_wav):
    if sys.platform != "darwin":
        pytest.skip("macOS-specific platform gate")
    proc = qasr.run("--backend", "vllm", "transcribe", short_wav, "--json", timeout=30)
    assert_exit(proc, 3, context="explicit vllm on macOS")
    text = proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    lowered = text.lower()
    assert "vllm" in lowered
    assert "linux" in lowered or "cuda" in lowered or "gpu" in lowered


def test_explicit_vllm_does_not_fall_back_to_transformers(qasr, short_wav):
    if sys.platform != "darwin":
        pytest.skip("macOS-specific platform gate")
    proc = qasr.run("--backend", "vllm", "transcribe", short_wav, "--json", timeout=30)
    assert_exit(proc, 3, context="vllm no silent fallback")
    payload = parse_json(proc)
    assert payload["ok"] is False
    assert "backend_fallback" not in payload
    dumped = str(payload).lower()
    assert "fallback" not in dumped or payload.get("backend_fallback") is False


def test_auto_on_macos_selects_transformers(qasr, short_wav):
    if sys.platform != "darwin":
        pytest.skip("macOS-specific auto selection")
    proc = qasr.run(
        "--backend", "auto", "--dry-run", "--json", "transcribe", short_wav
    )
    assert_exit(proc, 0, context="auto dry-run")
    payload = parse_json(proc)
    plan = payload["plan"]
    assert plan["backend"] == "transformers"
    # In a torch-free review env --dry-run must not import torch to probe,
    # so the plan may legitimately keep the requested "auto" placeholder.
    assert plan["device"] in {"mps", "cpu", "auto"}, plan


def test_gpu_memory_utilization_is_vllm_only(qasr, short_wav):
    proc = qasr.run(
        "--backend", "transformers", "transcribe", short_wav,
        "--gpu-memory-utilization", "0.8", "--json",
    )
    assert_exit(proc, 2, context="gpu-memory-utilization with transformers")
    assert "Traceback" not in proc.stderr


def test_gpu_memory_utilization_validation_does_not_crash(qasr, short_wav):
    proc = qasr.run(
        "--backend", "vllm", "transcribe", short_wav,
        "--gpu-memory-utilization", "2", "--json", timeout=30,
    )
    assert proc.returncode in {2, 3}, proc.stderr
    assert "Traceback" not in proc.stderr


def test_vllm_rejects_mps_device(qasr, short_wav):
    if sys.platform != "darwin":
        pytest.skip("MPS is macOS-only")
    proc = qasr.run(
        "--backend", "vllm", "--device", "mps",
        "transcribe", short_wav, "--json", timeout=30,
    )
    assert_exit(proc, 3, context="vllm + mps")
    assert "Traceback" not in proc.stderr


def test_server_conflicts_with_explicit_backend(qasr, fake_openai, short_wav):
    server = fake_openai()
    for backend in ("transformers", "vllm"):
        proc = qasr.run(
            "--server", server.url, "--backend", backend,
            "transcribe", short_wav, "--json", timeout=30,
        )
        assert_exit(proc, 2, context=f"--server + --backend {backend}")
        assert not server.post_requests()


def test_server_with_backend_auto_is_remote(qasr, fake_openai, short_wav):
    server = fake_openai()
    proc = qasr.run(
        "--server", server.url, "--backend", "auto",
        "transcribe", short_wav, "--json",
    )
    assert_exit(proc, 0, context="--server + --backend auto")
    payload = parse_json(proc)
    assert payload["model"]["backend"] == "remote"
    assert len(server.post_requests()) == 1


def test_doctor_reports_vllm_importability(qasr):
    proc = qasr.run("doctor", "--json")
    assert_exit(proc, 0, context="doctor vllm check")
    payload = parse_json(proc)
    check = find_check(payload, "vllm")
    assert check is not None, f"doctor has no vllm check: {payload}"
    assert check.get("status") in {"ok", "warn"}, check
    text = str(check).lower()
    assert "vllm" in text


def test_help_does_not_import_runtime(qasr, import_blocker, repo_root):
    import os

    blocker, marker = import_blocker
    env = {
        "PYTHONPATH": f"{blocker}{os.pathsep}{repo_root}",
        "COLUMNS": "200",
    }
    proc = qasr.run("--help", env=env)
    assert_exit(proc, 0, context="--help import blocker")
    assert not marker.exists(), "--help imported torch/qwen_asr/vllm"
    assert "--server" in proc.stdout
    assert "serve" in proc.stdout


@pytest.mark.gpu
def test_vllm_matches_transformers_on_gpu():
    """Executed on gpu06; evidence is collected by the coordinator."""
    pytest.skip("requires a Linux/CUDA host with vllm; gpu06 evidence is external")
