"""Optional P1 integration tests (RUN_INTEGRATION=1 + real 0.6B models)."""

from __future__ import annotations

import pytest
from fake_openai import json_response
from helpers import assert_exit, parse_json

pytestmark = pytest.mark.integration


def _env(integration_env: dict[str, str]) -> dict[str, str]:
    return {
        "QASR_MODEL_DIR": integration_env["QASR_TEST_MODEL_DIR"],
        "QASR_OFFLINE": "1",
    }


def test_local_transcribe_with_real_model(qasr, integration_env):
    audio = integration_env["QASR_TEST_AUDIO"]
    proc = qasr.run(
        "transcribe", audio, "--language", "zh", "--json",
        env=_env(integration_env), timeout=300,
    )
    assert_exit(proc, 0, context="integration transcribe")
    payload = parse_json(proc)
    assert payload["result"]["text"].strip()
    assert payload["model"]["backend"] == "transformers"


def test_short_audio_timestamps_monotonic(qasr, integration_env):
    audio = integration_env["QASR_TEST_AUDIO"]
    proc = qasr.run(
        "transcribe", audio, "--language", "zh", "--timestamps", "--json",
        env=_env(integration_env), timeout=600,
    )
    assert_exit(proc, 0, context="integration timestamps")
    segments = parse_json(proc)["result"]["segments"]
    previous = -1.0
    for segment in segments:
        assert segment["end"] >= segment["start"]
        assert segment["start"] >= previous - 1e-3, segment
        previous = segment["end"]


def test_align_with_real_model(qasr, integration_env, tmp_path):
    audio = integration_env["QASR_TEST_AUDIO"]
    first = qasr.run(
        "transcribe", audio, "--language", "zh", "--json",
        env=_env(integration_env), timeout=300,
    )
    assert_exit(first, 0, context="integration align seed")
    text = parse_json(first)["result"]["text"]
    reference = tmp_path / "reference.txt"
    reference.write_text(text, encoding="utf-8")
    proc = qasr.run(
        "align", audio, "--text-file", reference, "--language", "zh", "--json",
        env=_env(integration_env), timeout=600,
    )
    assert_exit(proc, 0, context="integration align")
    payload = parse_json(proc)
    items = payload["result"]["items"]
    previous = -1.0
    for item in items:
        assert item["end"] >= item["start"]
        assert item["start"] >= previous - 1e-3
        previous = item["end"]
    assert float(payload["result"]["coverage"]) >= 0.5


def test_remote_fake_with_real_audio(qasr, integration_env, fake_openai):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>远程集成文本"}
    )
    proc = qasr.run(
        "--server", server.url, "transcribe", integration_env["QASR_TEST_AUDIO"],
        "--json", timeout=120,
    )
    assert_exit(proc, 0, context="integration remote")
    assert parse_json(proc)["result"]["text"] == "远程集成文本"


@pytest.mark.gpu
def test_remote_one_hour_requires_external_flac():
    pytest.skip("1h FLAC and gpu06 tunnel are provided by the coordinator")
