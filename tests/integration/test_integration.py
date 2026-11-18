"""Real-model integration tests.

Run with:
    RUN_INTEGRATION=1 QASR_TEST_MODEL_DIR=/path/to/models pytest -m integration
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qasr_cli.cli import main

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1",
                       reason="set RUN_INTEGRATION=1 to run model-backed tests"),
]

MODEL_ROOT = Path(os.environ.get("QASR_TEST_MODEL_DIR",
                                 "/Users/walter/workspace/QwenLM/Qwen3-ASR/models"))
AUDIO = Path(os.environ.get("QASR_TEST_AUDIO", "/tmp/qasr_asr_zh.wav"))
EXPECTED = "甚至出现交易几乎停滞的情况。"


def _require_assets() -> None:
    if not (MODEL_ROOT / "Qwen3-ASR-0.6B").is_dir():
        pytest.skip(f"missing 0.6B model under {MODEL_ROOT}")
    if not AUDIO.is_file():
        pytest.skip(f"missing test audio {AUDIO}")


def test_verify_models():
    _require_assets()
    from qasr_cli.catalog import resolve_model, verify_model
    from qasr_cli.config import DEFAULTS, Config

    cfg = Config(values={**DEFAULTS, "model_dir": str(MODEL_ROOT)})
    assert verify_model(resolve_model("0.6b", cfg))["status"] == "complete"
    assert verify_model(resolve_model("aligner", cfg))["status"] == "complete"
    payload = verify_model(resolve_model("1.7b", cfg))
    assert payload["status"] == "complete"


def test_transcribe_text_and_language(capsys):
    _require_assets()
    code = main(["transcribe", str(AUDIO), "--model-dir", str(MODEL_ROOT),
                 "--model", "0.6b", "--language", "zh", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload
    assert "交易" in payload["result"]["text"]
    assert payload["result"]["language"] == "Chinese"
    assert payload["result"]["language_source"] == "forced"
    assert payload["metrics"]["infer_ms"] > 0


def test_transcribe_auto_lid(capsys):
    _require_assets()
    code = main(["transcribe", str(AUDIO), "--model-dir", str(MODEL_ROOT),
                 "--model", "0.6b", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload
    assert payload["result"]["language"] in ("Chinese", "Chinese,English")
    assert payload["result"]["language_source"] == "auto"


def test_transcribe_timestamps_srt(capsys, tmp_path):
    _require_assets()
    out = tmp_path / "out"
    code = main(["transcribe", str(AUDIO), "--model-dir", str(MODEL_ROOT),
                 "--model", "0.6b", "--language", "zh", "--timestamps",
                 "-f", "srt,json", "-o", str(out)])
    assert code == 0
    srt = (out / "qasr_asr_zh.srt").read_text(encoding="utf-8")
    assert "-->" in srt
    payload = json.loads((out / "qasr_asr_zh.json").read_text(encoding="utf-8"))
    assert payload["result"]["timestamp_source"] == "forced_aligner"
    assert len(payload["result"]["segments"]) >= 13


def test_align_known_text(capsys):
    _require_assets()
    code = main(["align", str(AUDIO), "--model-dir", str(MODEL_ROOT),
                 "--text", EXPECTED, "--language", "zh", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload
    items = payload["result"]["items"]
    assert len(items) == 13
    starts = [item["start"] for item in items]
    assert starts == sorted(starts)
    assert payload["result"]["coverage"] >= 0.9
