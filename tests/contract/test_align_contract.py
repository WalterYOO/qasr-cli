import json
from pathlib import Path

import jsonschema
import pytest

from qasr_cli import align as align_core
from qasr_cli.align import compute_coverage, normalize_for_coverage
from qasr_cli.backends import AlignResult, Segment
from qasr_cli.cli import main

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
TEXT = "甚至出现交易几乎停滞的情况。"


def validate(name, payload):
    schema = json.loads((SCHEMA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)


class FakeAligner:
    def __init__(self, text=TEXT):
        self.text = text

    def ensure_aligner(self):
        return None

    def align(self, jobs, opts):
        chars = [ch for ch in self.text if ch not in "。！？!?；;，,、"]
        step = 0.2
        items = [Segment(ch, i * step, (i + 1) * step, "char") for i, ch in enumerate(chars)]
        return [AlignResult(items=items)]


@pytest.fixture
def fake_aligner(monkeypatch):
    def _install(text=TEXT):
        monkeypatch.setattr(align_core, "build_transformers_backend",
                            lambda *a, **k: FakeAligner(text))

    return _install


def test_align_json(capsys, wav_file, model_factory, fake_aligner):
    model_factory("Qwen3-ForcedAligner-0.6B")
    fake_aligner()
    code = main(["align", str(wav_file), "--text", TEXT, "--language", "zh", "--device", "cpu", "--dtype", "fp32", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    validate("align", payload)
    assert payload["language"] == "Chinese"
    assert len(payload["result"]["items"]) == 13
    assert payload["result"]["coverage"] == 1.0
    starts = [item["start"] for item in payload["result"]["items"]]
    assert starts == sorted(starts)


def test_align_default_stdout_is_json(capsys, wav_file, model_factory, fake_aligner):
    model_factory("Qwen3-ForcedAligner-0.6B")
    fake_aligner()
    code = main(["align", str(wav_file), "--text", TEXT, "--language", "zh", "--device", "cpu", "--dtype", "fp32"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["command"] == "align"


def test_align_srt_out_dir(capsys, tmp_path, wav_file, model_factory, fake_aligner):
    model_factory("Qwen3-ForcedAligner-0.6B")
    fake_aligner()
    out = tmp_path / "aligned"
    code = main(["align", str(wav_file), "--text", TEXT, "--language", "zh",
                 "-f", "srt,json", "-o", str(out), "--device", "cpu", "--dtype", "fp32"])
    assert code == 0
    assert (out / "clip.srt").is_file()
    assert (out / "clip.json").is_file()
    assert "-->" in (out / "clip.srt").read_text()


def test_align_coverage_gate(capsys, wav_file, model_factory, fake_aligner):
    model_factory("Qwen3-ForcedAligner-0.6B")
    fake_aligner(text="完全无关的内容在这里")
    code = main(["align", str(wav_file), "--text", TEXT, "--language", "zh",
                 "--coverage-min", "0.9", "--device", "cpu", "--dtype", "fp32", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 5
    assert payload["error"]["kind"] == "quality"


def test_align_coverage_disabled(capsys, wav_file, model_factory, fake_aligner):
    model_factory("Qwen3-ForcedAligner-0.6B")
    fake_aligner(text="无关")
    code = main(["align", str(wav_file), "--text", TEXT, "--language", "zh",
                 "--coverage-min", "0", "--device", "cpu", "--dtype", "fp32", "--json"])
    assert code == 0


class FakeASR:
    language = "English"

    def ensure_loaded(self, need_align=False):
        return None

    def transcribe(self, jobs, opts):
        from qasr_cli.backends import TranscribeResult

        return [
            TranscribeResult(
                text="hello world",
                language=self.language,
                language_source="auto",
            )
            for _ in jobs
        ]


def test_align_auto_latin_uses_lid_probe(capsys, wav_file, model_factory, monkeypatch):
    model_factory("Qwen3-ForcedAligner-0.6B")
    model_factory("Qwen3-ASR-1.7B")

    def _build(path, **kwargs):
        if "ForcedAligner" in str(path):
            return FakeAligner("hello world")
        return FakeASR()

    monkeypatch.setattr(align_core, "build_transformers_backend", _build)
    code = main(["align", str(wav_file), "--text", "hello world", "--language", "auto", "--device", "cpu", "--dtype", "fp32", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0, payload
    assert payload["language"] == "English"
    assert payload["language_source"] == "lid_probe"
    assert "lid_probe" in captured.err


def test_align_auto_chinese(capsys, wav_file, model_factory, fake_aligner):
    model_factory("Qwen3-ForcedAligner-0.6B")
    fake_aligner()
    code = main(["align", str(wav_file), "--text", TEXT, "--language", "auto", "--device", "cpu", "--dtype", "fp32", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["language"] == "Chinese"


def test_align_rejects_non_alignable_language(capsys, wav_file, model_factory):
    model_factory("Qwen3-ForcedAligner-0.6B")
    code = main(["align", str(wav_file), "--text", TEXT, "--language", "th", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 4
    assert "cannot be aligned" in payload["error"]["message"]


def test_align_accepts_181s_with_windowing(
    monkeypatch, capsys, tmp_path, model_factory
):
    model_factory("Qwen3-ForcedAligner-0.6B")
    model_factory("Qwen3-ASR-1.7B")
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"RIFF")
    import numpy as np

    import qasr_cli.audio as audio_mod
    from qasr_cli.audio import AudioInfo

    def fake_info(_p):
        return AudioInfo(Path(_p), 181.0, 16000, 1, 181 * 16000, "WAV")

    monkeypatch.setattr(audio_mod, "probe", fake_info)
    monkeypatch.setattr(align_core, "probe", fake_info)
    monkeypatch.setattr(
        align_core, "read_window_16k",
        lambda *a, **k: np.zeros(16000, dtype="float32"),
    )

    def _build(path, **kwargs):
        if "ForcedAligner" in str(path):
            return FakeAligner(TEXT)
        return FakeASR()

    monkeypatch.setattr(align_core, "build_transformers_backend", _build)
    code = main([
        "align", str(audio), "--text", TEXT, "--language", "zh",
        "--device", "cpu", "--dtype", "fp32", "--json", "--coverage-min", "0",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload
    plan = payload["window_plan"]
    assert plan["count"] == 2
    starts = [item["start"] for item in payload["result"]["items"]]
    assert starts == sorted(starts)


def test_align_requires_one_text_source(capsys, wav_file, model_factory):
    model_factory("Qwen3-ForcedAligner-0.6B")
    code = main(["align", str(wav_file), "--language", "zh", "--device", "cpu", "--dtype", "fp32", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert "exactly one" in payload["error"]["message"]


def test_coverage_normalization():
    assert normalize_for_coverage("你好，世界！") == "你好世界"
    assert compute_coverage("你好，世界！", "你好世界") == 1.0
    assert compute_coverage("abcdef", "abc") == 0.5
    assert compute_coverage("", "") == 1.0


@pytest.mark.parametrize(
    "args",
    [
        ["--window-seconds", "181"],
        ["--window-seconds", "9"],
        ["--window-overlap", "11"],
        ["--window-overlap", "-1"],
    ],
)
def test_align_rejects_invalid_window_params(capsys, wav_file, model_factory, args):
    model_factory("Qwen3-ForcedAligner-0.6B")
    code = main([
        "align", str(wav_file), "--text", TEXT, "--language", "zh", *args, "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["error"]["kind"] == "usage"


def test_align_dry_run_reports_window_plan(monkeypatch, capsys, tmp_path, model_factory):
    from qasr_cli.audio import AudioInfo

    model_factory("Qwen3-ForcedAligner-0.6B")
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"RIFF")
    import qasr_cli.audio as audio_mod

    def fake_info(_p):
        return AudioInfo(Path(_p), 181.0, 16000, 1, 181 * 16000, "WAV")

    monkeypatch.setattr(audio_mod, "probe", fake_info)
    monkeypatch.setattr(align_core, "probe", fake_info)
    code = main([
        "align", str(audio), "--text", TEXT, "--language", "zh", "--dry-run", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    validate("dry-run", payload)
    plan = payload["plan"]
    assert plan["window_seconds"] == 180.0
    assert len(plan["windows"]) == 2
    assert plan["windows"][-1]["core_end_s"] == 181.0
