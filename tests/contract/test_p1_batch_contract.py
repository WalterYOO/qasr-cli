"""P1 N5: stdin lists, skip-existing/retry flags and tsv/md output."""

from __future__ import annotations

import io
import json

import pytest

from qasr_cli import transcribe as core
from qasr_cli.backends import Segment, TranscribeResult
from qasr_cli.cli import main
from qasr_cli.render import render_format
from tests.conftest import write_wav


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def ensure_loaded(self, need_align=False):
        return None

    def transcribe(self, jobs, opts):
        self.calls += len(jobs)
        return [
            TranscribeResult(
                text=f"text-{job.item.id}",
                language="Chinese",
                language_source="auto",
                segments=[Segment("你", 0.0, 0.5, "word")],
            )
            for job in jobs
        ]


@pytest.fixture
def fake_backend(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(core, "build_transformers_backend", lambda *a, **k: backend)
    monkeypatch.setattr(core, "build_vllm_backend", lambda *a, **k: backend)
    return backend


@pytest.fixture
def model_dir(model_factory):
    model_factory("Qwen3-ASR-1.7B")
    return None


def test_stdin_list_reads_paths_and_keeps_positional_order(
    capsys, monkeypatch, tmp_path, model_dir, fake_backend
):
    first = write_wav(tmp_path / "a.wav")
    second = write_wav(tmp_path / "b.wav", freq=660.0)
    third = write_wav(tmp_path / "c.wav", freq=880.0)
    out = tmp_path / "out"
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{first}\n\n{second}\n"))
    code = main([
        "transcribe", str(third), "--device", "cpu", "--dtype", "fp32",
        "--stdin-list", "-o", str(out), "--jsonl",
    ])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert code == 0
    assert [item["input"]["id"] for item in lines] == ["a", "b", "c"]
    assert fake_backend.calls == 3


def test_stdin_list_rejects_manifest(capsys, tmp_path, model_dir):
    code = main([
        "transcribe", "--stdin-list", "--json",
        "--manifest", str(tmp_path / "m.jsonl"), "--out-dir", str(tmp_path / "o"),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert "stdin-list" in payload["error"]["message"]


def test_retry_conflicts_with_fail_fast(capsys, tmp_path, model_dir):
    wav = write_wav(tmp_path / "a.wav")
    code = main([
        "transcribe", str(wav), "--device", "cpu", "--dtype", "fp32",
        "--retry", "--fail-fast", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert "--retry" in payload["error"]["message"]


def test_tsv_escapes_tabs_and_newlines():
    item = {
        "input": {"id": "a"},
        "language": "Chinese",
        "result": {"text": None},
    }
    segments = [
        Segment("hello\tworld\nagain", 0.0, 1.0, "word"),
    ]
    text = render_format(
        "tsv", item=item, result=TranscribeResult(text="", language="Chinese", language_source="auto", segments=segments)
    )
    lines = text.splitlines()
    assert lines[0] == "id\tstart\tend\ttext\tlanguage"
    assert lines[1] == "a\t0.000\t1.000\thello world again\tChinese"


def test_tsv_without_segments_has_empty_times():
    item = {"input": {"id": "a"}, "language": "English", "result": {"text": "hello"}}
    text = render_format("tsv", item=item, result=None)
    assert text.splitlines()[1] == "a\t\t\thello\tEnglish"


def test_markdown_sections_and_lists():
    item = {"input": {"id": "a"}, "language": "Chinese"}
    segments = [Segment("你好", 0.0, 0.5, "word"), Segment("再见", 0.5, 1.0, "word")]
    text = render_format(
        "md", item=item, result=TranscribeResult(text="", language="Chinese", language_source="auto", segments=segments)
    )
    assert "## a" in text
    assert "- [0.000 - 0.500] 你好" in text

    plain = {"input": {"id": "b"}, "language": "English", "result": {"text": "hi"}}
    assert "hi" in render_format("md", item=plain, result=None)


def test_tsv_output_file_via_out_dir(
    capsys, tmp_path, model_dir, fake_backend
):
    wav = write_wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    code = main([
        "transcribe", str(wav), "--device", "cpu", "--dtype", "fp32",
        "-o", str(out), "--format", "tsv", "--json",
    ])
    assert code == 0
    assert (out / "a.tsv").read_text(encoding="utf-8").startswith(
        "id\tstart\tend\ttext\tlanguage\n"
    )


def test_tsv_language_comes_from_result_payload(
    capsys, tmp_path, model_factory, fake_backend
):
    model_factory("Qwen3-ASR-1.7B")
    wav = write_wav(tmp_path / "a.wav")
    out = tmp_path / "out"
    code = main([
        "transcribe", str(wav), "--device", "cpu", "--dtype", "fp32",
        "--language", "zh", "-o", str(out), "--format", "tsv", "--json",
    ])
    assert code == 0, capsys.readouterr()
    rows = (out / "a.tsv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "id\tstart\tend\ttext\tlanguage"
    assert rows[1].split("\t")[4] == "Chinese"


def test_markdown_language_not_required_but_tsv_dict_result():
    from qasr_cli.render import render_format

    item = {
        "input": {"id": "a"},
        "result": {
            "text": "hello",
            "language": "English",
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0, "kind": "word"}],
        },
    }
    text = render_format("tsv", item=item, result=None)
    assert text.splitlines()[1].endswith("\tEnglish")
