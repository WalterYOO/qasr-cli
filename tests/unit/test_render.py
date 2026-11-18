import json

from qasr_cli.backends import Segment, TranscribeResult
from qasr_cli.render import (
    batch_payload,
    item_payload,
    render_format,
    result_payload,
    subtitle_cues,
)


def _result(**kwargs):
    return TranscribeResult(
        text=kwargs.get("text", "你好世界"),
        language=kwargs.get("language", "Chinese"),
        language_source=kwargs.get("language_source", "forced"),
        segments=kwargs.get("segments", [
            Segment("你", 0.0, 0.2, "char"),
            Segment("好", 0.2, 0.4, "char"),
        ]),
        timestamp_source=kwargs.get("timestamp_source", "forced_aligner"),
        raw_segments=kwargs.get("raw_segments", []),
    )


def _item(result=None, outputs=None):
    result = result or _result()
    return item_payload(
        command="transcribe",
        input_info={"id": "a", "source": "a.wav", "duration_s": 1.0, "sha256": "x"},
        model_info={"id": "Qwen/Qwen3-ASR-0.6B", "backend": "transformers",
                    "device": "cpu", "dtype": "fp32", "fingerprint": None},
        result=result_payload(result),
        metrics={"load_ms": 0, "infer_ms": 10, "wall_ms": 10, "rtf": 0.01},
        outputs=outputs or [],
    )


def test_result_payload_shape():
    payload = result_payload(_result())
    assert payload["text"] == "你好世界"
    assert payload["language"] == "Chinese"
    assert payload["segments"][0]["kind"] == "char"


def test_item_payload_shape():
    payload = _item()
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["command"] == "transcribe"
    assert payload["input"]["id"] == "a"


def test_batch_payload_summary():
    item = _item()
    payload = batch_payload([item], [], total=1)
    assert payload["summary"] == {"total": 1, "ok": 1, "failed": 0, "skipped": 0}
    assert payload["ok"] is True


def test_batch_payload_with_errors():
    payload = batch_payload([], [{"id": "b", "error": {"kind": "input", "code": 4}}], total=1)
    assert payload["ok"] is False
    assert payload["summary"]["failed"] == 1


def test_render_txt():
    text = render_format("txt", item=_item(), result=_result())
    assert text == "你好世界\n"


def test_render_json_roundtrip():
    text = render_format("json", item=_item(), result=_result())
    parsed = json.loads(text)
    assert parsed["result"]["text"] == "你好世界"


def test_render_jsonl_single_line():
    text = render_format("jsonl", item=_item(), result=_result())
    assert text.count("\n") == 1
    assert json.loads(text)["command"] == "transcribe"


def test_render_srt_requires_segments():
    result = _result()
    text = render_format("srt", item=_item(result), result=result,
                         audio_duration_s=1.0, cues=[])
    assert text == ""


def test_empty_text_txt_is_empty():
    result = _result(text="")
    assert render_format("txt", item=_item(result), result=result) == ""


def test_subtitle_cues_prefer_raw_segments():
    result = _result(
        segments=[Segment("你好", 0.0, 1.2, "sentence")],
        raw_segments=[Segment("你", 0.0, 0.2, "char"),
                      Segment("好", 1.0, 1.2, "char")],
    )
    cues = subtitle_cues(result, audio_duration_s=2.0)
    # The 800 ms pause inside the sentence still splits the subtitle.
    assert len(cues) == 2
