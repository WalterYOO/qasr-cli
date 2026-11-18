"""Result rendering: txt/json/jsonl payloads and subtitle text (pure Python)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from . import SCHEMA_VERSION
from .backends import AlignResult, TranscribeResult
from .subtitles import Cue, build_cues, render_srt, render_vtt

FORMATS = ("txt", "json", "jsonl", "srt", "vtt", "tsv", "md")
SUBTITLE_FORMATS = ("srt", "vtt")
JSON_FORMATS = ("json", "jsonl")
TABLE_FORMATS = ("tsv", "md")
TSV_HEADER = ("id", "start", "end", "text", "language")


def error_payload(
    command: str,
    error: Dict[str, Any],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "command": command,
        "error": error,
    }
    if extra:
        payload.update(extra)
    return payload


def item_payload(
    *,
    command: str,
    input_info: Dict[str, Any],
    model_info: Dict[str, Any],
    result: Dict[str, Any],
    metrics: Dict[str, Any],
    outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": command,
        "input": input_info,
        "model": model_info,
        "result": result,
        "metrics": metrics,
        "outputs": outputs or [],
    }


def item_error_payload(
    *,
    command: str,
    input_info: Dict[str, Any],
    error: Dict[str, Any],
    model_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "command": command,
        "input": input_info,
        "error": error,
    }
    if model_info:
        payload["model"] = model_info
    return payload


def result_payload(result: TranscribeResult) -> Dict[str, Any]:
    return {
        "text": result.text,
        "language": result.language,
        "language_source": result.language_source,
        "segments": [seg.to_dict() for seg in result.segments],
        "timestamp_source": result.timestamp_source,
        "text_transform": result.text_transform,
        "warnings": list(result.warnings),
    }


def align_result_payload(result: AlignResult, coverage: Optional[float]) -> Dict[str, Any]:
    return {
        "items": [
            {"text": seg.text, "start": round(seg.start, 3), "end": round(seg.end, 3)}
            for seg in result.items
        ],
        "coverage": coverage,
        "warnings": list(result.warnings),
    }


def batch_payload(
    items: Sequence[Dict[str, Any]],
    errors: Sequence[Dict[str, Any]],
    *,
    total: int,
    skipped: int = 0,
) -> Dict[str, Any]:
    ok_count = sum(1 for item in items if item.get("ok") and not item.get("skipped"))
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": len(errors) == 0,
        "command": "transcribe",
        "summary": {
            "total": total,
            "ok": ok_count,
            "failed": len(errors),
            "skipped": skipped,
        },
        "items": list(items),
        "errors": list(errors),
    }


def subtitle_cues(result: TranscribeResult, audio_duration_s: Optional[float]) -> List[Cue]:
    # Prefer token-level items so cue splitting keeps intra-sentence pauses
    # even when JSON segments were aggregated to sentence granularity.
    segments = getattr(result, "raw_segments", None) or result.segments
    return build_cues(segments, audio_duration_s=audio_duration_s)


def _item_id(item: Dict[str, Any]) -> str:
    info = item.get("input") or {}
    value = info.get("id") or info.get("source") or ""
    return str(value)


def _item_language(item: Dict[str, Any]) -> str:
    # transcribe keeps language inside result; align puts it at the top level.
    payload = item.get("result") or {}
    return str(payload.get("language") or item.get("language") or "")


def _item_segments(
    item: Dict[str, Any], result: Optional[TranscribeResult]
) -> List[Any]:
    if result is not None:
        return list(result.segments)
    payload = item.get("result") or {}
    return list(payload.get("items") or [])


def _field(seg: Any, name: str, default: Any = None) -> Any:
    if isinstance(seg, dict):
        return seg.get(name, default)
    return getattr(seg, name, default)


def _cell(value: Any) -> str:
    """Escape tabs and newlines so one record stays on one TSV row."""
    return " ".join(str(value or "").split())


def render_tsv(item: Dict[str, Any], segments: Sequence[Any]) -> str:
    lines = ["\t".join(TSV_HEADER)]
    if segments:
        for seg in segments:
            lines.append(
                "\t".join(
                    [
                        _cell(_item_id(item)),
                        f"{float(_field(seg, 'start', 0.0)):.3f}",
                        f"{float(_field(seg, 'end', 0.0)):.3f}",
                        _cell(_field(seg, "text", "")),
                        _cell(_item_language(item)),
                    ]
                )
            )
    else:
        payload = item.get("result") or {}
        text = payload.get("text") or ""
        lines.append(
            "\t".join([_cell(_item_id(item)), "", "", _cell(text), _cell(_item_language(item))])
        )
    return "\n".join(lines) + "\n"


def render_markdown(item: Dict[str, Any], segments: Sequence[Any]) -> str:
    lines = [f"## {_item_id(item)}", ""]
    if segments:
        for seg in segments:
            lines.append(
                f"- [{float(_field(seg, 'start', 0.0)):.3f} - "
                f"{float(_field(seg, 'end', 0.0)):.3f}] {_field(seg, 'text', '')}"
            )
    else:
        text = (item.get("result") or {}).get("text") or ""
        lines.append(str(text))
    return "\n".join(lines) + "\n"


def render_format(
    fmt: str,
    *,
    item: Dict[str, Any],
    result: Optional[TranscribeResult] = None,
    audio_duration_s: Optional[float] = None,
    cues: Optional[Sequence[Cue]] = None,
) -> str:
    """Render one item in a file-output format."""
    if fmt == "txt":
        text = result.text if result is not None else ""
        return text.rstrip("\n") + "\n" if text else ""
    if fmt == "json":
        return json.dumps(item, ensure_ascii=False, indent=2) + "\n"
    if fmt == "jsonl":
        return json.dumps(item, ensure_ascii=False, sort_keys=False) + "\n"
    if fmt == "srt":
        return render_srt(list(cues or []))
    if fmt == "vtt":
        return render_vtt(list(cues or []))
    if fmt == "tsv":
        return render_tsv(item, _item_segments(item, result))
    if fmt == "md":
        return render_markdown(item, _item_segments(item, result))
    raise ValueError(f"unknown format: {fmt}")


def stdout_text(result: TranscribeResult) -> str:
    return result.text


def dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def dump_json_pretty(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
