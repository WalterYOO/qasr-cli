"""Pure helpers for the black-box qasr review suite.

This module deliberately imports nothing from ``qasr_cli``. Tests should be
able to run against an installed wheel, a source checkout, or a colleague's
worktree without coupling to internal names.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import unicodedata
import wave
from array import array
from pathlib import Path
from typing import Any, Iterable, Sequence

import pytest

# ---------------------------------------------------------------------------
# JSON / process assertions
# ---------------------------------------------------------------------------

def parse_json(proc) -> dict[str, Any]:
    raw = proc.stdout.strip()
    if not raw:
        pytest.fail(
            f"stdout is empty (exit {proc.returncode})\n"
            f"stderr:\n{proc.stderr.strip()}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"stdout is not a single JSON value: {exc}\n"
            f"stdout:\n{proc.stdout[:4000]}\n"
            f"stderr:\n{proc.stderr[:4000]}"
        )
    if not isinstance(payload, dict):
        pytest.fail(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def parse_jsonl(proc) -> list[dict[str, Any]]:
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        pytest.fail(
            f"stdout has no JSONL lines (exit {proc.returncode})\n"
            f"stderr:\n{proc.stderr.strip()}"
        )
    parsed: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"stdout line {index} is not JSON: {exc}\n"
                f"line: {line[:2000]}\n"
                f"stderr:\n{proc.stderr[:4000]}"
            )
        if not isinstance(value, dict):
            pytest.fail(f"stdout line {index} is not a JSON object")
        parsed.append(value)
    return parsed


def assert_exit(proc, expected: int | Iterable[int], *, context: str = "") -> None:
    allowed = {expected} if isinstance(expected, int) else set(expected)
    if proc.returncode not in allowed:
        prefix = f"{context}: " if context else ""
        pytest.fail(
            f"{prefix}unexpected exit code {proc.returncode}, expected {sorted(allowed)}\n"
            f"stdout:\n{proc.stdout[:4000]}\n"
            f"stderr:\n{proc.stderr[:4000]}"
        )


def assert_error(payload: dict[str, Any], *, code: int | None = None, kind: str | None = None) -> dict[str, Any]:
    assert payload.get("ok") is False, f"expected ok=false, got: {payload!r}"
    error = payload.get("error")
    assert isinstance(error, dict), f"missing error object: {payload!r}"
    if code is not None:
        assert error.get("code") == code, f"error.code={error.get('code')!r}, want {code}: {error!r}"
    if kind is not None:
        assert error.get("kind") == kind, f"error.kind={error.get('kind')!r}, want {kind!r}: {error!r}"
    return error


def find_values(payload: Any, key: str) -> list[Any]:
    """Return every value for ``key`` anywhere in a nested JSON payload."""
    found: list[Any] = []
    if isinstance(payload, dict):
        for name, value in payload.items():
            if name == key:
                found.append(value)
            found.extend(find_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(find_values(value, key))
    return found


def find_status(payload: Any) -> str | None:
    for value in find_values(payload, "status"):
        if isinstance(value, str) and value in {"complete", "incomplete", "missing"}:
            return value
    return None


# ---------------------------------------------------------------------------
# Audio fixtures (stdlib only, no numpy / soundfile needed to *create* them)
# ---------------------------------------------------------------------------

def make_wav(
    path: str | Path,
    *,
    seconds: float = 1.0,
    sample_rate: int = 16_000,
    channels: int = 1,
    silent: bool = False,
    freq: float = 440.0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(0, int(round(seconds * sample_rate)))

    if silent or frames == 0:
        raw = b"\x00\x00" * (frames * channels)
    else:
        data = array("h")
        for index in range(frames):
            value = int(0.2 * 32767 * _sin(2.0 * 3.141592653589793 * freq * index / sample_rate))
            data.extend([value] * channels)
        raw = data.tobytes()
        if sys.byteorder == "big":
            data.byteswap()
            raw = data.tobytes()

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(raw)
    return path


def _sin(value: float) -> float:
    # math.sin is fine; the tiny wrapper keeps the module import list obvious.
    import math

    return math.sin(value)


def empty_wav(path: str | Path) -> Path:
    return make_wav(path, seconds=0)


def zero_byte(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def truncated_wav(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00")
    return path


def text_as_wav(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is definitely not a wave file\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def write_manifest(path: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    return write_jsonl(path, records)


# ---------------------------------------------------------------------------
# Fake model directories for ``models verify`` (no torch required)
# ---------------------------------------------------------------------------

def safetensors_blob(data_bytes: int = 32) -> bytes:
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "w": {
                "dtype": "F16",
                "shape": [max(1, data_bytes // 2)],
                "data_offsets": [0, data_bytes],
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"\x00" * data_bytes


def make_model_dir(
    path: str | Path,
    *,
    aria2: bool = False,
    missing_shard: bool = False,
    truncate_shard: bool = False,
) -> Path:
    """Create a minimal, *plausible* HF model directory.

    ``verify`` is expected to look at the safetensors index, shard presence,
    shard size/header, and leftover download markers. This fixture keeps the
    files syntactically valid so that a failure means qasr did not check the
    intended invariant.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    shard_name = "model-00001-of-00001.safetensors"
    weight_map: dict[str, str] = {"w1": shard_name}
    if missing_shard:
        weight_map["w2"] = "model-00002-of-00001.safetensors"

    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_asr",
                "architectures": ["Qwen3ASRForConditionalGeneration"],
                "torch_dtype": "float16",
            }
        ),
        encoding="utf-8",
    )
    (path / "generation_config.json").write_text('{"max_new_tokens": 512}', encoding="utf-8")
    (path / "tokenizer_config.json").write_text('{"tokenizer_class": "Qwen2Tokenizer"}', encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "preprocessor_config.json").write_text(
        '{"feature_extractor_type": "Qwen3ASRFeatureExtractor"}', encoding="utf-8"
    )
    (path / "chat_template.json").write_text("{}", encoding="utf-8")
    (path / "vocab.json").write_text('{"<pad>": 0, "<eos>": 1}', encoding="utf-8")
    (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 32}, "weight_map": weight_map}),
        encoding="utf-8",
    )

    if truncate_shard:
        (path / shard_name).write_bytes(struct.pack("<Q", 10_000_000) + b"{}")
    else:
        (path / shard_name).write_bytes(safetensors_blob())

    if aria2:
        (path / (shard_name + ".aria2")).write_bytes(b"")
    return path


# ---------------------------------------------------------------------------
# Subtitle helpers
# ---------------------------------------------------------------------------

_SRT_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_VTT_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})")


def _to_ms(match: tuple[str, str, str, str]) -> int:
    hours, minutes, seconds, millis = match
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(millis)


def parse_srt(text: str) -> list[tuple[int, int, str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    cues: list[tuple[int, int, str]] = []
    for block in blocks:
        lines = block.splitlines()
        for index, line in enumerate(lines):
            if "-->" not in line:
                continue
            stamps = _SRT_TS.findall(line)
            assert len(stamps) == 2, f"bad SRT timestamp line: {line!r}"
            cues.append((_to_ms(stamps[0]), _to_ms(stamps[1]), "\n".join(lines[index + 1 :])))
            break
        else:
            pytest.fail(f"SRT block has no timestamp line: {block!r}")
    return cues


def parse_vtt(text: str) -> list[tuple[int, int, str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    assert normalized.lstrip().startswith("WEBVTT"), "VTT must start with WEBVTT"
    body = normalized.split("\n\n", 1)[1] if "\n\n" in normalized else ""
    cues: list[tuple[int, int, str]] = []
    for block in [item.strip() for item in body.split("\n\n") if item.strip()]:
        lines = block.splitlines()
        for index, line in enumerate(lines):
            if "-->" not in line:
                continue
            stamps = _VTT_TS.findall(line)
            assert len(stamps) == 2, f"bad VTT timestamp line: {line!r}"
            cues.append((_to_ms(stamps[0]), _to_ms(stamps[1]), "\n".join(lines[index + 1 :])))
            break
    return cues


def display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def copy_wav(src: str | Path, dst: str | Path) -> Path:
    from shutil import copyfile

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    copyfile(src, dst)
    return dst
