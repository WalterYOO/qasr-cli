"""Pure helpers for the independent P2 black-box review suite.

No test may import ``qasr_cli``. Everything here is stdlib-only (plus the
test environment's ``jsonschema`` for schema validation) and is used to
create fixtures, parse process output, or scan for leaked secrets.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path
from typing import Any, Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_json(proc) -> dict[str, Any]:
    raw = proc.stdout.strip()
    if not raw:
        pytest.fail(
            f"stdout is empty (rc={proc.returncode})\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"stdout is not one JSON value: {exc}\n"
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
            f"stdout has no JSONL lines (rc={proc.returncode})\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    values: list[dict[str, Any]] = []
    for index, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"stdout line {index} is not JSON: {exc}\n{line[:2000]}"
            )
        if not isinstance(value, dict):
            pytest.fail(f"stdout line {index} is not an object")
        values.append(value)
    return values


def assert_exit(proc, expected: int | Iterable[int], *, context: str = "") -> None:
    allowed = {expected} if isinstance(expected, int) else set(expected)
    if proc.returncode not in allowed:
        prefix = f"{context}: " if context else ""
        pytest.fail(
            f"{prefix}unexpected exit {proc.returncode}, expected {sorted(allowed)}\n"
            f"stdout:\n{proc.stdout[:4000]}\n"
            f"stderr:\n{proc.stderr[:4000]}"
        )


def assert_error(payload: dict[str, Any], *, code: int | None = None,
                 kind: str | None = None) -> dict[str, Any]:
    assert payload.get("ok") is False, f"expected ok=false: {payload!r}"
    error = payload.get("error")
    assert isinstance(error, dict), f"missing error object: {payload!r}"
    if code is not None:
        assert error.get("code") == code, f"error.code={error.get('code')!r}, want {code}"
    if kind is not None:
        assert error.get("kind") == kind, f"error.kind={error.get('kind')!r}, want {kind}"
    return error


def assert_no_traceback(proc) -> None:
    text = f"{proc.stdout}\n{proc.stderr}"
    assert "Traceback (most recent call last)" not in text, text[-4000:]
    assert "ImportError" not in text, text[-4000:]


def find_key(payload: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(payload, dict):
        for name, value in payload.items():
            if name == key:
                found.append(value)
            found.extend(find_key(value, key))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(find_key(value, key))
    return found


def schema_path(name: str) -> Path:
    return REPO_ROOT / "schemas" / name


def load_schema(name: str) -> dict[str, Any]:
    path = schema_path(name)
    if not path.is_file():
        pytest.fail(f"schema is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        pytest.fail(f"schema is not JSON: {path}: {exc}")
    assert isinstance(value, dict), path
    return value


def validate_schema(payload: dict[str, Any], name: str) -> None:
    import jsonschema

    schema = load_schema(name)
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        pytest.fail(
            f"{name} validation failed at {list(exc.absolute_path)}: {exc.message}\n"
            f"payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)[:4000]}"
        )


def read_jsonl_file(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    for index, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{target}:{index} is not JSON: {exc}")
        if isinstance(value, dict):
            records.append(value)
    return records


# ---------------------------------------------------------------------------
# audio fixtures (stdlib only)
# ---------------------------------------------------------------------------

def write_wav(
    path: str | Path,
    *,
    seconds: float,
    sample_rate: int = 16_000,
    frequency: float | None = 440.0,
    amplitude: int = 8000,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = int(round(seconds * sample_rate))
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        chunk_frames = sample_rate
        written = 0
        while written < total:
            count = min(chunk_frames, total - written)
            frames = bytearray()
            for index in range(count):
                sample = 0
                if frequency is not None:
                    sample = int(
                        amplitude
                        * math.sin(
                            2.0
                            * math.pi
                            * frequency
                            * (written + index)
                            / sample_rate
                        )
                    )
                frames.extend(struct.pack("<h", sample))
            wav.writeframes(bytes(frames))
            written += count
    return target


def write_stereo_wav(
    path: str | Path,
    *,
    seconds: float,
    sample_rate: int = 16_000,
    left: int = 1000,
    right: int = 3000,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = int(round(seconds * sample_rate))
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for _ in range(total):
            frames.extend(struct.pack("<hh", left, right))
        wav.writeframes(bytes(frames))
    return target


def pcm_s16le(
    frames: int,
    *,
    channels: int = 1,
    sample: int = 1000,
    left: int | None = None,
    right: int | None = None,
) -> bytes:
    out = bytearray()
    for _ in range(frames):
        if channels == 1:
            out.extend(struct.pack("<h", sample))
        else:
            out.extend(struct.pack("<hh", left if left is not None else sample,
                                   right if right is not None else sample))
    return bytes(out)


def make_model_dir(path: str | Path) -> Path:
    """Minimal synthetic model directory that model resolution accepts."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    shard = "model-00001-of-00001.safetensors"
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
    (path / "generation_config.json").write_text(
        '{"max_new_tokens": 512}', encoding="utf-8"
    )
    (path / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "Qwen2Tokenizer"}', encoding="utf-8"
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "preprocessor_config.json").write_text(
        '{"feature_extractor_type": "Qwen3ASRFeatureExtractor"}', encoding="utf-8"
    )
    (path / "chat_template.json").write_text("{}", encoding="utf-8")
    (path / "vocab.json").write_text('{"<pad>": 0, "<eos>": 1}', encoding="utf-8")
    (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 32},
                "weight_map": {"w": shard},
            }
        ),
        encoding="utf-8",
    )
    (path / shard).write_bytes(safetensors_blob())
    return path


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
