"""Pure helpers for the independent P1 black-box review suite.

No test may import ``qasr_cli``. Everything here is stdlib-only and is used
to create fixtures, parse process output, or scan for leaked secrets.
"""

from __future__ import annotations

import json
import math
import os
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
            pytest.fail(f"stdout line {index} is not JSON: {exc}\n{line[:2000]}")
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


def first_key(payload: Any, key: str) -> Any:
    values = find_key(payload, key)
    return values[0] if values else None


def find_check(payload: dict[str, Any], needle: str) -> dict[str, Any] | None:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name", "")).lower()
        if needle.lower() in name:
            return check
    return None


# ---------------------------------------------------------------------------
# audio fixtures (stdlib only)
# ---------------------------------------------------------------------------

def write_wav(path: str | Path, *, seconds: float, sample_rate: int = 16_000,
              frequency: float | None = 440.0, amplitude: int = 8000) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = int(round(seconds * sample_rate))
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        # Write in one-second chunks to keep memory bounded for 181 s fixtures.
        chunk_frames = sample_rate
        written = 0
        while written < total:
            count = min(chunk_frames, total - written)
            frames = bytearray()
            for index in range(count):
                sample = 0
                if frequency is not None:
                    sample = int(amplitude * math.sin(
                        2.0 * math.pi * frequency * (written + index) / sample_rate
                    ))
                frames.extend(struct.pack("<h", sample))
            wav.writeframes(bytes(frames))
            written += count
    return target


def write_silence(path: str | Path, *, seconds: float,
                  sample_rate: int = 16_000) -> Path:
    return write_wav(path, seconds=seconds, sample_rate=sample_rate, frequency=None)


def write_silent_prefix(path: str | Path, *, prefix_s: float = 30.0,
                        speech_s: float = 10.0, sample_rate: int = 16_000) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = int(round((prefix_s + speech_s) * sample_rate))
    prefix = int(round(prefix_s * sample_rate))
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * prefix)
        for index in range(total - prefix):
            sample = int(8000 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
            wav.writeframes(struct.pack("<h", sample))
    return target


def make_incomplete_model(root: str | Path, folder: str = "Qwen3-ASR-0.6B") -> Path:
    """Synthetic model dir that verify must call ``incomplete``.

    The fixture is intentionally fake: it has the metadata files and a
    truncated ``model.safetensors``. Tests must not depend on a real 1.7B
    download being incomplete.
    """
    dest = Path(root) / folder
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(
        json.dumps({"model_type": "qwen3_asr", "architectures": ["Qwen3ASR"]}),
        encoding="utf-8",
    )
    (dest / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (dest / "vocab.json").write_text("{}", encoding="utf-8")
    (dest / "model.safetensors").write_bytes(b"\x00" * 64)
    (dest / "model.safetensors.aria2").write_bytes(b"partial")
    return dest


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
    """Minimal synthetic model directory that P0 ``verify`` treats as complete."""
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


def read_jsonl_file(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    records: list[dict[str, Any]] = []
    if not target.is_file():
        return records
    for index, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{target}:{index} is not JSON: {exc}")
        if not isinstance(value, dict):
            pytest.fail(f"{target}:{index} is not an object")
        records.append(value)
    return records


def read_state(out_dir: str | Path) -> list[dict[str, Any]]:
    return read_jsonl_file(Path(out_dir) / ".qasr" / "state.jsonl")


def read_job_index(qasr_home: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    jobs_dir = Path(qasr_home) / "jobs"
    if not jobs_dir.is_dir():
        return records
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            pytest.fail(f"job index {path} is not JSON: {exc}")
        if isinstance(value, dict):
            records.append(value)
    return records


# ---------------------------------------------------------------------------
# schema helpers
# ---------------------------------------------------------------------------

def schema_path(name: str) -> Path:
    return REPO_ROOT / "schemas" / name


def load_schema(name: str) -> dict[str, Any]:
    path = schema_path(name)
    if not path.is_file():
        pytest.fail(f"missing schema: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_schema(payload: dict[str, Any], name: str) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    try:
        jsonschema.validate(payload, load_schema(name))
    except jsonschema.ValidationError as exc:
        pytest.fail(f"payload does not match {name}: {exc.message}\npayload:\n{payload!r}")


# ---------------------------------------------------------------------------
# secret scanning
# ---------------------------------------------------------------------------

def scan_text_for_secret(text: str, secret: str) -> bool:
    return bool(secret) and secret in text


def scan_tree_for_secret(root: str | Path, secret: str) -> list[str]:
    """Return files under *root* whose UTF-8 text contains *secret*."""
    hits: list[str] = []
    if not secret:
        return hits
    base = Path(root)
    if not base.exists():
        return hits
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if secret in text:
            hits.append(str(path))
    return hits


def env_without_qasr() -> dict[str, str]:
    values = dict(os.environ)
    for key in list(values):
        if key.startswith("QASR_"):
            del values[key]
    return values

