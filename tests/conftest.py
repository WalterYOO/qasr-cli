"""Shared fixtures: isolated QASR_HOME, synthetic WAVs and models."""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    home = tmp_path / "qasr-home"
    home.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("QASR_HOME", str(home))
    for name in (
        "QASR_CONFIG", "QASR_MODEL", "QASR_MODEL_DIR", "QASR_BACKEND",
        "QASR_DEVICE", "QASR_DTYPE", "QASR_OFFLINE", "QASR_LANGUAGE",
        "QASR_SERVE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


def write_wav(path: Path, seconds: float = 1.0, sr: int = 16000, freq: float = 440.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = max(0, int(seconds * sr))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        frames = bytearray()
        for i in range(total):
            value = int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sr))
            frames += struct.pack("<h", value)
        handle.writeframes(bytes(frames))
    return path


@pytest.fixture
def wav_file(tmp_path):
    return write_wav(tmp_path / "clip.wav", seconds=1.0)


def make_safetensors(path: Path, n_floats: int = 64, truncate_to: int | None = None) -> Path:
    data_len = n_floats * 4
    header = {
        "w": {"dtype": "F32", "shape": [n_floats], "data_offsets": [0, data_len]},
    }
    raw = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(b"\0" * data_len)
    if truncate_to is not None:
        with path.open("r+b") as handle:
            handle.truncate(truncate_to)
    return path


def make_model_dir(
    root: Path,
    name: str = "Qwen3-ASR-0.6B",
    *,
    sharded: bool = False,
    truncate: bool = False,
    partial_marker: bool = False,
) -> Path:
    model = root / "models" / name
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model / "vocab.json").write_text("{}", encoding="utf-8")
    if sharded:
        shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        index = {"metadata": {"format": "pt"},
                 "weight_map": {"a.0": shards[0], "a.1": shards[1]}}
        (model / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        make_safetensors(model / shards[0], n_floats=32)
        make_safetensors(
            model / shards[1],
            n_floats=32,
            truncate_to=20 if truncate else None,
        )
    else:
        make_safetensors(
            model / "model.safetensors",
            n_floats=64,
            truncate_to=24 if truncate else None,
        )
    if partial_marker:
        (model / "model.safetensors.aria2").write_text("stub", encoding="utf-8")
    return model


@pytest.fixture
def model_factory(tmp_path, isolated_home):
    def _make(name: str = "Qwen3-ASR-0.6B", **kwargs):
        return make_model_dir(isolated_home, name, **kwargs)

    return _make
