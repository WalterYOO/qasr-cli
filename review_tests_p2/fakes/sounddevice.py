"""Fake sounddevice used only by subprocess black-box mic tests."""

from __future__ import annotations

import json
import os
import struct
import threading
import time

import numpy as np

__version__ = "0.0-fake"


class PortAudioError(RuntimeError):
    pass


def _log(event: str, **fields) -> None:
    path = os.environ.get("QASR_FAKE_MIC_LOG")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, **fields}) + "\n")


def _load_pcm() -> np.ndarray:
    path = os.environ.get("QASR_FAKE_MIC_PCM")
    if not path:
        return np.zeros(16_000, dtype=np.float32)
    raw = open(path, "rb").read()
    usable = len(raw) - (len(raw) % 2)
    samples = struct.unpack("<" + "h" * (usable // 2), raw[:usable])
    return np.asarray(samples, dtype=np.float32) / 32768.0


class InputStream:
    def __init__(
        self,
        samplerate=16_000,
        channels=1,
        blocksize=0,
        dtype="float32",
        callback=None,
        **kwargs,
    ) -> None:
        _log(
            "open",
            samplerate=samplerate,
            channels=channels,
            blocksize=blocksize,
            dtype=dtype,
        )
        if os.environ.get("QASR_FAKE_MIC_FAIL") == "open":
            raise PortAudioError("fake microphone open failure")
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.blocksize = int(blocksize or round(self.samplerate * 0.1))
        self.dtype = dtype
        self.callback = callback
        self._data = _load_pcm()
        self._position = 0
        self._thread = None
        self._stopped = False

    def start(self) -> None:
        if self.callback is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stopped and self._position < self._data.size:
            block = self._data[self._position:self._position + self.blocksize]
            self._position += block.size
            if block.size == 0:
                break
            if self.channels > 1:
                frames = np.repeat(block.reshape(-1, 1), self.channels, axis=1)
            else:
                frames = block.reshape(-1, 1)
            self.callback(frames, block.size, None, None)
            time.sleep(0.001)

    def read(self, frames: int):
        block = self._data[self._position:self._position + frames]
        self._position += block.size
        if block.size == 0:
            # A real microphone blocks; emulate that instead of busy-spinning
            # so tests can send SIGINT or let the parent observe snapshots.
            time.sleep(0.05)
        if self.channels > 1:
            block = np.repeat(block.reshape(-1, 1), self.channels, axis=1)
        else:
            block = block.reshape(-1, 1)
        return block, False

    def stop(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2)

    def close(self) -> None:
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


RawInputStream = InputStream


def query_devices(*args, **kwargs):
    return [{"name": "fake-mic", "max_input_channels": 1,
             "default_samplerate": 16_000}]


def check_input_settings(*args, **kwargs):
    return None


class _Default:
    device = (0, 0)


default = _Default()
