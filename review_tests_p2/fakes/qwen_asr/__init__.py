"""Fake qwen_asr streaming model used only by subprocess black-box tests.

The real qwen-asr API buffers incoming PCM and only decodes when a full
``chunk_size_sec`` worth of samples is available. This fake mirrors that
shape so the CLI's snapshot/finish logic is exercised without torch/vLLM.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

__version__ = "0.0-fake"


def _log(event: str, **fields) -> None:
    path = os.environ.get("QASR_FAKE_STREAM_LOG")
    if not path:
        return
    record = {"event": event, **fields}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _banner() -> None:
    if os.environ.get("QASR_FAKE_STREAM_BANNER") != "1":
        return
    print("FAKE-VLLM-BANNER-STDOUT", flush=True)


def _sleep() -> None:
    value = os.environ.get("QASR_FAKE_STREAM_SLEEP")
    if value:
        time.sleep(float(value))


def _init_sleep() -> None:
    value = os.environ.get("QASR_FAKE_STREAM_INIT_SLEEP")
    if value:
        time.sleep(float(value))


class _State:
    def __init__(
        self,
        *,
        context: str,
        language,
        unfixed_chunk_num: int,
        unfixed_token_num: int,
        chunk_size_sec: float,
    ) -> None:
        self.context = context or ""
        self.language_arg = language
        self.unfixed_chunk_num = unfixed_chunk_num
        self.unfixed_token_num = unfixed_token_num
        self.chunk_size_sec = float(chunk_size_sec)
        self.chunk_samples = max(1, int(round(self.chunk_size_sec * 16_000)))
        self.buffer = np.zeros(0, dtype=np.float32)
        self.audio_accum = np.zeros(0, dtype=np.float32)
        self.chunk_id = 0
        self.text = ""
        self.language = None
        self.finished = False


class _Model:
    def __init__(self, model_path, **kwargs) -> None:
        self.model_path = str(model_path)
        self.kwargs = kwargs
        _init_sleep()
        _log(
            "model_init",
            model_path=self.model_path,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            kwargs={
                key: value
                for key, value in kwargs.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        )
        _banner()
        if os.environ.get("QASR_FAKE_STREAM_FAIL") == "init":
            raise RuntimeError("fake stream init failure")

    def init_streaming_state(
        self,
        context: str = "",
        language=None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> _State:
        _log(
            "init",
            context=context,
            language=language,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            chunk_size_sec=chunk_size_sec,
        )
        return _State(
            context=context,
            language=language,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            chunk_size_sec=chunk_size_sec,
        )

    def streaming_transcribe(self, pcm16k, state: _State) -> _State:
        audio = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
        _log(
            "feed",
            n=int(audio.size),
            total_before=int(state.audio_accum.size),
            first=[round(float(value), 6) for value in audio[:4]],
        )
        state.buffer = (
            np.concatenate([state.buffer, audio])
            if state.buffer.size
            else audio.copy()
        )
        state.audio_accum = (
            np.concatenate([state.audio_accum, audio])
            if state.audio_accum.size
            else audio.copy()
        )
        while state.buffer.size >= state.chunk_samples:
            state.buffer = state.buffer[state.chunk_samples:]
            state.chunk_id += 1
            self._fail(f"chunk:{state.chunk_id}")
            _sleep()
            state.text = " ".join(f"c{i}" for i in range(1, state.chunk_id + 1))
            state.language = self._language(state.language_arg)
            _log(
                "chunk",
                chunk_id=state.chunk_id,
                text=state.text,
                language=state.language,
                audio_accum=int(state.audio_accum.size),
                buffer=int(state.buffer.size),
            )
        return state

    def finish_streaming_transcribe(self, state: _State) -> _State:
        self._fail("finish")
        tail = int(state.buffer.size)
        if tail:
            state.audio_accum = (
                np.concatenate([state.audio_accum, state.buffer])
                if state.audio_accum.size
                else state.buffer.copy()
            )
            state.buffer = np.zeros(0, dtype=np.float32)
            state.chunk_id += 1
            _sleep()
            state.text = " ".join(f"c{i}" for i in range(1, state.chunk_id + 1))
            state.language = self._language(state.language_arg)
        state.finished = True
        _log(
            "finish",
            tail=tail,
            chunk_id=state.chunk_id,
            text=state.text,
            language=state.language,
            audio_accum=int(state.audio_accum.size),
        )
        return state

    def _fail(self, where: str) -> None:
        if os.environ.get("QASR_FAKE_STREAM_FAIL") == where:
            raise RuntimeError(f"fake stream failure at {where}")

    @staticmethod
    def _language(value):
        override = os.environ.get("QASR_FAKE_STREAM_LANGUAGE")
        if override == "none":
            return None
        if override:
            return override
        if value in (None, "auto", "zh", "Chinese"):
            return "Chinese"
        return str(value)


class Qwen3ASRModel:
    @staticmethod
    def LLM(model_path, **kwargs):
        return _Model(model_path, **kwargs)

    @staticmethod
    def from_pretrained(model_path, **kwargs):
        return _Model(model_path, **kwargs)


class Qwen3ForcedAligner:
    @staticmethod
    def from_pretrained(model_path, **kwargs):
        return _Model(model_path, **kwargs)
