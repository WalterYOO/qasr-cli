"""N3: VLLMBackend against fake torch/qwen_asr modules."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from qasr_cli.backends import (
    AlignJob,
    AlignOptions,
    Segment,
    TranscribeJob,
    TranscribeOptions,
    VLLMBackend,
    build_vllm_backend,
    clear_backend_cache,
)
from qasr_cli.cli import main
from qasr_cli.commands.common import default_batch_size, resolve_backend
from qasr_cli.errors import RuntimeFailure
from qasr_cli.inputs import InputItem
from tests.conftest import write_wav


class _Raw:
    def __init__(self, text, language="Chinese", stamps=None):
        self.text = text
        self.language = language
        self.time_stamps = stamps


class _StampItem:
    def __init__(self, text, start, end):
        self.text = text
        self.start_time = start
        self.end_time = end


class _StampResult:
    def __init__(self, items):
        self.items = items


class _Engine:
    instances = []

    def __init__(self, **kwargs):
        self.transcribe_kwargs = kwargs
        # The real Qwen3ASRModel.LLM object exposes sampling_params, which is
        # where vLLM reads max_tokens from.
        self.sampling_params = types.SimpleNamespace(
            max_tokens=kwargs.get("max_new_tokens")
        )
        type(self).instances.append(self)

    def transcribe(self, *, audio, context, language, return_time_stamps=False):
        self.last_call = {
            "audio": list(audio),
            "context": list(context),
            "language": list(language),
            "return_time_stamps": return_time_stamps,
        }
        results = []
        for index, _path in enumerate(audio):
            stamps = None
            if return_time_stamps:
                stamps = _StampResult([_StampItem(f"w{index}", 0.0, 0.5)])
            results.append(_Raw(f"text-{index}", "Chinese", stamps))
        return results


class _FakeQwen:
    last_kwargs = None
    last_model = None

    @classmethod
    def LLM(cls, model, **kwargs):
        cls.last_model = model
        cls.last_kwargs = kwargs
        return _Engine(**kwargs)


def _install_fakes(monkeypatch, *, cuda=True):
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda, empty_cache=lambda: None
    )
    vllm = types.ModuleType("vllm")
    qwen = types.ModuleType("qwen_asr")
    qwen.Qwen3ASRModel = _FakeQwen
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "qwen_asr", qwen)
    monkeypatch.setattr(sys, "platform", "linux")
    _FakeQwen.last_kwargs = None
    _FakeQwen.last_model = None
    _Engine.instances = []
    clear_backend_cache()
    return torch


def _job(wav: Path, language="Chinese", context="") -> TranscribeJob:
    item = InputItem(id=wav.stem, source=str(wav), path=wav)
    return TranscribeJob(item=item, info=None, language=language, context=context)


def test_vllm_constructor_kwargs_and_batch(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    wav_a = write_wav(tmp_path / "a.wav", seconds=0.2)
    wav_b = write_wav(tmp_path / "b.wav", seconds=0.2)
    backend = build_vllm_backend(
        tmp_path,
        device="cuda:1",
        dtype="fp16",
        max_inference_batch_size=8,
        max_new_tokens=256,
        gpu_memory_utilization=0.7,
    )
    results = backend.transcribe(
        [_job(wav_a, context="terms"), _job(wav_b)],
        TranscribeOptions(),
    )
    kwargs = _FakeQwen.last_kwargs
    assert _FakeQwen.last_model == str(tmp_path.resolve())
    assert "device" not in kwargs
    assert kwargs["dtype"] == "float16"
    assert kwargs["max_inference_batch_size"] == 8
    assert kwargs["max_new_tokens"] == 256
    assert kwargs["gpu_memory_utilization"] == 0.7
    assert kwargs["forced_aligner"] is None
    engine = _Engine.instances[0]
    assert engine.last_call["context"] == ["terms", ""]
    assert engine.last_call["language"] == ["Chinese", "Chinese"]
    assert engine.last_call["return_time_stamps"] is False
    assert [result.text for result in results] == ["text-0", "text-1"]
    assert results[0].language_source == "forced"


def test_vllm_forwards_per_call_token_budget(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    wav = write_wav(tmp_path / "a.wav", seconds=0.2)
    backend = build_vllm_backend(tmp_path)
    backend.transcribe([_job(wav)], TranscribeOptions(max_new_tokens=9376))
    assert _Engine.instances[0].sampling_params.max_tokens == 9376

    backend.transcribe([_job(wav)], TranscribeOptions())
    assert _Engine.instances[0].sampling_params.max_tokens == 512


def test_vllm_timestamp_path_uses_aligner(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    wav = write_wav(tmp_path / "a.wav", seconds=0.2)
    aligner = tmp_path / "aligner"
    aligner.mkdir()
    backend = build_vllm_backend(tmp_path, aligner_path=aligner)
    results = backend.transcribe(
        [_job(wav)], TranscribeOptions(timestamps=True, granularity="word")
    )
    kwargs = _FakeQwen.last_kwargs
    assert kwargs["forced_aligner"] == str(aligner.resolve())
    assert kwargs["forced_aligner_kwargs"]["local_files_only"] is False
    assert _Engine.instances[0].last_call["return_time_stamps"] is True
    assert results[0].timestamp_source == "forced_aligner"
    assert results[0].segments == [Segment("w0", 0.0, 0.5, "word")]


def test_vllm_requires_cuda(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, cuda=False)
    backend = VLLMBackend(tmp_path)
    with pytest.raises(RuntimeFailure, match="CUDA"):
        backend.ensure_loaded()


def test_vllm_requires_linux(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    backend = VLLMBackend(tmp_path)
    with pytest.raises(RuntimeFailure, match="Linux"):
        backend.ensure_loaded()


def test_vllm_align_loads_aligner(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    calls = {}

    class _Aligner:
        class model:
            @staticmethod
            def to(device):
                calls["device"] = device

        @classmethod
        def align(cls, *, audio, text, language):
            calls["align"] = (audio, text, language)
            return [_StampResult([_StampItem("hello", 0.0, 0.4)])]

    class _AlignerFactory:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls["source"] = source
            return _Aligner

    sys.modules["qwen_asr"].Qwen3ForcedAligner = _AlignerFactory
    aligner = tmp_path / "aligner"
    aligner.mkdir()
    backend = VLLMBackend(tmp_path, aligner_path=aligner)
    jobs = [AlignJob(item=_job(write_wav(tmp_path / "a.wav")).item, info=None,
                     text="hello", language="Chinese")]
    results = backend.align(jobs, AlignOptions())
    assert calls["source"] == str(aligner)
    assert calls["align"][1] == ["hello"]
    assert results[0].items == [Segment("hello", 0.0, 0.4, "word")]


def test_vllm_unavailable_falls_back_in_auto(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", None)  # import raises
    monkeypatch.setattr(sys, "platform", "linux")

    class _Config:
        def get(self, key):
            return {"server": None, "backend": "auto"}.get(key)

    assert resolve_backend(_Config()) == ("transformers", True)


def test_vllm_available_in_auto(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setattr(sys, "platform", "linux")

    class _Config:
        def get(self, key):
            return {"server": None, "backend": "auto"}.get(key)

    assert resolve_backend(_Config()) == ("vllm", False)


def test_default_batch_size():
    assert default_batch_size("vllm") == 8
    assert default_batch_size("transformers") == 1
    assert default_batch_size("remote") == 1


def test_cli_explicit_vllm_fails_fast_on_mac(capsys, tmp_path):
    wav = write_wav(tmp_path / "a.wav")
    code = main(["--backend", "vllm", "transcribe", str(wav), "--json"])
    out = capsys.readouterr()
    assert code == 3
    assert "Linux" in out.out


def test_cli_auto_stays_transformers_on_mac(capsys, tmp_path):
    wav = write_wav(tmp_path / "a.wav", seconds=0.2)
    code = main(["--backend", "auto", "transcribe", str(wav), "--dry-run", "--json"])
    out = capsys.readouterr()
    assert code == 0, out.err
    plan = __import__("json").loads(out.out)["plan"]
    assert plan["backend"] == "transformers"
    assert plan["backend_fallback"] is False


def test_gpu_memory_utilization_conflicts_with_transformers(capsys, tmp_path):
    wav = write_wav(tmp_path / "a.wav")
    code = main(["--backend", "transformers", "--gpu-memory-utilization", "0.8",
                 "transcribe", str(wav), "--json"])
    out = capsys.readouterr()
    assert code == 2
    assert "--gpu-memory-utilization" in out.out


def test_gpu_memory_utilization_conflicts_with_remote(capsys, tmp_path):
    wav = write_wav(tmp_path / "a.wav")
    code = main(["--server", "http://127.0.0.1:9", "--gpu-memory-utilization", "0.8",
                 "transcribe", str(wav), "--json"])
    out = capsys.readouterr()
    assert code == 2
    assert "--gpu-memory-utilization" in out.out


def test_doctor_reports_auto_backend_resolution(capsys):
    import json

    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    check = next(item for item in payload["checks"] if item["name"] == "backend")
    assert check["value"]["selected"] == "auto"
    assert check["value"]["resolved"] == "transformers"


def test_cuda_index_sets_visible_device_before_import(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    backend = VLLMBackend(tmp_path, device="cuda:7")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert backend.requested_device == "cuda:0"
    backend.ensure_loaded()
    assert "device" not in _FakeQwen.last_kwargs


def test_auto_and_plain_cuda_do_not_touch_visible_devices(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    VLLMBackend(tmp_path, device="auto")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    VLLMBackend(tmp_path, device="cuda")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_explicit_cuda_zero_selects_device_and_rewrites(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    backend = VLLMBackend(tmp_path, device="cuda:0")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert backend.requested_device == "cuda:0"


def test_set_cuda_visible_device_is_idempotent(monkeypatch):
    from qasr_cli.runtime import set_cuda_visible_device

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert set_cuda_visible_device("cuda:0") == "0"
    assert set_cuda_visible_device("cuda:0") == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_auto_sets_cuda_visible_devices_before_probing(monkeypatch, tmp_path):
    seen = {}

    def _probe():
        seen["env"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return False, False

    monkeypatch.setattr("qasr_cli.runtime.vllm_probe", _probe)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wav = write_wav(tmp_path / "a.wav")
    main(["--backend", "auto", "--device", "cuda:1", "--dry-run", "transcribe", str(wav), "--json"])
    assert seen["env"] == "1"


def test_vllm_logging_env_and_handler_redirect(monkeypatch):
    import logging

    from qasr_cli import runtime

    monkeypatch.delenv("VLLM_CONFIGURE_LOGGING", raising=False)
    runtime.configure_vllm_logging()
    assert os.environ["VLLM_CONFIGURE_LOGGING"] == "0"

    logger = logging.getLogger("vllm.engine")
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    try:
        runtime.redirect_vllm_logging()
        assert handler.stream is sys.stderr
    finally:
        logger.removeHandler(handler)


def test_vllm_exit_noise_goes_to_stderr(monkeypatch, capsys):
    import json

    from qasr_cli import runtime

    monkeypatch.setattr(runtime, "_STDOUT_EXIT_GUARD", False)
    registered = []
    monkeypatch.setattr(runtime.atexit, "register", registered.append)
    monkeypatch.setattr(runtime.os, "dup2", lambda *args: None)
    runtime.guard_stdout_at_exit()
    runtime.guard_stdout_at_exit()  # idempotent
    assert len(registered) == 1

    print(json.dumps({"ok": True}))
    registered[0]()  # runs before earlier (vLLM) atexit handlers
    print("EngineCore shutdown")
    captured = capsys.readouterr()
    assert captured.out.strip() == json.dumps({"ok": True})
    assert "EngineCore shutdown" in captured.err
