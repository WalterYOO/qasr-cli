"""Fake ``torch`` used only by the subprocess black-box review harness.

The CLI gates the vLLM backend on Linux + CUDA. Review tests simulate that
host shape with ``QASR_FAKE_LINUX=1`` plus this module; no implementation
switch is involved and results never count as gpu06 evidence.
"""

from __future__ import annotations

import os

__version__ = "0.0-fake"


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return os.environ.get("QASR_FAKE_LINUX") == "1"

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_device_name(index: int = 0) -> str:
        return "fake-cuda"


class _Mps:
    @staticmethod
    def is_available() -> bool:
        return False


class _Backends:
    mps = _Mps()


cuda = _Cuda()
backends = _Backends()
float32 = "torch.float32"
float16 = "torch.float16"
bfloat16 = "torch.bfloat16"
