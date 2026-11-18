"""Fake vllm module used only by subprocess black-box tests."""

from __future__ import annotations

import atexit
import os

__version__ = "0.0-fake"

if os.environ.get("QASR_FAKE_STREAM_BANNER") == "1":
    print("FAKE-VLLM-IMPORT-STDOUT", flush=True)
    atexit.register(lambda: print("FAKE-VLLM-IMPORT-ATEXIT", flush=True))
