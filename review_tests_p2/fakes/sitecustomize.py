"""Simulate a Linux host for the fake-vLLM review harness.

Only active when ``QASR_FAKE_LINUX=1``. This lives in the review suite's
temporary PYTHONPATH; it does not touch the implementation.
"""

from __future__ import annotations

import os
import platform
import sys

if os.environ.get("QASR_FAKE_LINUX") == "1":
    # soundfile picks its shared-library name from sys.platform at import
    # time. Load it against the *real* host first, then pretend to be Linux
    # for the CLI's vLLM platform gate.
    try:
        import soundfile  # noqa: F401
    except Exception:
        pass
    sys.platform = "linux"
    platform.system = lambda: "Linux"
