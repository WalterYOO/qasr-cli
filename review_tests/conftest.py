"""Fixtures for the black-box qasr review suite.

Every test invokes ``qasr`` through ``subprocess``. The runner isolates
``QASR_HOME`` and strips inherited ``QASR_*`` variables so a developer shell
cannot silently change the result.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from helpers import (  # noqa: E402
    copy_wav,
    empty_wav,
    make_model_dir,
    parse_json,
    parse_jsonl,
    parse_srt,
    parse_vtt,
    text_as_wav,
    truncated_wav,
    write_jsonl,
    write_manifest,
    zero_byte,
)
from helpers import (  # noqa: E402
    make_wav as _make_wav,
)

RUNTIME_MODULES = ("torch", "qwen_asr", "vllm")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires RUN_INTEGRATION=1 and local model files"
    )
    config.addinivalue_line("markers", "slow: long-running integration scenario")


def _resolve_command() -> list[str]:
    explicit = os.environ.get("QASR_BIN")
    if explicit:
        return [explicit]

    # Default to the checkout under review; an installed console script may be
    # stale or belong to a different version.
    if (REPO_ROOT / "qasr_cli").is_dir() and os.environ.get("QASR_USE_INSTALLED") != "1":
        python = os.environ.get("QASR_PYTHON", sys.executable)
        return [python, "-m", "qasr_cli"]

    found = shutil.which("qasr")
    if found:
        return [found]

    pytest.exit(
        "cannot find qasr: set QASR_BIN, QASR_PYTHON, install qasr-cli, "
        "or run from a checkout containing qasr_cli/",
        returncode=2,
    )


class QasrRunner:
    def __init__(self, command: list[str], home: Path, work: Path) -> None:
        self.command = command
        self.home = home
        self.work = work
        self.uses_module = command[-2:] == ["-m", "qasr_cli"]

    def env(self, overrides: dict[str, str] | None = None) -> dict[str, str]:
        values = os.environ.copy()
        for key in list(values):
            if key.startswith("QASR_"):
                del values[key]
        values.update(
            {
                "QASR_HOME": str(self.home),
                "NO_COLOR": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "TERM": "dumb",
                "COLUMNS": "120",
            }
        )
        if self.uses_module:
            existing = values.get("PYTHONPATH")
            values["PYTHONPATH"] = (
                str(REPO_ROOT) if not existing else os.pathsep.join([str(REPO_ROOT), existing])
            )
        if overrides:
            values.update({str(key): str(value) for key, value in overrides.items()})
        return values

    def run(
        self,
        *args,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 60,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*self.command, *map(str, args)],
            cwd=str(cwd or self.work),
            env=self.env(env),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def popen(
        self,
        *args,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.Popen:
        return subprocess.Popen(
            [*self.command, *map(str, args)],
            cwd=str(cwd or self.work),
            env=self.env(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


@pytest.fixture(scope="session")
def qasr_command() -> list[str]:
    return _resolve_command()


@pytest.fixture
def qasr(tmp_path: Path, qasr_command: list[str]) -> QasrRunner:
    home = tmp_path / "qasr_home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    return QasrRunner(qasr_command, home, work)


@pytest.fixture
def models_root(qasr: QasrRunner) -> Path:
    root = qasr.home / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def make_wav():
    return _make_wav


@pytest.fixture
def make_model():
    return make_model_dir


@pytest.fixture
def import_blocker(tmp_path: Path):
    """Make torch/qwen_asr/vllm imports fail and log every import attempt."""
    block_dir = tmp_path / "import_block"
    block_dir.mkdir()
    log_path = tmp_path / "imports.log"
    bootstrap = "\n".join(
        [
            "import sys",
            f"LOG = {str(log_path)!r}",
            "BLOCKED = ('torch', 'qwen_asr', 'vllm')",
            "",
            "def _hook(event, args):",
            "    if event != 'import':",
            "        return",
            "    name = args[0]",
            "    with open(LOG, 'a', encoding='utf-8') as handle:",
            "        handle.write(name + '\\n')",
            "    if name.split('.')[0] in BLOCKED:",
            "        raise ImportError('blocked by qasr review harness: ' + name)",
            "",
            "sys.addaudithook(_hook)",
            "",
        ]
    )
    (block_dir / "sitecustomize.py").write_text(bootstrap, encoding="utf-8")

    def env(**extra: str) -> dict[str, str]:
        values = {
            "PYTHONPATH": os.pathsep.join([str(block_dir), str(REPO_ROOT)]),
        }
        values.update({key: str(value) for key, value in extra.items()})
        return values

    def attempts() -> list[str]:
        if not log_path.exists():
            return []
        return [
            line.strip()
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return SimpleNamespace(env=env, attempts=attempts, log=log_path)


@pytest.fixture(scope="session")
def integration_env() -> SimpleNamespace:
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 to run model-backed tests")
    root_value = os.environ.get("QASR_TEST_MODEL_DIR")
    if not root_value:
        pytest.skip("set QASR_TEST_MODEL_DIR to the directory containing Qwen3-ASR-0.6B")

    root = Path(root_value).expanduser()
    asr = root / "Qwen3-ASR-0.6B"
    aligner = root / "Qwen3-ForcedAligner-0.6B"
    if not asr.is_dir():
        pytest.skip(f"missing ASR model: {asr}")
    if not aligner.is_dir():
        pytest.skip(f"missing aligner model: {aligner}")

    audio = Path(os.environ.get("QASR_TEST_AUDIO", "/tmp/qasr_asr_zh.wav")).expanduser()
    if not audio.is_file():
        pytest.skip(f"missing test audio: {audio} (set QASR_TEST_AUDIO)")

    device = os.environ.get("QASR_TEST_DEVICE")
    extra = ["--model", str(asr), "--aligner", str(aligner)]
    if device:
        extra.extend(["--device", device])
    return SimpleNamespace(asr=asr, aligner=aligner, audio=audio, root=root, model_args=extra)


__all__ = [
    "QasrRunner",
    "RUNTIME_MODULES",
    "copy_wav",
    "empty_wav",
    "make_model_dir",
    "make_wav",
    "parse_json",
    "parse_jsonl",
    "parse_srt",
    "parse_vtt",
    "text_as_wav",
    "truncated_wav",
    "write_jsonl",
    "write_manifest",
    "zero_byte",
]
