"""Fixtures for the independent P1 black-box review suite."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from fake_openai import FakeOpenAIServer  # noqa: E402
from helpers import (  # noqa: E402
    make_incomplete_model,
    make_model_dir,
    write_silence,
    write_silent_prefix,
    write_wav,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "p1: P1 independent review suite")
    config.addinivalue_line("markers", "integration: requires RUN_INTEGRATION=1 and local models")
    config.addinivalue_line("markers", "gpu: requires CUDA/vLLM and gpu06 evidence")
    config.addinivalue_line("markers", "slow: long-running scenario")
    config.addinivalue_line("markers", "n3: requires the N3 vLLM/auto backend")
    config.addinivalue_line("markers", "n4: requires the N4 long-audio windowing work")
    config.addinivalue_line("markers", "n5: requires the N5 batch/ops commands")


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.p1)


def _resolve_command() -> list[str]:
    explicit = os.environ.get("QASR_BIN")
    if explicit:
        return [explicit]
    if (REPO_ROOT / "qasr_cli").is_dir() and os.environ.get("QASR_USE_INSTALLED") != "1":
        python = os.environ.get("QASR_PYTHON", sys.executable)
        return [python, "-m", "qasr_cli"]
    found = shutil.which("qasr")
    if found:
        return [found]
    pytest.exit(
        "cannot find qasr: set QASR_BIN, QASR_PYTHON, or run from the checkout",
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
        values.update({
            "QASR_HOME": str(self.home),
            "NO_COLOR": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "TERM": "dumb",
            "COLUMNS": "120",
        })
        if self.uses_module:
            existing = values.get("PYTHONPATH")
            values["PYTHONPATH"] = (
                str(REPO_ROOT) if not existing else os.pathsep.join([str(REPO_ROOT), existing])
            )
        if overrides:
            values.update({str(key): str(value) for key, value in overrides.items()})
        return values

    def run(self, *args, cwd: str | Path | None = None,
            env: dict[str, str] | None = None, timeout: float = 90,
            stdin: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*self.command, *map(str, args)],
            cwd=str(cwd or self.work),
            env=self.env(env),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def popen(self, *args, cwd: str | Path | None = None,
              env: dict[str, str] | None = None) -> subprocess.Popen:
        return subprocess.Popen(
            [*self.command, *map(str, args)],
            cwd=str(cwd or self.work),
            env=self.env(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


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
def fake_openai():
    servers: list[FakeOpenAIServer] = []

    def start(**kwargs) -> FakeOpenAIServer:
        server = FakeOpenAIServer(**kwargs)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()


@pytest.fixture(scope="session")
def short_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_wav(tmp_path_factory.mktemp("audio") / "short.wav", seconds=1.0)


@pytest.fixture(scope="session")
def medium_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_wav(tmp_path_factory.mktemp("audio") / "medium.wav", seconds=25.0)


@pytest.fixture(scope="session")
def long_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_wav(tmp_path_factory.mktemp("audio") / "long.wav", seconds=181.0)


@pytest.fixture(scope="session")
def exactly_180_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_wav(tmp_path_factory.mktemp("audio") / "exactly180.wav", seconds=180.0)


@pytest.fixture(scope="session")
def silent_prefix_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_silent_prefix(tmp_path_factory.mktemp("audio") / "silent_prefix.wav")


@pytest.fixture
def empty_wav(tmp_path: Path) -> Path:
    return write_silence(tmp_path / "empty.wav", seconds=0.0)


@pytest.fixture
def synthetic_models(qasr: QasrRunner) -> Path:
    root = qasr.home / "models"
    root.mkdir(parents=True, exist_ok=True)
    make_incomplete_model(root, "Qwen3-ASR-0.6B")
    return root


@pytest.fixture
def models_root(qasr: QasrRunner) -> Path:
    root = qasr.home / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def complete_model(models_root: Path):
    def create(folder: str = "Qwen3-ASR-0.6B") -> Path:
        return make_model_dir(models_root / folder)

    return create


@pytest.fixture
def incomplete_model(models_root: Path):
    def create(folder: str = "Qwen3-ASR-0.6B") -> Path:
        return make_incomplete_model(models_root, folder)

    return create


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    path = tmp_path / "reference.txt"
    path.write_text("今天天气不错。我们去公园散步。", encoding="utf-8")
    return path


@pytest.fixture
def empty_wavs(tmp_path: Path) -> list[Path]:
    first = write_silence(tmp_path / "empty-one.wav", seconds=0.0)
    second = write_silence(tmp_path / "empty-two.wav", seconds=0.0)
    return [first, second]


@pytest.fixture
def import_blocker(tmp_path: Path):
    """Create modules that record/raise if qasr imports heavy runtimes."""
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    marker = tmp_path / "imported.txt"
    for name in ("torch", "qwen_asr", "vllm"):
        (blocker / f"{name}.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
            f"raise ImportError('blocked {name}')\n",
            encoding="utf-8",
        )
    return blocker, marker


@pytest.fixture
def fake_serve_bin(tmp_path: Path):
    """Create a fake qwen-asr-serve that records argv and exits 0."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "argv.txt"
    script = bin_dir / "qwen-asr-serve"
    script.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$@" > "$QASR_TEST_ARGV_FILE"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return bin_dir, argv_file


@pytest.fixture
def empty_path(tmp_path: Path) -> Path:
    directory = tmp_path / "empty-bin"
    directory.mkdir()
    return directory


@pytest.fixture(scope="session")
def integration_env() -> dict[str, str]:
    model_dir = os.environ.get("QASR_TEST_MODEL_DIR")
    audio = os.environ.get("QASR_TEST_AUDIO")
    if os.environ.get("RUN_INTEGRATION") != "1" or not model_dir or not audio:
        pytest.skip(
            "set RUN_INTEGRATION=1, QASR_TEST_MODEL_DIR and QASR_TEST_AUDIO"
        )
    return {"QASR_TEST_MODEL_DIR": model_dir, "QASR_TEST_AUDIO": audio}

