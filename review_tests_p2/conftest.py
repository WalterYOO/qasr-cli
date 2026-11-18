"""Fixtures for the independent P2 black-box review suite."""

from __future__ import annotations

import os
import pty
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
FAKES = HERE / "fakes"
sys.path.insert(0, str(HERE))

from helpers import (  # noqa: E402
    make_model_dir,
    pcm_s16le,
    read_jsonl_file,
    write_stereo_wav,
    write_wav,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "p2: P2 independent review suite")
    config.addinivalue_line("markers", "stream: qasr stream")
    config.addinivalue_line("markers", "completion: qasr completion")
    config.addinivalue_line("markers", "finetune: qasr finetune lint-jsonl")
    config.addinivalue_line("markers", "integration: requires RUN_INTEGRATION=1 and local models")
    config.addinivalue_line("markers", "gpu: requires CUDA/vLLM and gpu06 evidence")
    config.addinivalue_line(
        "markers",
        "simulated: runs on a simulated Linux host with fake third-party modules; "
        "never counts as gpu06 evidence",
    )
    config.addinivalue_line("markers", "slow: slow wall-clock scenario")
    config.addinivalue_line("markers", "ambiguous: maps to an open P2-Q contract question")


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.p2)
        if item.get_closest_marker("integration") and os.environ.get("RUN_INTEGRATION") != "1":
            item.add_marker(pytest.mark.skip(reason="set RUN_INTEGRATION=1"))
        if item.get_closest_marker("gpu") and os.environ.get("RUN_GPU") != "1":
            item.add_marker(pytest.mark.skip(reason="set RUN_GPU=1 on a vLLM host"))


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
        if overrides:
            values.update({str(key): str(value) for key, value in overrides.items()})
        if self.uses_module:
            # Apply overrides first, then prepend the snapshot root: fake-module
            # harnesses also pass PYTHONPATH and must not shadow qasr_cli with an
            # editable install from another checkout.
            existing = values.get("PYTHONPATH")
            values["PYTHONPATH"] = (
                str(REPO_ROOT)
                if not existing
                else os.pathsep.join([str(REPO_ROOT), existing])
            )
        return values

    def run(
        self,
        *args,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 90,
        stdin=None,
    ) -> subprocess.CompletedProcess:
        command = [*self.command, *map(str, args)]
        if isinstance(stdin, (bytes, bytearray)):
            proc = subprocess.run(
                command,
                cwd=str(cwd or self.work),
                env=self.env(env),
                input=bytes(stdin),
                capture_output=True,
                timeout=timeout,
            )
            return subprocess.CompletedProcess(
                proc.args,
                proc.returncode,
                proc.stdout.decode("utf-8", "replace"),
                proc.stderr.decode("utf-8", "replace"),
            )
        return subprocess.run(
            command,
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
        binary: bool = False,
    ) -> subprocess.Popen:
        return subprocess.Popen(
            [*self.command, *map(str, args)],
            cwd=str(cwd or self.work),
            env=self.env(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=not binary,
        )


class FakeStream:
    def __init__(self, root: Path, log: Path) -> None:
        self.root = root
        self.log = log

    def env(self, **overrides: str) -> dict[str, str]:
        values = {
            "PYTHONPATH": str(self.root),
            "QASR_FAKE_STREAM_LOG": str(self.log),
            # The real CLI refuses vLLM off Linux. The review harness fakes
            # the *host* (sitecustomize) as well as the third-party modules;
            # this is test-side only and never an implementation switch.
            "QASR_FAKE_LINUX": "1",
        }
        values.update({key: str(value) for key, value in overrides.items()})
        return values

    def records(self) -> list[dict]:
        return read_jsonl_file(self.log)

    def events(self, name: str) -> list[dict]:
        return [record for record in self.records() if record.get("event") == name]


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
def fake_stream(tmp_path: Path) -> FakeStream:
    root = tmp_path / "fake_stream"
    root.mkdir()
    shutil.copytree(FAKES / "qwen_asr", root / "qwen_asr")
    shutil.copytree(FAKES / "vllm", root / "vllm")
    shutil.copyfile(FAKES / "torch.py", root / "torch.py")
    shutil.copyfile(FAKES / "sitecustomize.py", root / "sitecustomize.py")
    log = tmp_path / "stream.log"
    return FakeStream(root, log)


@pytest.fixture
def fake_mic(tmp_path: Path):
    root = tmp_path / "fake_mic"
    root.mkdir()
    shutil.copyfile(FAKES / "sounddevice.py", root / "sounddevice.py")

    class FakeMic:
        def __init__(self) -> None:
            self.root = root

        def pythonpath(self, *extra: str) -> str:
            return os.pathsep.join([str(root), *map(str, extra)])

        def env(self, pcm: bytes | None = None, **overrides: str) -> dict[str, str]:
            values = {"PYTHONPATH": str(root)}
            if pcm is not None:
                path = tmp_path / "mic.pcm"
                path.write_bytes(pcm)
                values["QASR_FAKE_MIC_PCM"] = str(path)
            values.update({key: str(value) for key, value in overrides.items()})
            return values

    return FakeMic()


@pytest.fixture
def import_blocker(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    marker = tmp_path / "imported.txt"
    for name in ("torch", "vllm", "qwen_asr", "sounddevice", "datasets"):
        (blocker / f"{name}.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text({name!r}, encoding='utf-8')\n"
            f"raise ImportError('blocked {name}')\n",
            encoding="utf-8",
        )
    return blocker, marker


@pytest.fixture
def fake_model(tmp_path: Path) -> Path:
    return make_model_dir(tmp_path / "model")


@pytest.fixture
def short_wav(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "short.wav", seconds=1.0)


@pytest.fixture
def five_second_wav(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "five.wav", seconds=5.0)


@pytest.fixture
def four_second_wav(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "four.wav", seconds=4.0)


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    return write_stereo_wav(tmp_path / "stereo.wav", seconds=1.0)


@pytest.fixture
def one_second_pcm() -> bytes:
    return pcm_s16le(16_000)


@pytest.fixture
def pty_runner(qasr: QasrRunner):
    def run(args, *, env=None, timeout: float = 20.0):
        master, slave = pty.openpty()
        proc = subprocess.Popen(
            [*qasr.command, *map(str, args)],
            cwd=str(qasr.work),
            env=qasr.env(env),
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
        )
        os.close(slave)
        data = bytearray()
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                readable, _, _ = select.select([master], [], [], 0.2)
                if not readable:
                    if proc.poll() is not None:
                        break
                    continue
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                data.extend(chunk)
            if proc.poll() is None:
                proc.kill()
        finally:
            os.close(master)
        return proc.wait(timeout=5), data.decode("utf-8", "replace")

    return run
