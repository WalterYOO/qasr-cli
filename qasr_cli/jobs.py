"""Durable job state: atomic writes, locks, state.jsonl and resume keys.

Pure Python; must never import torch/qwen_asr.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import SCHEMA_VERSION, __version__
from .errors import InputError, UsageError

STATE_DIRNAME = ".qasr"
STATE_FILENAME = "state.jsonl"
MANIFEST_FILENAME = "manifest.json"
LOCK_FILENAME = "lock"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, *, chunk_size: int = CHUNK) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        _fsync_dir(target.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def append_jsonl(path: str | os.PathLike, record: Dict[str, Any]) -> None:
    """Append one JSON line and fsync (state durability)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def job_key(
    request: Dict[str, Any],
    *,
    audio_sha256: str,
    model_fingerprint: Optional[str],
    aligner_fingerprint: Optional[str],
    backend: str,
    output_rel: str,
    qasr_version: str = __version__,
) -> str:
    payload = canonical_json(
        {
            "request": request,
            "audio_sha256": audio_sha256,
            "model_fingerprint": model_fingerprint,
            "aligner_fingerprint": aligner_fingerprint,
            "backend": backend,
            "qasr_version": qasr_version,
            "output_rel": output_rel,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


class OutputLock:
    """PID-based lock in ``<out-dir>/.qasr/lock``."""

    def __init__(self, out_dir: str | os.PathLike) -> None:
        self.state_dir = Path(out_dir) / STATE_DIRNAME
        self.path = self.state_dir / LOCK_FILENAME
        self.acquired = False

    def check(self) -> None:
        """Raise if a live process holds the lock (stale locks are ignored)."""
        if not self.path.exists():
            return
        holder = self._read_holder()
        if holder is None:
            return
        if _pid_alive(holder.get("pid")) and holder.get("host") == socket.gethostname():
            raise UsageError(
                f"output directory is locked by pid {holder.get('pid')}",
                hint=f"remove {self.path} if that process is gone",
            )

    def acquire(self) -> OutputLock:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.check()
        self.path.write_text(
            json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "created_at": now_iso()}),
            encoding="utf-8",
        )
        self.acquired = True
        return self

    def _read_holder(self) -> Optional[Dict[str, Any]]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
        # Tolerate a bare PID written by other tooling (JSON parses it as int).
        try:
            return {"pid": int(raw.strip()), "host": socket.gethostname()}
        except ValueError:
            return None

    def release(self) -> None:
        if self.acquired:
            try:
                holder = self._read_holder()
                if holder is None or holder.get("pid") == os.getpid():
                    self.path.unlink(missing_ok=True)
            except OSError:
                pass
            self.acquired = False

    def __enter__(self) -> OutputLock:
        return self.acquire()

    def __exit__(self, *_exc: Any) -> None:
        self.release()


def ensure_output_unlocked(out_dir: str | os.PathLike) -> None:
    """Fail fast (before model work) when another live run owns ``out_dir``."""
    OutputLock(out_dir).check()


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_state(state_path: str | os.PathLike) -> Dict[str, Dict[str, Any]]:
    """Return the latest record per id."""
    path = Path(state_path)
    latest: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = record.get("id")
            if isinstance(item_id, str):
                latest[item_id] = record
    return latest


def recover_running(state_path: str | os.PathLike, ids: Iterable[str]) -> List[str]:
    """Rewrite stale ``running`` records as ``pending``; return the ids touched."""
    latest = read_state(state_path)
    touched: List[str] = []
    for item_id in ids:
        record = latest.get(item_id)
        if record and record.get("status") == STATUS_RUNNING:
            append_jsonl(
                state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": item_id,
                    "job_key": record.get("job_key"),
                    "status": STATUS_PENDING,
                    "outputs": [],
                    "error": None,
                    "started_at": None,
                    "finished_at": now_iso(),
                },
            )
            touched.append(item_id)
    return touched


def outputs_match(outputs: Any) -> bool:
    """Every recorded output must still exist with the recorded SHA-256."""
    if not isinstance(outputs, list) or not outputs:
        return False
    for entry in outputs:
        if not isinstance(entry, dict):
            return False
        path = entry.get("path")
        digest = entry.get("sha256")
        if not path or not digest:
            return False
        target = Path(path)
        if not target.is_file():
            return False
        if sha256_file(target) != digest:
            return False
    return True


MANIFEST_KEYS = {
    "id",
    "audio",
    "audio_base64",
    "language",
    "context",
    "formats",
    "output",
    "timestamps",
    "granularity",
}

GRANULARITIES = ("sentence", "word", "char")


def load_manifest(path: str | os.PathLike) -> List[Dict[str, Any]]:
    """Parse and strictly validate a JSONL manifest."""
    target = Path(path)
    if not target.is_file():
        raise InputError(f"manifest not found: {target}")
    records: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UsageError(f"manifest line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise UsageError(f"manifest line {lineno}: expected an object")
        unknown = set(record) - MANIFEST_KEYS
        if unknown:
            raise UsageError(
                f"manifest line {lineno}: unknown field(s): {', '.join(sorted(unknown))}",
                hint="allowed: " + ", ".join(sorted(MANIFEST_KEYS)),
            )
        if bool(record.get("audio")) == bool(record.get("audio_base64")):
            raise UsageError(
                f"manifest line {lineno}: exactly one of 'audio' or 'audio_base64' is required"
            )
        if "formats" in record:
            formats = record["formats"]
            if not isinstance(formats, list) or not all(isinstance(f, str) for f in formats):
                raise UsageError(f"manifest line {lineno}: 'formats' must be a list of strings")
        if "granularity" in record and record["granularity"] not in GRANULARITIES:
            raise UsageError(
                f"manifest line {lineno}: 'granularity' must be one of {GRANULARITIES}"
            )
        if "timestamps" in record and not isinstance(record["timestamps"], bool):
            raise UsageError(f"manifest line {lineno}: 'timestamps' must be a boolean")
        if "output" in record:
            output = record["output"]
            if not isinstance(output, str) or not output:
                raise UsageError(f"manifest line {lineno}: 'output' must be a non-empty string")
            if "\x00" in output:
                raise UsageError(f"manifest line {lineno}: 'output' contains NUL")
            if Path(output).is_absolute():
                raise UsageError(f"manifest line {lineno}: 'output' must be relative to --out-dir")
            if any(part == ".." for part in Path(output).parts):
                raise UsageError(f"manifest line {lineno}: 'output' must not contain '..'")
        from .inputs import default_manifest_id

        record.setdefault("id", default_manifest_id(record))
        if not isinstance(record["id"], str) or not record["id"].strip():
            raise UsageError(f"manifest line {lineno}: 'id' must be a non-empty string")
        if record["id"] in seen_ids:
            raise UsageError(f"manifest line {lineno}: duplicate id '{record['id']}'")
        seen_ids.add(record["id"])
        records.append(record)
    if not records:
        raise UsageError(f"manifest is empty: {target}")
    return records


def should_skip(
    record: Optional[Dict[str, Any]],
    key: str,
    *,
    resume: bool,
    overwrite: bool,
) -> bool:
    if overwrite or not resume or not record:
        return False
    return (
        record.get("status") in (STATUS_DONE, STATUS_SKIPPED)
        and record.get("job_key") == key
        and outputs_match(record.get("outputs"))
    )
