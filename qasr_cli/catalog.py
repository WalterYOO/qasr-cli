"""Model aliases, resolution, integrity verification and fingerprinting.

Pure Python; must never import torch/qwen_asr.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import SCHEMA_VERSION
from .config import Config, models_root, qasr_home, write_json_atomic
from .errors import InputError, ModelError, UsageError
from .jobs import now_iso

PULL_HINT = "qasr models pull {alias} --resume"

# Files whose presence means a download is still in flight or was interrupted.
PARTIAL_SUFFIXES = (".aria2", ".incomplete", ".part", ".tmp", ".download")


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo: str
    kind: str  # "asr" | "aligner"


ALIASES: Dict[str, ModelSpec] = {
    "1.7b": ModelSpec("1.7b", "Qwen/Qwen3-ASR-1.7B", "asr"),
    "0.6b": ModelSpec("0.6b", "Qwen/Qwen3-ASR-0.6B", "asr"),
    "aligner": ModelSpec("aligner", "Qwen/Qwen3-ForcedAligner-0.6B", "aligner"),
}

REPO_TO_ALIAS = {spec.repo.casefold(): spec for spec in ALIASES.values()}
BASENAME_TO_ALIAS = {spec.repo.rsplit("/", 1)[-1].casefold(): spec for spec in ALIASES.values()}


@dataclass
class ResolvedModel:
    value: str
    alias: Optional[str] = None
    repo: Optional[str] = None
    path: Optional[Path] = None
    source: str = "default"
    kind: str = "asr"

    @property
    def present(self) -> bool:
        return self.path is not None and self.path.is_dir()

    @property
    def display(self) -> str:
        return str(self.path) if self.path is not None else (self.repo or self.value)


def spec_for(value: str) -> Optional[ModelSpec]:
    if not value:
        return None
    key = value.strip().casefold().rstrip("/")
    if key in ALIASES:
        return ALIASES[key]
    if key in REPO_TO_ALIAS:
        return REPO_TO_ALIAS[key]
    if key in BASENAME_TO_ALIAS:
        return BASENAME_TO_ALIAS[key]
    return None


def _is_path_like(value: str) -> bool:
    if value.startswith((".", "~", "/", os.sep)):
        return True
    if os.sep in value or (os.altsep and os.altsep in value):
        # repo ids contain "/" too; only treat as a path when it exists.
        return Path(value).expanduser().exists()
    return False


def candidate_dirs(value: str, config: Optional[Config] = None, cwd: Optional[Path] = None) -> List[Path]:
    """Local directories that might hold the requested model, in priority order."""
    cwd = cwd or Path.cwd()
    spec = spec_for(value)
    names: List[str] = []
    if spec is not None:
        names.extend([spec.alias, spec.repo.rsplit("/", 1)[-1]])
    else:
        cleaned = value.strip().rstrip("/")
        if cleaned:
            names.append(cleaned.rsplit("/", 1)[-1])
    roots: List[Path] = [models_root(config)]
    roots.extend([cwd / "models", cwd.parent / "Qwen3-ASR" / "models"])
    out: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for name in names:
            cand = (root / name).expanduser()
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def resolve_model(
    value: Optional[str],
    config: Optional[Config] = None,
    cwd: Optional[Path] = None,
) -> ResolvedModel:
    """Resolve a user value (alias/path/repo) to a local path when possible.

    Order: explicit path -> alias/repo local lookup -> online repo id.
    """
    if value is None or str(value).strip() == "":
        value = str((config.get("model") if config else None) or "1.7b")
    value = str(value).strip()
    spec = spec_for(value)
    source = "cli"
    if config is not None and value == str(config.get("model")):
        source = config.source_of("model")

    expanded = Path(value).expanduser()
    if expanded.exists():
        path = expanded.resolve()
        kind = "aligner" if "forcedaligner" in str(path).casefold() else (spec.kind if spec else "asr")
        return ResolvedModel(value=value, alias=spec.alias if spec else None,
                             repo=spec.repo if spec else None, path=path, source=source, kind=kind)

    for cand in candidate_dirs(value, config, cwd):
        if cand.is_dir():
            return ResolvedModel(
                value=value,
                alias=spec.alias if spec else (BASENAME_TO_ALIAS.get(cand.name.casefold()).alias
                                               if cand.name.casefold() in BASENAME_TO_ALIAS else None),
                repo=spec.repo if spec else None,
                path=cand.resolve(),
                source=source if cand.parent == models_root(config) else "dev-path",
                kind=spec.kind if spec else ("aligner" if "forcedaligner" in cand.name.casefold() else "asr"),
            )

    if spec is not None:
        return ResolvedModel(value=value, alias=spec.alias, repo=spec.repo, path=None,
                             source=source, kind=spec.kind)
    return ResolvedModel(value=value, alias=None, repo=value if "/" in value else None,
                         path=None, source=source,
                         kind="aligner" if "forcedaligner" in value.casefold() else "asr")


def resolve_aligner(
    value: Optional[str],
    config: Optional[Config] = None,
    cwd: Optional[Path] = None,
    *,
    required: bool = False,
) -> Optional[ResolvedModel]:
    """Resolve the aligner model (``auto``/``none``/path/repo)."""
    raw = (value if value is not None else (config.get("aligner") if config else "auto"))
    raw = str(raw or "auto").strip()
    if raw.casefold() == "none":
        if required:
            raise ModelError("this operation requires a forced aligner but --aligner none was given",
                             hint="pass --aligner auto or a local aligner path")
        return None
    if raw.casefold() == "auto":
        resolved = resolve_model("aligner", config, cwd)
        if not resolved.present:
            if required:
                raise ModelError(
                    "forced aligner is not available locally",
                    hint=PULL_HINT.format(alias="aligner"),
                )
            return resolved
        return resolved
    resolved = resolve_model(raw, config, cwd)
    resolved.kind = "aligner"
    if required and not resolved.present:
        raise ModelError(f"aligner not found: {raw}", hint="check the path or run 'qasr models pull aligner'")
    return resolved


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

LARGE_FILE_BYTES = 32 * 1024 * 1024
FP_CHUNK = 1024 * 1024


def _hash_file_range(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        fh.seek(offset)
        remaining = length
        while remaining > 0:
            block = fh.read(min(FP_CHUNK, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def fingerprint_file(path: str | os.PathLike, *, full: bool = False) -> str:
    """``sha256:...`` for small files; size + head/tail 1MiB hash for large ones."""
    target = Path(path)
    size = target.stat().st_size
    if full or size <= LARGE_FILE_BYTES:
        digest = hashlib.sha256()
        with target.open("rb") as fh:
            while True:
                block = fh.read(FP_CHUNK)
                if not block:
                    break
                digest.update(block)
        return "sha256:" + digest.hexdigest()
    head = _hash_file_range(target, 0, FP_CHUNK)
    tail = _hash_file_range(target, max(0, size - FP_CHUNK), FP_CHUNK)
    digest = hashlib.sha256()
    digest.update(f"{size}:{head}:{tail}".encode("ascii"))
    return "sha256-partial:" + digest.hexdigest()


class FingerprintCache:
    """Cache fingerprints keyed by path + size + mtime."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (qasr_home() / "cache" / "fingerprints.json")
        self._data: Dict[str, Any] = {}
        self._dirty = False
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self._data, dict):
                self._data = {}
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def file_fingerprint(self, path: str | os.PathLike, *, full: bool = False) -> str:
        target = Path(path).resolve()
        stat = target.stat()
        key = str(target)
        entry = self._data.get(key)
        if (
            isinstance(entry, dict)
            and entry.get("size") == stat.st_size
            and entry.get("mtime_ns") == stat.st_mtime_ns
            and (entry.get("full") or not full)
        ):
            return entry["fingerprint"]
        fp = fingerprint_file(target, full=full)
        self._data[key] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "fingerprint": fp,
            "full": bool(full),
        }
        self._dirty = True
        return fp

    def dir_fingerprint(self, directory: str | os.PathLike, *, full: bool = False) -> str:
        root = Path(directory)
        parts: List[str] = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.name.startswith(".") and path.suffix in (".tmp",):
                continue
            rel = path.relative_to(root).as_posix()
            parts.append(f"{rel}:{self.file_fingerprint(path, full=full)}")
        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        return "sha256:" + digest

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------

def _check(name: str, status: str, detail: str, value: Any = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": name, "status": status, "detail": detail}
    if value is not None:
        out["value"] = value
    return out


def _parse_safetensors_header(path: Path) -> tuple[int, int]:
    """Return (header_len, expected_file_size); raise ValueError when corrupt."""
    size = path.stat().st_size
    if size < 9:
        raise ValueError("file too small")
    with path.open("rb") as fh:
        raw = fh.read(8)
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 0 or header_len > 100 * 1024 * 1024:
            raise ValueError(f"implausible header length {header_len}")
        if 8 + header_len > size:
            raise ValueError("truncated header")
        header = json.loads(fh.read(header_len))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    end = 0
    for entry in tensors.values():
        offsets = entry.get("data_offsets")
        if not offsets or len(offsets) != 2:
            raise ValueError("tensor without data_offsets")
        end = max(end, int(offsets[1]))
    return header_len, 8 + header_len + end


def _check_weights(directory: Path) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = sorted(set(index.get("weight_map", {}).values()))
        except (OSError, json.JSONDecodeError) as exc:
            checks.append(_check("weights-index", "fail", f"cannot read index: {exc}"))
            return checks
        checks.append(_check("weights-index", "ok", f"{len(shards)} shard(s) declared", len(shards)))
        for shard in shards:
            shard_path = directory / shard
            if not shard_path.is_file():
                checks.append(_check(f"shard:{shard}", "fail", "missing shard"))
                continue
            checks.extend(_check_single_safetensors(shard_path, shard))
        return checks

    single = directory / "model.safetensors"
    if single.is_file():
        checks.extend(_check_single_safetensors(single, "model.safetensors"))
        return checks

    checks.append(_check("weights", "fail", "no model.safetensors or model.safetensors.index.json"))
    return checks


def _check_single_safetensors(path: Path, label: str) -> List[Dict[str, Any]]:
    try:
        _header_len, expected = _parse_safetensors_header(path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return [_check(f"shard:{label}", "fail", f"unparseable safetensors header: {exc}")]
    actual = path.stat().st_size
    if actual != expected:
        return [
            _check(
                f"shard:{label}",
                "fail",
                f"size mismatch: expected {expected} bytes, found {actual} "
                f"({actual - expected:+d})",
                {"expected": expected, "actual": actual},
            )
        ]
    return [_check(f"shard:{label}", "ok", f"{actual} bytes, header ok", actual)]


def find_partial_files(directory: Path, max_depth: int = 2) -> List[Path]:
    found: List[Path] = []
    root_depth = len(directory.parts)
    for dirpath, dirnames, filenames in os.walk(directory):
        current = Path(dirpath)
        if len(current.parts) - root_depth >= max_depth:
            dirnames[:] = []
        for name in filenames:
            if name.casefold().endswith(PARTIAL_SUFFIXES):
                found.append(current / name)
    return sorted(found)


def verify_model(
    resolved: ResolvedModel,
    *,
    full: bool = False,
    cache: Optional[FingerprintCache] = None,
) -> Dict[str, Any]:
    """Return a verify payload with ``status`` = complete | incomplete | missing."""
    checks: List[Dict[str, Any]] = []
    path = resolved.path

    if path is None or not path.is_dir():
        checks.append(_check("directory", "fail", f"not found: {resolved.display}"))
        return _verify_payload(resolved, "missing", checks, None, None)

    checks.append(_check("directory", "ok", str(path)))

    partials = find_partial_files(path)
    if partials:
        checks.append(
            _check(
                "partial-download",
                "fail",
                "interrupted download markers: " + ", ".join(p.name for p in partials[:5]),
                len(partials),
            )
        )
    else:
        checks.append(_check("partial-download", "ok", "no .aria2/.incomplete/.part files"))

    for name in ("config.json", "tokenizer_config.json"):
        target = path / name
        checks.append(
            _check(f"file:{name}", "ok" if target.is_file() else "fail",
                   "present" if target.is_file() else "missing")
        )
    tokenizer_files = [n for n in ("tokenizer.json", "vocab.json") if (path / n).is_file()]
    checks.append(
        _check("tokenizer-vocab", "ok" if tokenizer_files else "fail",
               "present: " + ", ".join(tokenizer_files) if tokenizer_files else "missing tokenizer.json/vocab.json")
    )

    checks.extend(_check_weights(path))

    status = "complete" if all(c["status"] == "ok" for c in checks) else "incomplete"

    fingerprint = None
    size_bytes = None
    try:
        cache = cache or FingerprintCache()
        size_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        if status == "complete":
            fingerprint = cache.dir_fingerprint(path, full=full)
            cache.flush()
    except OSError as exc:
        checks.append(_check("fingerprint", "warn", f"could not fingerprint: {exc}"))

    return _verify_payload(resolved, status, checks, fingerprint, size_bytes)


def _verify_payload(
    resolved: ResolvedModel,
    status: str,
    checks: List[Dict[str, Any]],
    fingerprint: Optional[str],
    size_bytes: Optional[int],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ok": status == "complete",
        "command": "models",
        "action": "verify",
        "alias": resolved.alias,
        "repo": resolved.repo,
        "path": str(resolved.path) if resolved.path else None,
        "status": status,
        "checks": checks,
        "fingerprint": fingerprint,
        "size_bytes": size_bytes,
    }
    if status != "complete":
        alias = resolved.alias or resolved.value
        payload["hint"] = PULL_HINT.format(alias=alias)
    return payload


# ---------------------------------------------------------------------------
# Listing / pulling
# ---------------------------------------------------------------------------

def models_list(config: Optional[Config] = None, cwd: Optional[Path] = None) -> Dict[str, Any]:
    cache = FingerprintCache()
    entries: List[Dict[str, Any]] = []
    for alias in ("1.7b", "0.6b", "aligner"):
        resolved = resolve_model(alias, config, cwd)
        verify = verify_model(resolved, cache=cache)
        entries.append(
            {
                "alias": alias,
                "repo": resolved.repo,
                "path": str(resolved.path) if resolved.path else None,
                "present": resolved.present,
                "status": verify["status"],
                "size_bytes": verify["size_bytes"],
                "fingerprint": verify["fingerprint"],
                "capabilities": (
                    ["transcribe", "transcribe.timestamps", "align"]
                    if resolved.kind == "aligner"
                    else ["transcribe", "transcribe.lid", "transcribe.context"]
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": "models",
        "action": "list",
        "models": entries,
    }


def _snapshot_download(fn: Any, *args: Any, resume: bool, **kwargs: Any) -> Any:
    """Call ``snapshot_download`` with resume where the library supports it."""
    if resume:
        try:
            return fn(*args, resume_download=True, **kwargs)
        except TypeError:
            # Older hub releases dropped resume_download (always resuming).
            pass
    return fn(*args, **kwargs)


def _write_source_json(dest: Path, *, repo: str, revision: Optional[str],
                       source: str) -> None:
    """Record where the weights came from (repo/revision/files/sizes)."""
    files: List[Dict[str, Any]] = []
    for path in sorted(dest.rglob("*")):
        if path.is_file() and path.name != "source.json":
            files.append({
                "path": path.relative_to(dest).as_posix(),
                "size": path.stat().st_size,
            })
    payload = {
        "repo": repo,
        "revision": revision,
        "source": source,
        "files": files,
        "total_size": sum(item["size"] for item in files),
        "created_at": now_iso(),
    }
    write_json_atomic(dest / "source.json", payload)


def pull_model(
    value: str,
    *,
    source: str = "hf",
    revision: Optional[str] = None,
    offline: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    spec = spec_for(value)
    if spec is None:
        raise InputError(f"unknown model alias: {value}",
                         hint="known aliases: " + ", ".join(ALIASES))
    if offline:
        raise ModelError("refusing to download in offline mode", hint="unset --offline / QASR_OFFLINE")

    dest = qasr_home() / "models" / spec.repo.rsplit("/", 1)[-1]
    dest.mkdir(parents=True, exist_ok=True)

    kwargs: Dict[str, Any] = {"revision": revision} if revision else {}
    try:
        if source == "hf":
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ModelError(
                    "huggingface_hub is not installed",
                    hint='uv pip install "qasr-cli[hf]"  (or use --source modelscope)',
                ) from exc
            _snapshot_download(snapshot_download, repo_id=spec.repo,
                               local_dir=str(dest), resume=resume, **kwargs)
        elif source == "modelscope":
            try:
                from modelscope import snapshot_download as ms_download
            except ImportError as exc:
                raise ModelError(
                    "modelscope is not installed",
                    hint='uv pip install "qasr-cli[modelscope]"',
                ) from exc
            _snapshot_download(ms_download, spec.repo, local_dir=str(dest),
                               resume=resume, **kwargs)
        else:
            raise InputError(f"unknown --source: {source}", hint="use hf or modelscope")
    except (ModelError, InputError):
        raise
    except Exception as exc:  # network/API errors -> exit 3
        raise ModelError(f"download failed for {spec.repo}: {exc}",
                         hint=PULL_HINT.format(alias=spec.alias)) from exc

    try:
        _write_source_json(dest, repo=spec.repo, revision=revision, source=source)
    except OSError:
        pass  # metadata is best-effort; the weights are what matter
    resolved = ResolvedModel(value=value, alias=spec.alias, repo=spec.repo,
                             path=dest.resolve(), source="pull", kind=spec.kind)
    verify = verify_model(resolved)
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": verify["status"] == "complete",
        "command": "models",
        "action": "pull",
        "alias": spec.alias,
        "repo": spec.repo,
        "path": str(dest),
        "source": source,
        "revision": revision,
        "status": verify["status"],
        "checks": verify["checks"],
        "hint": verify.get("hint"),
    }


def models_path(value: str, config: Optional[Config] = None, cwd: Optional[Path] = None) -> Dict[str, Any]:
    resolved = resolve_model(value, config, cwd)
    if not resolved.present:
        raise ModelError(
            f"model '{value}' is not available locally",
            hint=PULL_HINT.format(alias=resolved.alias or value),
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": "models",
        "action": "path",
        "alias": resolved.alias,
        "repo": resolved.repo,
        "kind": resolved.kind,
        "path": str(resolved.path),
        "source": resolved.source,
    }


def dir_size(path: Optional[Path]) -> Optional[int]:
    if path is None or not path.is_dir():
        return None
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def require_local(resolved: ResolvedModel, *, action: str = "transcribe") -> Path:
    if not resolved.present or resolved.path is None:
        alias = resolved.alias or resolved.value
        raise ModelError(
            f"model '{resolved.value}' is not available locally"
            + (" (online download is P1)" if resolved.repo else ""),
            hint=PULL_HINT.format(alias=alias) if resolved.alias else None,
            details={"action": action},
        )
    return resolved.path


__all__ = [
    "ALIASES",
    "FingerprintCache",
    "ModelSpec",
    "ResolvedModel",
    "candidate_dirs",
    "find_partial_files",
    "models_list",
    "models_path",
    "show_model",
    "remove_model",
    "pull_model",
    "require_local",
    "resolve_aligner",
    "resolve_model",
    "spec_for",
    "verify_model",
]


def show_model(value: str, config: Optional[Config] = None) -> Dict[str, Any]:
    """Detailed single-model view used by ``qasr models show``."""
    resolved = resolve_model(value, config)
    verify = verify_model(resolved)
    source_summary = None
    if resolved.path is not None:
        source_file = Path(resolved.path) / "source.json"
        if source_file.is_file():
            try:
                raw = json.loads(source_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raw = None
            if isinstance(raw, dict):
                source_summary = {
                    key: raw.get(key)
                    for key in ("source", "repo", "revision")
                    if key in raw
                }
    file_count = None
    if resolved.path is not None and Path(resolved.path).is_dir():
        file_count = sum(1 for item in Path(resolved.path).rglob("*") if item.is_file())
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": "models",
        "action": "show",
        "alias": resolved.alias,
        "value": resolved.value,
        "repo": resolved.repo,
        "kind": resolved.kind,
        "path": str(resolved.path) if resolved.path else None,
        "present": resolved.present,
        "source": resolved.source,
        "status": verify["status"],
        "size_bytes": verify["size_bytes"],
        "file_count": file_count,
        "fingerprint": verify["fingerprint"],
        "source_json": source_summary,
        "checks": verify.get("checks", []),
    }


def remove_model(
    value: str,
    config: Optional[Config] = None,
    *,
    assume_yes: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Delete a managed model directory under ``models_root``."""
    # Check the user-supplied candidate for a symlink *before* resolve_model
    # canonicalises it: otherwise a symlinked alias would be silently resolved
    # to the real directory and we would delete that instead.
    import os
    import sys

    root_path = models_root(config)
    spec = spec_for(value)
    if spec is not None:
        candidate = root_path / spec.alias
    elif os.sep in value or (os.altsep and os.altsep in value):
        candidate = Path(value).expanduser()
    else:
        candidate = root_path / value
    if os.path.islink(candidate):
        raise UsageError(
            f"refusing to remove symlinked model directory: {candidate}",
            hint="remove the link manually; qasr only deletes managed directories",
        )

    resolved = resolve_model(value, config)
    alias = resolved.alias or resolved.value
    if resolved.path is None:
        raise ModelError(f"model '{value}' is not present locally")
    target = Path(resolved.path)
    if target.is_symlink():
        raise UsageError(f"refusing to remove symlinked model directory: {target}")
    root = models_root(config).resolve()
    try:
        real = target.resolve(strict=True)
    except OSError as exc:
        raise ModelError(f"model '{value}' is not present locally") from exc
    if real == root or root not in real.parents:
        raise UsageError(
            f"refusing to remove a non-managed directory: {target}",
            hint=f"only directories under {root} are managed by qasr",
        )
    if real.parent != root:
        raise UsageError(f"refusing to remove nested directory: {target}")

    size = dir_size(real) or 0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": "models",
        "action": "rm",
        "alias": alias,
        "value": resolved.value,
        "kind": resolved.kind,
        "present": True,
        "status": "present",
        "path": str(real),
        "size_bytes": size,
        "dry_run": bool(dry_run),
        "removed": False,
    }
    if dry_run:
        return payload
    if not assume_yes:
        if not sys.stdin.isatty():
            raise UsageError(
                "refusing to delete without confirmation",
                hint="pass --yes in non-interactive runs",
            )
        answer = input(f"remove {alias} at {real}? [y/N] ").strip().casefold()
        if answer not in ("y", "yes"):
            raise UsageError("aborted")
    import shutil

    shutil.rmtree(real)
    payload["removed"] = True
    payload["present"] = False
    payload["status"] = "removed"
    return payload
