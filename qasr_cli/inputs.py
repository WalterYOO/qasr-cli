"""Input expansion: files, directories, globs, URLs, and path safety.

Pure Python; must never import torch/qwen_asr.
"""

from __future__ import annotations

import glob as globmod
import hashlib
import os
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .audio import AUDIO_EXTENSIONS
from .config import cache_dir
from .errors import InputError, UsageError

URL_TIMEOUT = 60
URL_MAX_BYTES = 2 * 1024 * 1024 * 1024
URL_CHUNK = 1024 * 1024

GLOB_CHARS = re.compile(r"[*?\[]")
_UNSAFE = re.compile(r"[^\w\-.]+", re.UNICODE)


@dataclass
class InputItem:
    id: str
    source: str
    path: Optional[Path] = None
    rel_path: Optional[str] = None
    url: Optional[str] = None
    base64_data: Optional[str] = None
    origin: str = "file"  # file | dir | glob | url | manifest | base64

    def to_plan(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "path": str(self.path) if self.path else None,
            "origin": self.origin,
        }


@dataclass
class ExpandResult:
    items: List[InputItem] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)


def is_url(value: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(value)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except ValueError:
        return False


def is_data_url(value: str) -> bool:
    return value.startswith("data:audio")


def sanitize_id(value: str, *, fallback: str = "item") -> str:
    """Make a filesystem-safe identifier (no separators, no dot-dot)."""
    value = unicodedata.normalize("NFKC", str(value)).strip()
    value = value.replace("\\", "/")
    parts = [p for p in value.split("/") if p not in ("", ".", "..")]
    cleaned = "_".join(_UNSAFE.sub("_", p).strip("._") for p in parts)
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = fallback
    return cleaned


def sanitize_rel_path(value: str, *, fallback: str = "item") -> str:
    """Sanitize a relative path, preserving directory segments for mirroring."""
    value = unicodedata.normalize("NFKC", str(value)).strip().replace("\\", "/")
    if value.startswith("/") or "\x00" in value:
        raise UsageError(f"unsafe relative path: {value!r}")
    parts: List[str] = []
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise UsageError(f"unsafe relative path (contains '..'): {value!r}")
        cleaned = _UNSAFE.sub("_", part).strip("._")
        if cleaned:
            parts.append(cleaned)
    if not parts:
        return fallback
    return "/".join(parts)


def safe_output_path(out_dir: str | Path, rel: str) -> Path:
    """Join ``rel`` under ``out_dir`` and refuse escapes."""
    if "\x00" in rel:
        raise UsageError("output path contains NUL")
    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise UsageError(f"output path must be relative: {rel}")
    if any(part == ".." for part in rel_path.parts):
        raise UsageError(f"output path must not contain '..': {rel}")
    base = Path(out_dir).expanduser().resolve()
    target = (base / rel_path).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise UsageError(f"output path escapes --out-dir: {rel}") from exc
    return target


def _default_id_for_path(path: Path, rel_base: Optional[Path]) -> Tuple[str, Optional[str]]:
    if rel_base is not None:
        try:
            rel = path.relative_to(rel_base)
        except ValueError:
            rel = Path(path.name)
        rel_no_ext = rel.with_suffix("").as_posix()
        rel_safe = sanitize_rel_path(rel_no_ext, fallback=sanitize_id(path.stem))
        return rel_safe, rel.as_posix()
    return sanitize_id(path.stem), None


def _scan_directory(directory: Path, recursive: bool) -> List[Path]:
    files: List[Path] = []
    if recursive:
        for root, dirnames, filenames in os.walk(directory, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not Path(root, d).is_symlink())
            for name in sorted(filenames):
                p = Path(root) / name
                if p.suffix.casefold() in AUDIO_EXTENSIONS and p.is_file():
                    files.append(p)
    else:
        for entry in sorted(directory.iterdir(), key=lambda p: p.name):
            if entry.is_file() and entry.suffix.casefold() in AUDIO_EXTENSIONS:
                files.append(entry)
    return files


def _expand_glob(pattern: str, recursive: bool) -> List[Path]:
    matches = [Path(p) for p in globmod.glob(pattern, recursive=recursive)]
    out = [
        p for p in matches
        if p.is_file() and p.suffix.casefold() in AUDIO_EXTENSIONS
    ]
    return sorted(out, key=lambda p: str(p))


def download_url(url: str, *, offline: bool = False) -> Path:
    """Download (or reuse cached) URL audio under QASR_HOME/cache/url."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    suffix = Path(urllib.parse.urlparse(url).path).suffix.casefold()
    if suffix not in AUDIO_EXTENSIONS and suffix not in (".bin", ".raw", ".aac", ".wma"):
        suffix = ".bin"
    target = cache_dir() / "url" / f"{digest}{suffix}"
    if target.is_file() and target.stat().st_size > 0:
        return target
    if offline:
        raise InputError(
            f"URL is not cached and --offline is set: {url}",
            hint=f"run once without --offline (cache: {target})",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "qasr-cli"})
        total = 0
        with urllib.request.urlopen(req, timeout=URL_TIMEOUT) as resp, tmp.open("wb") as fh:
            while True:
                block = resp.read(URL_CHUNK)
                if not block:
                    break
                total += len(block)
                if total > URL_MAX_BYTES:
                    raise InputError(f"URL exceeds the 2 GiB limit: {url}")
                fh.write(block)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except InputError:
        tmp.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise InputError(f"download failed for {url}: {exc}") from exc
    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise InputError(f"download produced an empty file: {url}")
    return target


def make_url_item(url: str, *, offline: bool = False) -> InputItem:
    path = download_url(url, offline=offline)
    item_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return InputItem(id=item_id, source=url, path=path, rel_path=None, url=url, origin="url")


def make_base64_item(data: str, item_id: Optional[str] = None) -> InputItem:
    import base64

    payload = data
    if "," in payload and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise InputError(f"invalid base64 audio: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    target = cache_dir() / "base64" / f"{digest}.wav"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".wav.part")
        tmp.write_bytes(raw)
        os.replace(tmp, target)
    return InputItem(
        id=sanitize_id(item_id or digest[:12]),
        source=f"base64:{digest[:12]}",
        path=target,
        rel_path=None,
        base64_data=None,
        origin="base64",
    )


def expand_inputs(
    specs: Sequence[str],
    *,
    recursive: bool = False,
    offline: bool = False,
    cwd: Optional[Path] = None,
) -> ExpandResult:
    """Expand CLI inputs into a stable, de-duplicated list of local items."""
    cwd = cwd or Path.cwd()
    result = ExpandResult()
    seen: set[Path] = set()

    def add(item: InputItem) -> None:
        key = item.path.resolve() if item.path else None
        if key is not None:
            if key in seen:
                return
            seen.add(key)
        result.items.append(item)

    for spec in specs:
        if is_url(spec):
            try:
                add(make_url_item(spec, offline=offline))
            except InputError as exc:
                result.errors.append({"source": spec, "error": exc})
            continue

        expanded = Path(spec).expanduser()
        if not expanded.is_absolute():
            expanded = (cwd / expanded)
        if expanded.is_dir():
            files = _scan_directory(expanded, recursive)
            if not files:
                result.errors.append(
                    {
                        "source": spec,
                        "error": InputError(
                            f"no audio files found in directory: {spec}",
                            hint="supported: " + ", ".join(AUDIO_EXTENSIONS),
                        ),
                    }
                )
                continue
            for path in files:
                item_id, rel = _default_id_for_path(path, expanded)
                add(InputItem(id=item_id, source=str(path), path=path.resolve(),
                              rel_path=rel, origin="dir"))
            continue

        if expanded.is_file():
            item_id, _ = _default_id_for_path(expanded, None)
            add(InputItem(id=item_id, source=spec, path=expanded.resolve(), origin="file"))
            continue

        if GLOB_CHARS.search(spec):
            matches = _expand_glob(spec, recursive)
            if not matches:
                result.errors.append(
                    {"source": spec, "error": InputError(f"glob matched no audio files: {spec}")}
                )
                continue
            for path in matches:
                item_id, _ = _default_id_for_path(path, None)
                add(InputItem(id=item_id, source=str(path), path=path.resolve(), origin="glob"))
            continue

        result.errors.append(
            {
                "source": spec,
                "error": InputError(
                    f"input not found: {spec}",
                    hint="paths, directories, globs and http(s) URLs are accepted",
                ),
            }
        )

    # IDs must be unique; disambiguate deterministically.
    used: Dict[str, int] = {}
    for item in result.items:
        base = item.id
        count = used.get(base, 0)
        used[base] = count + 1
        if count:
            item.id = sanitize_id(f"{base}_{count + 1}")
    return result


def resolve_manifest_audio(record: Dict[str, Any], *, offline: bool = False) -> InputItem:
    """Build an InputItem from a manifest record (audio XOR audio_base64)."""
    audio = record.get("audio")
    b64 = record.get("audio_base64")
    if bool(audio) == bool(b64):
        raise UsageError("manifest record must contain exactly one of 'audio' or 'audio_base64'")
    item_id = record.get("id")
    if b64:
        return make_base64_item(b64, item_id=item_id)
    assert isinstance(audio, str)
    if is_url(audio):
        item = make_url_item(audio, offline=offline)
        if item_id:
            item.id = sanitize_id(item_id)
        return item
    path = Path(audio).expanduser()
    if not path.is_file():
        raise InputError(f"manifest audio not found: {audio}")
    item = InputItem(
        id=sanitize_id(item_id) if item_id else sanitize_id(path.stem),
        source=audio,
        path=path.resolve(),
        origin="manifest",
    )
    return item


def default_manifest_id(record: Dict[str, Any]) -> str:
    seed = record.get("audio") or record.get("audio_base64") or ""
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:12]
