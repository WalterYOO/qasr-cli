"""Configuration layering and platform directories.

Priority (low -> high):
    built-in defaults < QASR_HOME/config.json < project .qasr/config.json
    < environment variables < --config file < explicit CLI options.

This module is pure Python and must never import torch/qwen_asr.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import ConfigError

CONFIG_KEYS: tuple[str, ...] = (
    "model",
    "aligner",
    "backend",
    "device",
    "dtype",
    "attn_implementation",
    "language",
    "context",
    "format",
    "batch_size",
    "max_inference_batch_size",
    "jobs",
    "model_dir",
    "offline",
    "server",
    "timeout",
    "retries",
)

DEFAULTS: Dict[str, Any] = {
    "model": "1.7b",
    "aligner": "auto",
    "backend": "auto",
    "device": "auto",
    "dtype": "auto",
    "attn_implementation": "auto",
    "language": "auto",
    "context": [],
    "format": None,
    "batch_size": None,
    "max_inference_batch_size": None,
    "jobs": None,
    "model_dir": None,
    "offline": False,
    "server": None,
    "timeout": 300,
    "retries": 2,
}

ENV_KEYS: Dict[str, str] = {
    "QASR_MODEL": "model",
    "QASR_MODEL_DIR": "model_dir",
    "QASR_BACKEND": "backend",
    "QASR_DEVICE": "device",
    "QASR_DTYPE": "dtype",
    "QASR_OFFLINE": "offline",
    "QASR_SERVE_URL": "server",
    "QASR_LANGUAGE": "language",
}

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

DEVICE_PATTERN = re.compile(r"auto|cpu|mps|cuda|cuda:\d+")
DTYPE_VALUES = ("auto", "bf16", "fp16", "fp32")
BACKEND_VALUES = ("auto", "transformers", "vllm", "remote")
ATTN_IMPLEMENTATION_VALUES = ("auto", "sdpa", "flash_attention_2", "eager")


def qasr_home() -> Path:
    """Resolve QASR_HOME (env override wins)."""
    env = os.environ.get("QASR_HOME")
    if env:
        return Path(env).expanduser().resolve()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return (base / "qasr").resolve()


def models_root(config: Optional[Config] = None) -> Path:
    """Directory that holds model folders."""
    if config is not None and config.get("model_dir"):
        return Path(str(config.get("model_dir"))).expanduser().resolve()
    return qasr_home() / "models"


def cache_dir() -> Path:
    return qasr_home() / "cache"


def project_config_path(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from ``start`` looking for .qasr/config.json."""
    cur = (start or Path.cwd()).resolve()
    for parent in (cur, *cur.parents):
        candidate = parent / ".qasr" / "config.json"
        if candidate.is_file():
            return candidate
    return None


def home_config_path() -> Path:
    return qasr_home() / "config.json"


def config_search_paths(explicit: Optional[str] = None, cwd: Optional[Path] = None) -> List[Path]:
    """Return the config file candidates in priority order (existing or not)."""
    ordered: List[Path] = []
    if explicit:
        ordered.append(Path(explicit).expanduser().resolve())
    env_path = os.environ.get("QASR_CONFIG")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p not in ordered:
            ordered.append(p)
    proj = project_config_path(cwd)
    if proj is not None and proj not in ordered:
        ordered.append(proj)
    home = home_config_path()
    if home not in ordered:
        ordered.append(home)
    return ordered


def active_config_path(explicit: Optional[str] = None, cwd: Optional[Path] = None) -> Path:
    """The config file ``qasr config path`` should print."""
    paths = config_search_paths(explicit, cwd)
    for path in paths:
        if path.is_file():
            return path
    return paths[-1]


def _coerce_bool(value: Any, key: str, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in TRUE_VALUES:
            return True
        if text in FALSE_VALUES:
            return False
        raise ConfigError(
            f"{source}: '{key}' must be a boolean, got {value!r}",
            hint="true|false|1|0|yes|no|on|off",
        )
    if isinstance(value, (int, float)):
        return bool(value)
    raise ConfigError(f"{source}: '{key}' must be a boolean, got {type(value).__name__}")


def _coerce_enum(value: Any, key: str, source: str, allowed: tuple[str, ...]) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{source}: '{key}' must be a string, got {type(value).__name__}")
    normalized = value.strip().lower()
    if key == "device":
        if not DEVICE_PATTERN.fullmatch(normalized):
            raise ConfigError(
                f"{source}: invalid device {value!r}",
                hint="auto|cpu|mps|cuda|cuda:N",
            )
        return normalized
    if normalized not in allowed:
        raise ConfigError(
            f"{source}: invalid {key} {value!r}",
            hint="|".join(allowed),
        )
    return normalized


def _coerce_int(value: Any, key: str, source: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{source}: '{key}' must be an integer, got boolean")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{source}: '{key}' must be an integer, got {value!r}") from exc


def _coerce_context(value: Any, key: str, source: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ConfigError(f"{source}: '{key}' must be a string or list of strings")


def _coerce_format(value: Any, key: str, source: str) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
        return items or None
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ConfigError(f"{source}: '{key}' must be a string or list of strings")


def coerce_value(key: str, value: Any, source: str) -> Any:
    """Type-check a configuration value from any layer."""
    if key == "offline":
        return _coerce_bool(value, key, source)
    if key in ("batch_size", "max_inference_batch_size", "jobs", "timeout", "retries"):
        return _coerce_int(value, key, source)
    if key == "context":
        return _coerce_context(value, key, source)
    if key == "format":
        return _coerce_format(value, key, source)
    if key == "device":
        return _coerce_enum(value, key, source, ())
    if key == "dtype":
        return _coerce_enum(value, key, source, DTYPE_VALUES)
    if key == "backend":
        return _coerce_enum(value, key, source, BACKEND_VALUES)
    if key == "attn_implementation":
        return _coerce_enum(value, key, source, ATTN_IMPLEMENTATION_VALUES)
    if key in ("model", "aligner", "language", "model_dir", "server"):
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConfigError(f"{source}: '{key}' must be a string, got {type(value).__name__}")
        return value
    raise ConfigError(f"{source}: unknown configuration key '{key}'")


@dataclass
class Config:
    values: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    files: List[Dict[str, Any]] = field(default_factory=list)
    explicit_path: Optional[Path] = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def source_of(self, key: str) -> str:
        return self.sources.get(key, "default")

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.values)

    def resolved(self) -> Dict[str, Any]:
        return {
            key: {"value": self.values.get(key), "source": self.source_of(key)}
            for key in CONFIG_KEYS
        }


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level JSON value must be an object")
    return data


def _load_layer(path: Path, source: str, base_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    data = _read_json(path)
    values: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in CONFIG_KEYS:
            raise ConfigError(f"{path}: unknown configuration key '{key}'",
                              hint="allowed keys: " + ", ".join(CONFIG_KEYS))
        value = coerce_value(key, value, str(path))
        if key in ("model_dir",) and isinstance(value, str) and value:
            value = str((base_dir / value).resolve()) if not os.path.isabs(value) else str(Path(value).expanduser())
        values[key] = value
    return values, data


def load_config(
    explicit: Optional[str] = None,
    cwd: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Merge every configuration layer into a single effective ``Config``.

    Low -> high: defaults, QASR_HOME/config.json, project .qasr/config.json,
    environment variables, $QASR_CONFIG file, --config file, CLI options.
    """
    cwd = cwd or Path.cwd()
    values = dict(DEFAULTS)
    sources = {key: "default" for key in CONFIG_KEYS}
    files: List[Dict[str, Any]] = []

    explicit_path = None
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        if not explicit_path.is_file():
            raise ConfigError(f"config file not found: {explicit_path}")

    home = home_config_path()
    project = project_config_path(cwd)
    env_config = Path(os.environ["QASR_CONFIG"]).expanduser().resolve() if os.environ.get("QASR_CONFIG") else None

    base_layers: List[tuple[Path, str]] = [(home, "home")]
    if project is not None:
        base_layers.append((project, "project"))
    high_layers: List[tuple[Path, str]] = []
    if env_config is not None:
        high_layers.append((env_config, "env:QASR_CONFIG"))
    if explicit_path is not None:
        high_layers.append((explicit_path, "cli:--config"))

    for path, layer in base_layers + high_layers:
        entry = {"path": str(path), "exists": path.is_file(), "layer": layer}
        if not any(f["path"] == entry["path"] for f in files):
            files.append(entry)

    for path, layer in base_layers:
        if path.is_file():
            layer_values, _ = _load_layer(path, f"file:{path}", path.parent)
            values.update(layer_values)
            for key in layer_values:
                sources[key] = f"file:{path}"

    for env_name, key in ENV_KEYS.items():
        if env_name in os.environ and os.environ[env_name] != "":
            raw = os.environ[env_name]
            values[key] = coerce_value(key, raw, f"env:{env_name}")
            sources[key] = f"env:{env_name}"

    for path, _layer in high_layers:
        if path.is_file():
            layer_values, _ = _load_layer(path, f"file:{path}", path.parent)
            values.update(layer_values)
            for key in layer_values:
                sources[key] = f"file:{path}"

    for key, value in (cli_overrides or {}).items():
        if value is None:
            continue
        if key not in CONFIG_KEYS:
            continue
        values[key] = coerce_value(key, value, "cli")
        sources[key] = "cli"

    return Config(values=values, sources=sources, files=files, explicit_path=explicit_path)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomic JSON write (temp + fsync + rename)."""
    from .jobs import atomic_write_text

    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def ensure_home() -> Path:
    home = qasr_home()
    home.mkdir(parents=True, exist_ok=True)
    return home
