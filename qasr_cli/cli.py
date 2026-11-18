"""argparse command surface, dispatch, and JSON/error exit handling."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from typing import Any, Dict, Optional, Sequence

from . import __version__
from .commands import align as align_cmd
from .commands import config as config_cmd
from .commands import doctor as doctor_cmd
from .commands import languages as languages_cmd
from .commands import models as models_cmd
from .commands import serve as serve_cmd
from .commands import transcribe as transcribe_cmd
from .commands.common import CommandContext
from .config import load_config
from .errors import (
    EXIT_INTERRUPT,
    EXIT_USAGE,
    QasrError,
    UsageError,
)
from .render import error_payload

PROG = "qasr"

COMMAND_HANDLERS = {
    "transcribe": transcribe_cmd.handle,
    "align": align_cmd.handle,
    "models": models_cmd.handle,
    "languages": languages_cmd.handle,
    "doctor": doctor_cmd.handle,
    "config": config_cmd.handle,
    "serve": serve_cmd.handle,
}

CLI_CONFIG_KEYS = (
    "model",
    "aligner",
    "backend",
    "device",
    "dtype",
    "attn_implementation",
    "model_dir",
    "offline",
    "server",
    "timeout",
    "retries",
)


class QasrParser(argparse.ArgumentParser):
    """ArgumentParser that raises UsageError instead of calling sys.exit(2)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message, hint=self.format_usage().strip())


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Global flags, accepted both before and after the subcommand.

    Defaults are SUPPRESS so a subparser never clobbers a value that was set
    before the subcommand.
    """
    parser.add_argument("--config", metavar="PATH", default=argparse.SUPPRESS,
                        help="JSON config file (merged on top of other layers)")
    parser.add_argument("--model", metavar="NAME|PATH", default=argparse.SUPPRESS,
                        help="alias 1.7b|0.6b, a local path, or a HF repo id")
    parser.add_argument("--aligner", metavar="auto|none|PATH", default=argparse.SUPPRESS,
                        help="forced aligner model (default: auto)")
    parser.add_argument("--backend", metavar="auto|transformers|vllm|remote",
                        default=argparse.SUPPRESS,
                        help="execution backend (remote is implied by --server)")
    parser.add_argument("--device", metavar="auto|cpu|mps|cuda|cuda:N",
                        default=argparse.SUPPRESS)
    parser.add_argument("--dtype", metavar="auto|bf16|fp16|fp32", default=argparse.SUPPRESS)
    parser.add_argument("--attn-implementation", metavar="auto|sdpa|flash_attention_2|eager",
                        dest="attn_implementation", default=argparse.SUPPRESS)
    parser.add_argument("--model-dir", metavar="PATH", dest="model_dir",
                        default=argparse.SUPPRESS, help="root directory holding model folders")
    parser.add_argument("--offline", action="store_true", default=argparse.SUPPRESS,
                        help="never touch the network")
    parser.add_argument("--server", metavar="URL", default=argparse.SUPPRESS,
                        help="remote OpenAI-compatible server (implies --backend remote)")
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="write one JSON object to stdout")
    parser.add_argument("--jsonl", action="store_true", default=argparse.SUPPRESS,
                        help="write one JSON object per line to stdout")
    parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                        help="resolve and plan only; do not load weights")
    parser.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--log-level", dest="log_level", default=argparse.SUPPRESS,
                        choices=["debug", "info", "warning", "error"])


def _add_transcribe(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("transcribe", help="transcribe one or more audio inputs",
                       description="Transcribe files, directories, globs, URLs or a manifest.")
    _add_common_flags(p)
    p.add_argument("inputs", nargs="*", metavar="INPUT")
    p.add_argument("-f", "--format", metavar="LIST", default=None,
                   help="txt,json,jsonl,srt,vtt (comma separated)")
    p.add_argument("-o", "--out-dir", dest="out_dir", metavar="DIR", default=None)
    p.add_argument("--output", metavar="PATH|-", default=None,
                   help="single output file, or '-' for stdout")
    p.add_argument("-l", "--language", metavar="auto|zh|en|...", default=None)
    p.add_argument("--context", action="append", default=[], metavar="TEXT")
    p.add_argument("--context-file", dest="context_file", metavar="PATH", default=None)
    p.add_argument("--timestamps", action="store_true", default=False)
    p.add_argument("--granularity", choices=["sentence", "word", "char"], default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--max-inference-batch-size", dest="max_inference_batch_size",
                   type=int, default=None)
    p.add_argument("--max-new-tokens", dest="max_new_tokens", default=None,
                   help="'auto' (default) or a positive integer")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--timeout", type=float, default=argparse.SUPPRESS, metavar="SECONDS")
    p.add_argument("--retries", type=int, default=argparse.SUPPRESS)
    p.add_argument("--recursive", action="store_true", default=False)
    p.add_argument("--manifest", metavar="FILE", default=None)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--continue-on-error", dest="continue_on_error", action="store_true",
                   default=True)
    p.add_argument("--fail-fast", dest="fail_fast", action="store_true", default=False)
    p.add_argument("--no-progress", dest="no_progress", action="store_true", default=False)


def _add_align(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("align", help="force-align a known transcript",
                       description="Forced alignment for one audio file (P0: <=180 s).")
    _add_common_flags(p)
    p.add_argument("audio", metavar="AUDIO")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--text", metavar="TEXT", default=None)
    group.add_argument("--text-file", dest="text_file", metavar="FILE", default=None)
    group.add_argument("--srt", metavar="FILE", default=None)
    p.add_argument("-l", "--language", metavar="auto|zh|en|...", default=None,
                   help="one of the 11 alignable languages; 'auto' uses script detection")
    p.add_argument("--granularity", choices=["sentence", "word", "char"], default="word")
    p.add_argument("-f", "--format", metavar="LIST", default=None,
                   help="json,srt,vtt,txt (default: json)")
    p.add_argument("-o", "--out-dir", dest="out_dir", metavar="DIR", default=None)
    p.add_argument("--output", metavar="PATH|-", default=None)
    p.add_argument("--coverage-min", dest="coverage_min", type=float, default=0.5)


def _add_models(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("models", help="list, locate, verify or download models")
    models_sub = p.add_subparsers(dest="models_command", metavar="<action>")

    lp = models_sub.add_parser("list", help="list known aliases and their state")
    _add_common_flags(lp)

    pp = models_sub.add_parser("path", help="print the local path of a model")
    _add_common_flags(pp)
    pp.add_argument("model_target", metavar="ALIAS")

    vp = models_sub.add_parser("verify", help="verify model integrity")
    _add_common_flags(vp)
    vp.add_argument("model_target", metavar="ALIAS")
    vp.add_argument("--full", action="store_true", default=False,
                    help="hash entire weight files")

    up = models_sub.add_parser("pull", help="download a model")
    _add_common_flags(up)
    up.add_argument("model_target", metavar="ALIAS")
    up.add_argument("--source", choices=["hf", "modelscope"], default="hf")
    up.add_argument("--revision", default=None)
    up.add_argument("--resume", action="store_true", default=False)


def _add_languages(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("languages", help="show the three language tables")
    _add_common_flags(p)


def _add_serve(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("serve", help="start qwen-asr-serve for remote clients")
    _add_common_flags(p)
    p.add_argument("model_target", nargs="?", metavar="MODEL", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", dest="gpu_memory_utilization",
                   type=float, default=None)


def _add_doctor(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("doctor", help="check the local runtime and models")
    _add_common_flags(p)
    p.add_argument("--strict", action="store_true", default=False,
                   help="exit 3 when the overall status is warn/fail")


def _add_config(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("config", help="inspect configuration")
    config_sub = p.add_subparsers(dest="config_command", metavar="<action>")
    sp = config_sub.add_parser("show", help="show the merged effective configuration")
    _add_common_flags(sp)
    cp = config_sub.add_parser("path", help="print the active configuration file")
    _add_common_flags(cp)


def build_parser() -> QasrParser:
    parser = QasrParser(
        prog=PROG,
        description="Script-friendly local ASR runtime for Qwen3-ASR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_flags(parser)
    parser.add_argument("--version", action="store_true", default=argparse.SUPPRESS,
                        help="print the version and exit")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _add_transcribe(sub)
    _add_align(sub)
    _add_serve(sub)
    _add_models(sub)
    _add_languages(sub)
    _add_doctor(sub)
    _add_config(sub)
    return parser


def _collect_overrides(ns: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key in CLI_CONFIG_KEYS:
        if hasattr(ns, key):
            overrides[key] = getattr(ns, key)
    return overrides


# Flags that take a value, used to find the ``serve`` token without mistaking
# a flag value for the command.
_GLOBAL_VALUE_FLAGS = frozenset(
    {
        "--config",
        "--model",
        "--aligner",
        "--backend",
        "--device",
        "--dtype",
        "--attn-implementation",
        "--model-dir",
        "--log-level",
        "--server",
    }
)
_SERVE_VALUE_FLAGS = _GLOBAL_VALUE_FLAGS | {
    "--host",
    "--port",
    "--gpu-memory-utilization",
}
_SERVE_BOOL_FLAGS = frozenset(
    {"--json", "--jsonl", "--dry-run", "--quiet", "--verbose", "--offline", "-h", "--help"}
)


def _find_serve_index(argv: Sequence[str]) -> Optional[int]:
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "serve":
            return index
        name = token.split("=", 1)[0]
        index += 2 if name in _GLOBAL_VALUE_FLAGS and "=" not in token else 1
    return None


def _split_serve_tail(tail: Sequence[str]) -> tuple[Optional[str], list[str], list[str]]:
    """Split ``serve`` arguments into model, qasr-known flags, and passthrough.

    Everything unknown is forwarded verbatim to ``qwen-asr-serve``. Once an
    unknown option appears, every following bare token belongs to it rather
    than being mistaken for MODEL.
    """
    model: Optional[str] = None
    known: list[str] = []
    passthrough: list[str] = []
    seen_unknown = False
    index = 0
    while index < len(tail):
        token = tail[index]
        if token == "--":
            passthrough.extend(tail[index + 1:])
            break
        name = token.split("=", 1)[0]
        if name in _SERVE_VALUE_FLAGS:
            known.append(token)
            if "=" not in token:
                if index + 1 < len(tail):
                    known.append(tail[index + 1])
                    index += 2
                    continue
                known.append("")
            index += 1
            continue
        if token in _SERVE_BOOL_FLAGS:
            known.append(token)
            index += 1
            continue
        if token.startswith("-"):
            seen_unknown = True
            passthrough.append(token)
            index += 1
            continue
        if model is None and not seen_unknown:
            model = token
        else:
            passthrough.append(token)
        index += 1
    return model, known, passthrough


def _validate_backend(config: Any, command: str) -> None:
    """Reconcile ``--server`` with ``--backend``; reject an unbound remote."""
    if command == "serve":
        return
    backend = str(config.get("backend") or "auto").strip().lower()
    server = config.get("server")
    if backend == "remote" and not server:
        raise UsageError(
            "--backend remote requires --server URL",
            hint="pass --server URL or set QASR_SERVE_URL",
        )
    if server and backend in ("transformers", "vllm"):
        raise UsageError(
            f"--server conflicts with --backend {backend}",
            hint="drop --backend; --server implies the remote backend",
        )


def _silence_stdout() -> None:
    """Point stdout at /dev/null so the interpreter's exit flush stays quiet."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
    except OSError:
        pass


def _emit_error(command: str, exc: QasrError, *, json_mode: bool, quiet: bool = False) -> int:
    try:
        if json_mode:
            print(json.dumps(error_payload(command, exc.to_dict()), ensure_ascii=False))
            sys.stdout.flush()
        else:
            print(f"qasr: {exc.kind} error: {exc.message}", file=sys.stderr)
            if exc.hint:
                print(f"hint: {exc.hint}", file=sys.stderr)
    except BrokenPipeError:
        _silence_stdout()
        return 0
    return exc.exit_code


def _install_signal_handlers() -> None:
    def _handler(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM,):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    serve_passthrough: list[str] = []
    parse_argv: Sequence[str] = argv
    serve_index = _find_serve_index(argv)
    if serve_index is not None:
        model_token, known_tail, serve_passthrough = _split_serve_tail(argv[serve_index + 1:])
        parse_argv = (
            list(argv[: serve_index + 1])
            + ([model_token] if model_token else [])
            + known_tail
        )

    try:
        ns = parser.parse_args(parse_argv)
    except SystemExit as exc:  # --help
        return int(exc.code or 0)
    except QasrError as exc:
        return _emit_error("qasr", exc, json_mode="--json" in argv and "--jsonl" not in argv)

    if getattr(ns, "version", False):
        try:
            print(f"qasr {__version__}")
            sys.stdout.flush()
        except BrokenPipeError:
            _silence_stdout()
        return 0

    command = getattr(ns, "command", None)
    json_mode = bool(getattr(ns, "json", False))
    jsonl_mode = bool(getattr(ns, "jsonl", False))

    if not command:
        if json_mode:
            return _emit_error(
                "qasr",
                UsageError("no command given", hint="run 'qasr --help' for available commands"),
                json_mode=True,
            )
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    try:
        if json_mode and jsonl_mode:
            raise UsageError("--json and --jsonl are mutually exclusive")
        if jsonl_mode and command != "transcribe":
            raise UsageError(
                f"--jsonl is not supported by '{command}'",
                hint="use --json (JSONL is only produced by transcribe)",
            )

        config = load_config(
            getattr(ns, "config", None),
            cli_overrides=_collect_overrides(ns),
        )
        _validate_backend(config, command)
        ctx = CommandContext(
            config=config,
            json_mode=json_mode,
            jsonl_mode=jsonl_mode,
            quiet=bool(getattr(ns, "quiet", False)),
            verbose=bool(getattr(ns, "verbose", False)),
            dry_run=bool(getattr(ns, "dry_run", False)),
        )

        handler = COMMAND_HANDLERS.get(command)
        if handler is None:
            raise UsageError(f"unknown command: {command}")
        if command == "serve":
            ns.passthrough = serve_passthrough

        _install_signal_handlers()
        code = int(handler(ns, ctx))
        # Flush explicitly: under block buffering (a pipe), a closed consumer
        # only shows up here instead of during the handler's writes.
        sys.stdout.flush()
        return code
    except QasrError as exc:
        return _emit_error(command, exc, json_mode=json_mode,
                           quiet=bool(getattr(ns, "quiet", False)))
    except KeyboardInterrupt:
        print("qasr: interrupted", file=sys.stderr)
        return EXIT_INTERRUPT
    except BrokenPipeError:
        # A consumer like `head` closed the pipe: this is not an error, and we
        # must silence the interpreter's final stdout flush as well.
        _silence_stdout()
        return 0
    except Exception as exc:  # noqa: BLE001 - last-resort error envelope
        from .errors import RuntimeFailure

        return _emit_error(command, RuntimeFailure(f"internal error: {exc}"),
                           json_mode=json_mode)


if __name__ == "__main__":
    sys.exit(main())
