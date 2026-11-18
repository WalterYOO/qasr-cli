"""Regressions for the Codex phase-2 review findings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from qasr_cli.cli import main
from tests.conftest import write_wav

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _done_job_key(out_dir: Path) -> str:
    state = out_dir / ".qasr" / "state.jsonl"
    records = [json.loads(line) for line in state.read_text().splitlines() if line.strip()]
    done = [record for record in records if record.get("status") == "done"]
    assert done
    return done[-1]["job_key"]


def test_single_input_multiple_formats_to_stdout_is_usage_error(capsys, wav_file):
    code = main(["transcribe", str(wav_file), "-f", "txt,json"])
    assert code == 2
    capsys.readouterr()


def test_models_without_action_is_usage_error(capsys):
    code = main(["models"])
    assert code == 2
    capsys.readouterr()


def test_p1_config_keys_are_rejected(capsys, tmp_path):
    jobs = _write_config(tmp_path, {"jobs": 2})
    assert main(["languages", "--config", str(jobs)]) == 2
    capsys.readouterr()

    server = _write_config(tmp_path, {"server": "http://localhost:8080"})
    assert main(["languages", "--config", str(server)]) == 2
    capsys.readouterr()


def test_config_format_controls_output_extension(capsys, tmp_path):
    empty = write_wav(tmp_path / "empty.wav", seconds=0)
    config = _write_config(tmp_path, {"format": ["json"]})
    out = tmp_path / "out"

    code = main(["transcribe", str(empty), "-o", str(out), "--config", str(config)])
    assert code == 0
    capsys.readouterr()
    assert sorted(path.name for path in out.glob("empty.*")) == ["empty.json"]


def test_config_context_participates_in_job_key(capsys, tmp_path):
    empty = write_wav(tmp_path / "empty.wav", seconds=0)
    config = _write_config(tmp_path, {"context": ["hotword"]})
    from_config = tmp_path / "from-config"
    from_cli = tmp_path / "from-cli"

    assert main(["transcribe", str(empty), "-o", str(from_config),
                 "--config", str(config)]) == 0
    capsys.readouterr()
    assert main(["transcribe", str(empty), "-o", str(from_cli),
                 "--context", "hotword"]) == 0
    capsys.readouterr()

    assert _done_job_key(from_config) == _done_job_key(from_cli)


def test_empty_audio_with_timestamps_needs_no_aligner(capsys, tmp_path):
    empty = write_wav(tmp_path / "empty.wav", seconds=0)
    code = main(["transcribe", str(empty), "--timestamps", "--language", "zh", "--json"])
    out = capsys.readouterr()
    assert code == 0, out.err
    payload = json.loads(out.out)
    assert payload["result"]["text"] == ""
    assert payload["result"]["language"] is None


def test_align_dry_run_does_not_require_local_aligner(capsys, wav_file):
    code = main(["align", str(wav_file), "--text", "你好。", "--language", "zh",
                 "--dry-run", "--json"])
    out = capsys.readouterr()
    assert code == 0, out.err
    payload = json.loads(out.out)
    assert payload["command"] == "align"


def test_broken_pipe_exits_zero_without_traceback():
    # Default (block) buffering: small outputs never hit the pipe until the
    # explicit flush at the end of main(), which is what the exit flush used
    # to surface as rc=120 plus "Exception ignored ..." noise.
    script = """
import os, sys
from qasr_cli.cli import main
read_fd, write_fd = os.pipe()
os.close(read_fd)
sys.stdout = os.fdopen(write_fd, "w")
raise SystemExit(main(["languages", "--json"]))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "BrokenPipeError" not in proc.stderr
    assert "Traceback" not in proc.stderr


def test_manifest_ids_that_sanitize_alike_are_rejected(capsys, tmp_path):
    a = write_wav(tmp_path / "one.wav", seconds=0)
    b = write_wav(tmp_path / "two.wav", seconds=0)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps({"id": item_id, "audio": str(audio)})
                  for item_id, audio in (("a/b", a), ("a_b", b))) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"

    code = main(["transcribe", "--manifest", str(manifest), "-o", str(out),
                 "--overwrite", "--json"])
    captured = capsys.readouterr()
    assert code == 2, captured.err
    payload = json.loads(captured.out)
    details = payload["error"]["details"]
    assert details["ids"] == ["a/b", "a_b"]
    assert "a_b.txt" in payload["error"]["message"]


def test_directory_mirroring_disambiguates_instead_of_overwriting(capsys, tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    write_wav(root / "a b.wav", seconds=0)
    write_wav(root / "a_b.wav", seconds=0)
    out = tmp_path / "out"

    code = main(["transcribe", str(root), "-o", str(out), "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert sorted(path.name for path in out.glob("a_b*.txt")) == ["a_b.txt", "a_b_2.txt"]


def test_manifest_output_names_that_sanitize_alike_are_rejected(capsys, tmp_path):
    a = write_wav(tmp_path / "one.wav", seconds=0)
    b = write_wav(tmp_path / "two.wav", seconds=0)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"id": item_id, "audio": str(audio), "output": output})
            for item_id, audio, output in (
                ("first", a, "a b"),
                ("second", b, "a_b"),
            )
        ) + "\n",
        encoding="utf-8",
    )

    code = main(["transcribe", "--manifest", str(manifest), "-o", str(tmp_path / "out"),
                 "--json"])
    captured = capsys.readouterr()
    assert code == 2, captured.err
    assert "a_b.txt" in captured.out


def test_config_context_merge_deduplicates_in_order(capsys, tmp_path):
    empty = write_wav(tmp_path / "empty.wav", seconds=0)
    config = _write_config(tmp_path, {"context": ["hotword", "acme"]})

    def _run(name, *extra):
        out = tmp_path / name
        code = main(["transcribe", str(empty), "-o", str(out),
                     "--config", str(config), *extra])
        assert code == 0
        capsys.readouterr()
        return _done_job_key(out)

    config_only = _run("config-only")
    same_twice = _run("same-twice", "--context", "hotword", "--context", "acme")
    new_term = _run("new-term", "--context", "hotword", "--context", "new")

    assert config_only == same_twice
    assert config_only != new_term
