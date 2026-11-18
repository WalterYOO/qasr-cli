import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from qasr_cli import transcribe as core
from qasr_cli.backends import Segment, TranscribeResult
from qasr_cli.cli import main
from tests.conftest import write_wav

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_schema(name: str):
    return json.loads((SCHEMA_DIR / f"{name}.json").read_text(encoding="utf-8"))


def validate(name: str, payload: dict) -> None:
    jsonschema.validate(payload, load_schema(name))


def run_json(capsys, argv):
    code = main(argv)
    out = capsys.readouterr()
    return code, out


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "qasr 0.2.0"


def test_help(capsys):
    assert main(["--help"]) == 0
    assert "transcribe" in capsys.readouterr().out


def test_languages_json(capsys):
    code = main(["languages", "--json"])
    out = capsys.readouterr()
    assert code == 0
    payload = json.loads(out.out)
    validate("languages", payload)
    assert out.err == ""


def test_config_show_json(capsys):
    code = main(["config", "show", "--json"])
    out = capsys.readouterr()
    assert code == 0
    validate("config-show", json.loads(out.out))
    assert out.err == ""


def test_config_path_json(capsys):
    code = main(["config", "path", "--json"])
    out = capsys.readouterr()
    assert code == 0
    validate("config-path", json.loads(out.out))


def test_config_path_text(capsys):
    assert main(["config", "path"]) == 0
    assert capsys.readouterr().out.strip().endswith("config.json")


def test_models_list_json(capsys, model_factory):
    model_factory("Qwen3-ASR-0.6B")
    code = main(["models", "list", "--json"])
    out = capsys.readouterr()
    assert code == 0
    payload = json.loads(out.out)
    validate("models-list", payload)
    by_alias = {entry["alias"]: entry for entry in payload["models"]}
    assert by_alias["0.6b"]["status"] == "complete"


def test_models_verify_complete(capsys, model_factory):
    model_factory("Qwen3-ASR-0.6B")
    code = main(["models", "verify", "0.6b", "--json"])
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert code == 0
    validate("models-verify", payload)
    assert payload["status"] == "complete"


def test_models_verify_incomplete(capsys, model_factory):
    model_factory("Qwen3-ASR-1.7B", sharded=True, truncate=True)
    code = main(["models", "verify", "1.7b", "--json"])
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert code == 3
    validate("models-verify", payload)
    assert payload["status"] == "incomplete"
    assert "pull 1.7b --resume" in payload["hint"]


def test_models_verify_missing_error_envelope(capsys):
    code = main(["models", "verify", "0.6b", "--json"])
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert code == 3
    assert payload["status"] == "missing"


def test_models_path(capsys, model_factory):
    model_factory("Qwen3-ASR-0.6B")
    assert main(["models", "path", "0.6b"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("Qwen3-ASR-0.6B")


def test_models_path_missing_is_error(capsys):
    code = main(["models", "path", "0.6b", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 3
    validate("error", payload)


def test_doctor_json(capsys):
    code = main(["doctor", "--json"])
    out = capsys.readouterr()
    assert code == 0
    validate("doctor", json.loads(out.out))


def test_doctor_strict_can_fail(capsys):
    main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    code = main(["doctor", "--json", "--strict"])
    assert code == (0 if payload["status"] == "ok" else 3)


def test_usage_error_json(capsys):
    code = main(["--json", "--jsonl", "languages"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    validate("error", payload)
    assert payload["error"]["kind"] == "usage"


@pytest.mark.parametrize(
    "argv",
    [
        ["--device", "tpu"],
        ["--dtype", "int8"],
        ["--backend", "remote"],
        ["--attn-implementation", "bogus"],
        ["--device", "cuda:abc"],
    ],
)
def test_invalid_cli_config_value_is_usage_error(capsys, argv):
    code = main([*argv, "--json", "config", "show"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    validate("error", payload)
    assert payload["error"]["kind"] in {"usage", "config"}


@pytest.mark.parametrize(
    "env_name, value",
    [("QASR_DEVICE", "tpu"), ("QASR_DTYPE", "int8"), ("QASR_OFFLINE", "garbage")],
)
def test_invalid_env_config_value_is_usage_error(capsys, monkeypatch, env_name, value):
    monkeypatch.setenv(env_name, value)
    code = main(["--json", "config", "show"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    validate("error", payload)


def test_valid_config_values_are_normalized(capsys, monkeypatch):
    monkeypatch.setenv("QASR_OFFLINE", "off")
    monkeypatch.setenv("QASR_DEVICE", "CUDA:1")
    code = main(["--json", "config", "show"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["config"]["device"]["value"] == "cuda:1"
    assert payload["config"]["offline"]["value"] is False


def test_json_without_command_emits_error_envelope(capsys):
    code = main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    validate("error", payload)
    assert payload["command"] == "qasr"


def test_jsonl_rejected_outside_transcribe(capsys):
    code = main(["languages", "--jsonl"])
    out = capsys.readouterr()
    assert code == 2
    assert "jsonl" in out.err.lower()


def test_missing_input_reports_input_error_before_model(capsys, tmp_path):
    code = main(["transcribe", "--json", str(tmp_path / "nope.wav")])
    payload = json.loads(capsys.readouterr().out)
    assert code == 4
    validate("error", payload)
    assert payload["error"]["kind"] == "input"


def test_empty_audio_succeeds_without_loading_model(capsys, tmp_path):
    empty = write_wav(tmp_path / "empty.wav", seconds=0)
    code = main(["transcribe", str(empty), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    validate("transcribe-item", payload)
    assert payload["result"]["text"] == ""
    assert payload["result"]["language"] is None


def test_bad_audio_is_input_error_before_model(capsys, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    code = main(["transcribe", str(bad), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 4
    validate("error", payload)
    assert payload["error"]["kind"] == "input"


def test_missing_and_valid_input_runs_valid_one(capsys, tmp_path, wav_file,
                                              model_factory, fake_backend):
    model_factory("Qwen3-ASR-0.6B")
    fake_backend()
    code = main(["transcribe", str(wav_file), str(tmp_path / "nope.wav"),
                 "--model", "0.6b", "--device", "cpu", "--dtype", "fp32", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 6
    validate("transcribe-batch", payload)
    assert payload["summary"]["ok"] == 1
    assert payload["summary"]["failed"] == 1


def test_unknown_command(capsys):
    code = main(["nope"])
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_missing_input_usage_error(capsys):
    code = main(["transcribe"])
    assert code == 2


def test_bad_language_is_input_error(capsys, wav_file, model_factory):
    model_factory("Qwen3-ASR-0.6B")
    code = main(["transcribe", str(wav_file), "--language", "klingon", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 4
    validate("error", payload)


def test_timestamps_require_explicit_language(capsys, wav_file, model_factory):
    model_factory("Qwen3-ASR-0.6B")
    code = main(["transcribe", str(wav_file), "--timestamps", "--language", "auto", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 4
    assert "explicit" in payload["error"]["message"]


def test_multi_input_without_out_dir_is_usage_error(capsys, tmp_path, wav_file, model_factory):
    model_factory("Qwen3-ASR-0.6B")
    other = write_wav(tmp_path / "other.wav")
    code = main(["transcribe", str(wav_file), str(other)])
    assert code == 2


def test_format_json_with_json_flag_is_usage_error(capsys, wav_file, model_factory):
    model_factory("Qwen3-ASR-0.6B")
    code = main(["transcribe", str(wav_file), "--format", "json", "--json"])
    assert code == 2


def test_dry_run_plan(capsys, wav_file, model_factory):
    model_factory("Qwen3-ASR-0.6B")
    code = main(["transcribe", str(wav_file), "--model", "0.6b", "--device", "cpu", "--dtype", "fp32", "--dry-run", "--json"])
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert code == 0
    validate("dry-run", payload)
    assert payload["plan"]["inputs"][0]["duration_s"] > 0


# ---------------------------------------------------------------------------
# Fake-backend end-to-end contract
# ---------------------------------------------------------------------------

class FakeBackend:
    name = "fake"

    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)

    def ensure_loaded(self, need_align=False):
        return None

    def transcribe(self, jobs, opts):
        results = []
        for job in jobs:
            if job.item.id in self.fail_ids:
                raise RuntimeError("synthetic")
            results.append(
                TranscribeResult(
                    text="甚至出现交易几乎停滞的情况。",
                    language=job.language or "Chinese",
                    language_source="forced" if job.language else "auto",
                    segments=[Segment("甚", 0.0, 0.3, "char"),
                              Segment("至", 0.3, 0.6, "char")],
                    timestamp_source="forced_aligner" if opts.timestamps else None,
                )
            )
        return results


@pytest.fixture
def fake_backend(monkeypatch):
    def _install(fail_ids=()):
        backend = FakeBackend(fail_ids=fail_ids)
        monkeypatch.setattr(core, "build_transformers_backend", lambda *a, **k: backend)
        return backend

    return _install


def test_transcribe_single_json_schema(capsys, wav_file, model_factory, fake_backend):
    model_factory("Qwen3-ASR-0.6B")
    fake_backend()
    code = main(["transcribe", str(wav_file), "--model", "0.6b", "--device", "cpu", "--dtype", "fp32",
                 "--language", "zh", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    validate("transcribe-item", payload)
    assert payload["result"]["language"] == "Chinese"
    assert payload["result"]["language_source"] == "forced"


def test_transcribe_batch_jsonl(capsys, tmp_path, wav_file, model_factory, fake_backend):
    model_factory("Qwen3-ASR-0.6B")
    fake_backend()
    second = write_wav(tmp_path / "second.wav")
    code = main(["transcribe", str(wav_file), str(second), "--model", "0.6b", "--device", "cpu", "--dtype", "fp32", "--jsonl"])
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert code == 0
    assert len(lines) == 2
    for line in lines:
        validate("transcribe-item", json.loads(line))


def test_transcribe_partial_failure_exit_6(capsys, tmp_path, wav_file, model_factory, fake_backend):
    model_factory("Qwen3-ASR-0.6B")
    fake_backend(fail_ids={"clip"})
    second = write_wav(tmp_path / "second.wav")
    code = main(["transcribe", str(wav_file), str(second), "--model", "0.6b", "--device", "cpu", "--dtype", "fp32", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 6
    validate("transcribe-batch", payload)
    assert payload["summary"]["failed"] == 1


def test_transcribe_writes_srt_with_timestamps(capsys, wav_file, model_factory,
                                               fake_backend, monkeypatch):
    model_factory("Qwen3-ASR-0.6B")
    model_factory("Qwen3-ForcedAligner-0.6B")
    fake_backend()
    out_dir = wav_file.parent / "out"
    code = main(["transcribe", str(wav_file), "--model", "0.6b", "--device", "cpu", "--dtype", "fp32",
                 "--language", "zh", "--timestamps", "-f", "srt", "-o", str(out_dir)])
    assert code == 0
    srt = (out_dir / "clip.srt").read_text()
    assert "00:00:00,000 --> 00:00:00,600" in srt
    assert "甚至" in srt


def test_stdout_is_clean_json(capsys, model_factory, fake_backend, wav_file):
    model_factory("Qwen3-ASR-0.6B")
    fake_backend()
    main(["transcribe", str(wav_file), "--model", "0.6b", "--device", "cpu", "--dtype", "fp32", "--json"])
    out = capsys.readouterr()
    json.loads(out.out)
    assert "model loaded" in out.err  # progress belongs on stderr


def test_subprocess_help_and_json_purity(isolated_home):
    env = dict(os.environ)
    env["QASR_HOME"] = str(isolated_home)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    help_run = subprocess.run([sys.executable, "-m", "qasr_cli", "--help"],
                              capture_output=True, text=True, env=env)
    assert help_run.returncode == 0
    assert "transcribe" in help_run.stdout

    run = subprocess.run([sys.executable, "-m", "qasr_cli", "languages", "--json"],
                         capture_output=True, text=True, env=env)
    assert run.returncode == 0
    json.loads(run.stdout)


def test_help_does_not_import_torch():
    code = (
        "import sys\n"
        "import qasr_cli.cli, qasr_cli.transcribe, qasr_cli.align, qasr_cli.doctor\n"
        "print(int('torch' in sys.modules), int('qwen_asr' in sys.modules), "
        "int('transformers' in sys.modules))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "0 0 0"
