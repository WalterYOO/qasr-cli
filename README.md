# qasr-cli

Script-friendly local ASR runtime for Qwen3-ASR: transcription, language ID,
context biasing, timestamps, forced alignment, batch jobs with resumable state.

> P0 scope: single-machine core loop. `serve`, remote clients, vLLM and
> streaming are P1/P2 and are intentionally absent.

## Install

```bash
uv tool install --python 3.12 "qasr-cli[local]"
```

Python `>=3.10,<3.13` is required. Without `[local]` the CLI still provides
model management, `languages`, `config` and `doctor` (remote-only install).

## Models

Models live under `QASR_HOME/models` (macOS: `~/Library/Application Support/qasr`).

```bash
qasr models list
qasr models pull 0.6b --source hf
qasr models pull 0.6b --resume          # continue a partial download
qasr models pull 0.6b --source modelscope
qasr models verify 0.6b
```

Aliases: `1.7b` (default), `0.6b`, `aligner`. `--resume` is forwarded to the
hub downloader (legacy hubs that lack the option simply resume by default).
After a successful pull, `models/<name>/source.json` records the repo,
revision, source and per-file sizes. `--source modelscope` needs the optional
`qasr-cli[modelscope]` extra.

## Quickstart

```bash
# 1. Diagnose the environment and models.
qasr doctor

# 2. Transcribe one file to stdout.
qasr transcribe a.wav --language zh

# 3. Batch a directory: txt + json + SRT under out/.
qasr transcribe ./audio --recursive -f txt,json,srt --timestamps --language zh -o out/

# 4. Machine-readable output (one JSON line per input).
qasr transcribe a.wav b.wav --jsonl

# 5. Force-align a known transcript.
qasr align a.wav --text-file transcript.txt --language zh -f srt,json -o out/
```

### Batch manifest

`--manifest FILE` accepts JSONL records and always requires `--out-dir`:

```json
{"id": "ep01", "audio": "audio/ep01.wav", "language": "zh", "formats": ["txt", "json"]}
{"id": "ep02", "audio": "https://example.com/ep02.wav", "formats": ["srt"], "timestamps": true}
```

Allowed fields: `id`, `audio`, `audio_base64`, `language`, `context`, `formats`,
`output`, `timestamps`, `granularity`. Unknown fields, duplicate ids and unsafe
output paths are rejected. Re-running with `--resume` skips jobs whose request
key and output hashes are unchanged.

## Output and exit codes

`--json` prints one object (an item, a batch envelope, or an error envelope);
`--jsonl` prints one item per line. Progress and third-party logs go to stderr.
`--format json/jsonl` are file formats and are mutually exclusive with
`--json/--jsonl`.

| code | meaning |
|---|---|
| 0 | success |
| 2 | usage, configuration or manifest schema error |
| 3 | model, runtime, backend or download failure |
| 4 | input, audio, path or unsupported capability |
| 5 | alignment coverage / subtitle quality gate |
| 6 | partial batch failure (some items succeeded) |
| 130 | interrupted (SIGINT/SIGTERM) |

## Configuration

Priority: defaults < `QASR_HOME/config.json` < project `.qasr/config.json`
(searched upward) < environment variables < `$QASR_CONFIG` < `--config` file
< CLI flags. Keys: `model`, `aligner`, `backend`, `device`, `dtype`,
`attn_implementation`, `language`, `context`, `format`, `batch_size`,
`max_inference_batch_size`, `model_dir`, `offline`. `jobs` and `server` are
P1 keys: a non-default value (`jobs > 1` or any `server`) is rejected with a
usage error instead of being silently ignored.

Environment: `QASR_HOME`, `QASR_MODEL`, `QASR_MODEL_DIR`, `QASR_BACKEND`,
`QASR_DEVICE`, `QASR_DTYPE`, `QASR_OFFLINE`, `QASR_LANGUAGE`, `QASR_CONFIG`.

## Known limitations (P0)

- Forced timestamps require an explicit alignable language (11 languages);
  `--language auto --timestamps` is rejected.
- `qasr align` handles a single window of at most 180 s.
- Dialect labels are LID output only and cannot be forced.
- English words may be glued across internal chunk boundaries (library-level).
- Half-downloaded models are rejected by `qasr models verify`.
- Local inference runs one job at a time; use `--batch-size` for
  micro-batching. `--jobs > 1` only warns, and `jobs > 1` in a config file is
  rejected (parallel jobs arrive in P1).
- Zero-sample audio succeeds with empty text and `language: null`.
- `--jsonl` is only meaningful for `transcribe`; other commands reject it.

See `DESIGN.md` for the full contract and `P0-TASKS.md` for the milestone plan.
