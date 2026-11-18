# qasr-cli

Script-friendly local ASR runtime for Qwen3-ASR: transcription, language ID,
context biasing, timestamps, forced alignment, batch jobs with resumable state.

> P1 adds `serve`, an OpenAI-compatible remote client, vLLM, long-audio
> windowing, concurrent remote jobs and batch operations. Streaming,
> microphone input, completion and finetuning remain P2.

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
qasr models show 0.6b --json
qasr models rm 1.7b --yes          # only managed dirs under QASR_HOME/models
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

# 5. Force-align a known transcript (windows above 180 s automatically).
qasr align a.wav --text-file transcript.txt --language zh -f srt,json -o out/

# 6. Transcribe against a remote OpenAI-compatible server.
QASR_SERVE_TOKEN=... qasr --server http://127.0.0.1:18000 transcribe a.wav --jobs 4

# 7. Batch from stdin, skip finished outputs, retry transient failures.
qasr transcribe --stdin-list --skip-existing --retry -o out/ < paths.txt

# 8. Throughput without accuracy claims.
qasr bench --iterations 3 --json
```

### Remote server and vLLM

`qasr serve` launches `qwen-asr-serve`; any flags it does not know are passed
through. `--server URL` (or `QASR_SERVE_URL`) switches transcription to the
remote OpenAI-compatible API. Remote runs support plain text, `context`,
`--jobs` concurrency and `--max-chunk-seconds`; timestamps, granularity and
`align` are rejected with exit 4. The token is read only from
`QASR_SERVE_TOKEN` and is never written to config, JSON or state. Without an
explicit `--model`, the first id from `/v1/models` is used and reported.

`--backend vllm` needs Linux + CUDA and the `qasr-cli[vllm]` extra.
`--backend auto` picks vLLM on a CUDA host when it imports, otherwise
transformers. Explicit `--device cuda:N` (including `cuda:0`) overrides
`CUDA_VISIBLE_DEVICES` before torch is imported.

### Alignment JSON

`qasr align --json` reports `language` and `language_source`
(`explicit` / `script` / `lid_probe`) at the **top level** of the payload, not
inside `result`; `result` holds `items`, `coverage` and `warnings`. In
`transcribe` payloads the language fields stay inside `result`.

### Long audio

Audio longer than `--max-chunk-seconds` (default 270 for plain text, 180 with
`--timestamps`) is split into contiguous core windows with a 3 s lookback.
Aligners never receive more than 180 s. Timestamped long audio keeps the
canonical full-file ASR text and attaches per-window alignments; plain text
concatenates window transcripts with a `chunk_boundary_text_uncertain`
warning. Per-window checkpoints live in `<out-dir>/.qasr/chunks/` and are
reused by `--resume`. `qasr align` windows audio above `--window-seconds`
(10–180, default 180); `--window-overlap` is 0–10 s and defaults to 3.

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
`--json/--jsonl`. `tsv` writes
`id<TAB>start<TAB>end<TAB>text<TAB>language` rows (tabs/newlines in fields are
collapsed to spaces); `md` writes one `## <id>` section per input.

| code | meaning |
|---|---|
| 0 | success |
| 2 | usage, configuration or manifest schema error |
| 3 | model, runtime, backend or download failure |
| 4 | input, audio, path or unsupported capability |
| 5 | alignment coverage / subtitle quality gate |
| 6 | partial batch failure (some items succeeded; fail-fast reports the first error code instead) |
| 130 | interrupted (SIGINT/SIGTERM) |

## Configuration

Priority: defaults < `QASR_HOME/config.json` < project `.qasr/config.json`
(searched upward) < environment variables < `$QASR_CONFIG` < `--config` file
< CLI flags. Keys: `model`, `aligner`, `backend`, `device`, `dtype`,
`attn_implementation`, `language`, `context`, `format`, `batch_size`,
`max_inference_batch_size`, `jobs`, `model_dir`, `offline`, `server`,
`timeout`, `retries`, `window_seconds`, `window_overlap`, `max_chunk_seconds`.

```bash
qasr config get backend --json
qasr config set backend transformers            # QASR_HOME/config.json
qasr config set window_seconds 120 --project    # <project>/.qasr/config.json
```

`config set` is atomic, preserves unknown keys, validates enums and refuses
symlinked targets. Tokens are not configuration keys.

### Jobs and bench

Batch runs with `--out-dir` register under `QASR_HOME/jobs/`:

```bash
qasr jobs list --limit 20
qasr jobs show <JOB_ID|OUT_DIR> --json
```

`qasr bench [INPUT...]` reports `load_s`, `audio_s`, `elapsed_s`, `rtf`,
`iterations` and `peak_memory_mb`; with no input it uses synthetic silence.
Remote bench includes network time in RTF.

Environment: `QASR_HOME`, `QASR_MODEL`, `QASR_MODEL_DIR`, `QASR_BACKEND`,
`QASR_DEVICE`, `QASR_DTYPE`, `QASR_OFFLINE`, `QASR_LANGUAGE`, `QASR_CONFIG`.

## Known limitations

- Forced timestamps require an explicit alignable language (11 languages);
  `--language auto --timestamps` is rejected.
- Dialect labels are LID output only and cannot be forced.
- English words may be glued across plain-text chunk boundaries; listen for
  the `chunk_boundary_text_uncertain` warning.
- Timestamped long audio re-runs the canonical full-file ASR on every resume;
  window anchors and alignments are cached but the canonical pass is not.
- Half-downloaded models are rejected by `qasr models verify`.
- Local inference runs one job at a time; `--jobs` controls remote concurrency
  only. Use `--batch-size` for local micro-batching.
- Zero-sample audio succeeds with empty text and `language: null`.
- `--jsonl` is only meaningful for `transcribe`; other commands reject it.
- Remote timestamps/alignment and streaming are out of scope.

See `DESIGN.md` for the full contract and `P0-TASKS.md` for the milestone plan.
