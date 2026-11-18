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
qasr models verify 0.6b
```

Aliases: `1.7b` (default), `0.6b`, `aligner`.

## Quickstart

```bash
qasr doctor
qasr transcribe a.wav --language zh
qasr transcribe ./audio --recursive -f txt,json,srt --timestamps --language zh -o out/
qasr transcribe --manifest jobs.jsonl --out-dir out --resume
qasr align a.wav --text-file transcript.txt --language zh -f srt,json -o out/
qasr languages --json
```

## Known limitations (P0)

- Forced timestamps require an explicit alignable language (11 languages);
  `--language auto --timestamps` is rejected.
- `qasr align` handles a single window of at most 180 s.
- Dialect labels are LID output only and cannot be forced.
- English words may be glued across internal chunk boundaries (library-level).
- Half-downloaded models are rejected by `qasr models verify`.

See `DESIGN.md` for the full contract and `P0-TASKS.md` for the milestone plan.
