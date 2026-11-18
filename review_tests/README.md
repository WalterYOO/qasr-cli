# qasr P0 black-box review tests

These tests are intentionally independent of the implementation:

- every case invokes `qasr` as a subprocess;
- nothing imports `qasr_cli`;
- assertions come from `DESIGN.md` v0.3 and `P0-TASKS.md`, not from reading the code.

The suite is meant to be run by the independent reviewer before looking at the
implementation. A failure is evidence for the review report, not necessarily a
test bug.

## Requirements

- Python 3.10–3.12
- `pytest`
- a runnable `qasr` (source checkout or installed wheel)

No model download is performed by the non-integration tests. URL cases use
`example.invalid` and assert that `--dry-run` does not download.

## Running

From the repository root:

```bash
# Pure CPU / no-torch contract tests
uv run pytest review_tests -m "not integration" -q

# Model-backed acceptance tests
RUN_INTEGRATION=1 \
QASR_TEST_MODEL_DIR=/Users/walter/workspace/QwenLM/Qwen3-ASR/models \
QASR_TEST_AUDIO=/tmp/qasr_asr_zh.wav \
uv run pytest review_tests -m integration -q
```

If the checkout has no `.venv` yet, point the runner at an existing interpreter:

```bash
QASR_PYTHON=/path/to/python \
  /path/to/python -m pytest review_tests -m "not integration" -q
```

## Runner selection

The suite resolves the command in this order:

1. `QASR_BIN=/path/to/qasr`
2. `QASR_PYTHON=/path/to/python` + `-m qasr_cli` (when `qasr_cli/` exists)
3. `python -m qasr_cli` from the current interpreter
4. `qasr` on `PATH`

Set `QASR_USE_INSTALLED=1` to prefer an installed `qasr` even when `qasr_cli/`
exists in the checkout.

Each test gets:

- a fresh `QASR_HOME` under pytest's `tmp_path`;
- a fresh working directory (so `./models` and `../Qwen3-ASR/models` fallbacks
  cannot leak into the result);
- all inherited `QASR_*` variables removed.

## Environment variables

| Variable | Meaning |
|---|---|
| `QASR_BIN` | explicit `qasr` executable |
| `QASR_PYTHON` | interpreter used for `-m qasr_cli` |
| `QASR_USE_INSTALLED` | prefer an installed console script |
| `RUN_INTEGRATION` | must be `1` for model-backed tests |
| `QASR_TEST_MODEL_DIR` | directory containing `Qwen3-ASR-0.6B` and `Qwen3-ForcedAligner-0.6B` |
| `QASR_TEST_AUDIO` | short Chinese WAV (default `/tmp/qasr_asr_zh.wav`) |
| `QASR_TEST_AUDIO_TEXT` | optional expected transcript substring |
| `QASR_TEST_ALIGN_TEXT` | optional text to align (defaults to the 13-character fixture sentence) |
| `QASR_TEST_DEVICE` | optional `--device` override for integration tests |

## Markers

- `integration`: requires `RUN_INTEGRATION=1`, models, and audio.
- `slow`: kill -9 / SIGINT / lock scenarios.

## Model fixtures

Half-pack and corrupt-model cases are **synthetic**. The tests build a minimal
HF-style directory with:

- `config.json`, tokenizer metadata, `model.safetensors.index.json`;
- a safetensors shard with a valid `8-byte header length + JSON header + data`;
- optional `.aria2` marker, missing indexed shard, or truncated header.

This avoids depending on whether the real 1.7B directory happens to be complete
on the review machine. A test named `complete` expects the synthetic directory
to pass; `aria2` / `missing_shard` / `truncated` expect `incomplete` plus a
`pull --resume` hint.

## Contract assumptions worth knowing

These are places where the design leaves shape or ordering implicit. The tests
encode the reading listed in `REVIEW-PLAN.md` §8:

- model directories live under `QASR_HOME/models/<repo-basename>`
  (`Qwen3-ASR-1.7B`, `Qwen3-ASR-0.6B`);
- `--dry-run` does not download URLs or load models;
- `align` rejects >180s before resolving/loading the model;
- a dead-PID lock is stale and must be recovered;
- `models verify` exits non-zero for `incomplete`/`missing`;
- all-failed batch runs use the first item's error code (not 6);
- `--json`/`--jsonl` are accepted after the subcommand as well as before it.

If the implementation takes a different but defensible reading, report it as a
spec ambiguity first; do not silently weaken the assertion.

## What is not covered here

- disk-full / read-only filesystem behaviour (manual Phase 2 check);
- true byte-level SRT/VTT golden files (structure is checked instead);
- vLLM, remote, streaming, mic, and long-audio windowing (P1/P2).
