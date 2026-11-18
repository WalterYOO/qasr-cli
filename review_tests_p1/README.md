# P1 independent black-box review suite

Owned by the independent reviewer. These tests must not import `qasr_cli`;
every case drives the CLI as a subprocess and observes stdout/stderr/exit
codes/files/HTTP only.

## Run

Torch-free review environment:

```bash
/tmp/qasr-m1-venv/bin/python -m pytest review_tests_p1 -q -m "not integration"
/tmp/qasr-m1-venv/bin/python -m ruff check review_tests_p1
```

If `qasr` is not installed, the suite runs `python -m qasr_cli` from this
checkout. Override with `QASR_BIN=/path/to/qasr` or
`QASR_PYTHON=/path/to/python`.

Real 0.6B integration (no GPU required for short audio): set
`RUN_INTEGRATION=1`, `QASR_TEST_MODEL_DIR`, `QASR_TEST_AUDIO`, and
`QASR_PYTHON` to an interpreter where the runtime site-packages are
importable, then run:

```bash
/tmp/qasr-m1-venv/bin/python -m pytest review_tests_p1 -q -m integration
```

GPU cases are skipped here; gpu06 evidence is collected by the coordinator.

## Markers

| Marker | Meaning |
|---|---|
| `p1` | all cases (applied automatically) |
| `integration` | needs `RUN_INTEGRATION=1`, real models/audio |
| `gpu` | needs Linux/CUDA/vLLM; evidence comes from gpu06 |
| `slow` | long-running scenario |

## Pieces

- `helpers.py` — stdlib-only JSON/JSONL parsing, WAV/model fixtures, schema
  validation, recursive secret scanning.
- `fake_openai.py` — threaded fake OpenAI/vLLM server. Records requests,
  parses multipart, supports auth, delays, injected responses, response
  sequences, model lists, and `max_active` concurrency accounting.
- `conftest.py` — isolated `QASR_HOME`/cwd per test, subprocess runner,
  fake `qwen-asr-serve`, import blocker, audio fixtures.

## Coverage map

- N1: serve argv/exec/dry-run and `doctor --server`.
- N2: transcription/chat transport, remote language semantics (server LID is
  authoritative), 404 fallback, error/retry/timeout mapping, `--jobs`
  concurrency/order/fail-fast/state integrity.
- N3: explicit vLLM platform gate, auto selection, `gpu-memory-utilization`,
  `--server` conflicts.
- N4: window formula, 180s hard cap, 270s pure-text default, `overlap=0`,
  remote client-side chunking (25MB), dry-run without runtime imports.
- N5: stdin-list, skip-existing, retry, tsv/md, jobs, bench, models rm path
  safety, config get/set/project.
- Security: sentinel tokens must never appear in stdout/stderr/JSON/config/
  state/manifest/jobs files, including 401 body echo, URL userinfo, and
  cross-origin redirects.

## Expectations that encode finalized semantics

- Remote transcription responses carry `language X<asr_text>` and must be
  parsed; plain text yields `language=null` + `language_source="server"`.
- Chat must not send assistant prefill / `continue_final_message` (vLLM 0.14.0
  rejects it); explicit language on chat only warns.
- `--retries 2` means three total attempts.
- 408/429/5xx/timeout/connection errors exit 3; other 4xx exit 4; partial
  failure exits 6; all-failed uses the first input-order error code.
- `align` defaults to windowing in P1; 181s succeeds, <=180s is single-window,
  `overlap=0` is valid.
- Remote long audio is chunked client-side; every upload stays below 25MB.
- `--stdin-list` takes no value and reads stdin; `-` inside it is not a path.
- Any `-f` output format is mutually exclusive with `--json`/`--jsonl`.
- `jobs list --limit` must be 1–10000.

## Not covered here

- GPU/vLLM performance or RTF gates.
- Real 1h audio end-to-end (coordinator supplies gpu06 evidence).
- Any assertion based on importing implementation internals.
