# P2 independent review suite

Black-box tests for `qasr stream`, `qasr completion` and
`qasr finetune lint-jsonl`. Every test invokes `python -m qasr_cli` in a
subprocess; nothing imports `qasr_cli` internals.

## Running

```bash
# usage / capability / completion / finetune / schema (no GPU)
pytest -q review_tests_p2 -m "not gpu"

# real vLLM on a GPU host (gpu06); requires RUN_GPU=1
RUN_GPU=1 QASR_TEST_MODEL_DIR=/models QASR_TEST_AUDIO=/tmp/zh.wav \
  pytest -q review_tests_p2 -m gpu
```

`QASR_PYTHON` selects the interpreter used for `-m qasr_cli`
(default: the interpreter running pytest). `QASR_BIN` overrides it with an
installed console script.

## Fixtures

- `fake_stream` — temp `PYTHONPATH` with fake `torch`, `vllm`, `qwen_asr`
  plus a `sitecustomize` that reports Linux. The fake streaming model mirrors
  qwen-asr's buffering API and logs `model_init` / `init` / `feed` / `chunk` /
  `finish` events to JSONL. Failure injection:
  `QASR_FAKE_STREAM_FAIL=init|chunk:N|finish`, `QASR_FAKE_STREAM_SLEEP`,
  `QASR_FAKE_STREAM_BANNER`, `QASR_FAKE_STREAM_LANGUAGE`.
- `fake_mic` — fake `sounddevice` fed from `QASR_FAKE_MIC_PCM`; logs open
  parameters; `QASR_FAKE_MIC_FAIL=open` injects a PortAudio failure.
- `import_blocker` — modules that raise on import; used to prove `--help`,
  `--version`, `completion`, `finetune` and `--dry-run` do not import heavy
  runtimes.

Tests marked `simulated` use the fake Linux/vLLM harness for local state
machine and PCM coverage. They are **not** gpu06 evidence; the `gpu` tests are
the real-backend acceptance path.

## Contract references

`P2-TASKS.md` §7 (`522a130`) resolved Q1–Q16. Test docstrings cite the
relevant Q number where the assertion depends on a ruling.
