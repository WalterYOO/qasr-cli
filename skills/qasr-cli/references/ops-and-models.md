何时读我：要安装/定位/校验/下载/删除模型，改配置，读 doctor，跑 bench/completion，或校验 SFT JSONL 时。

# 运维：模型、配置、诊断与周边

版本基线 `qasr 0.3.2`。

## 模型解析

`--model`（或 config/env 的 `model`）接受：

- alias：`1.7b`、`0.6b`（ASR），以及 `aligner`（仅供 `--aligner`）；
- 本地目录（绝对/相对/`~`）；
- HF repo id（如 `Qwen/Qwen3-ASR-0.6B`）或 repo 目录名。

解析顺序：显式路径存在 → 在 `models_root` 找 alias/repo 目录 → 回退为 repo id（在线语义）。
搜索根由 `--model-dir` / `QASR_MODEL_DIR` 改写，默认 `QASR_HOME/models`。
`--aligner auto|none|PATH` 单独解析；`none` 表示不使用 aligner（需要时间戳时会 3）。

默认模型 `1.7b`。`transcribe`/`align`/`stream` **不会**自动下载；缺模型是 3。
`--dry-run` 不校验模型存在（`--model /nope` 也会 `ok:true`），体检用 `models verify`。

## models

```bash
qasr --json models list
qasr --json models show 1.7b
qasr --json models verify 1.7b
qasr --json models verify 1.7b --full          # 哈希整个权重文件
qasr --json models path 1.7b
qasr --json models rm 1.7b --dry-run           # 仅受管目录且模型存在时；无需 --yes
qasr --json models pull 0.6b --source hf --resume
```

- `.status`：`complete | incomplete | missing`（list/show/verify）；只有 `complete` 可用于推理/serve。
  只有 `rm` 的 payload 是 `.status=="present"`（不是 complete/incomplete/missing）。
  `.present` 不等于完整，必须看 `.status` 或 `models verify`；`models show <不存在>` 仍 rc0/`ok:true`。
- 半包特征：`.aria2`、`.incomplete`、`.part`、`.tmp`、`.download`；safetensors 头/分片也会校验。
- `.fingerprint`（`sha256:...`）可用于跨机核对；它也是 resume job key 的一部分。
- `pull --source hf`（默认）需要 `[hf]`；`--source modelscope` 需要 `[modelscope]`。
  `--revision` 固定版本；`--resume` 续传；`--offline` 会直接拒绝下载。
- `pull` 会下载数 GB 权重并联网：先说明来源、revision、体积与磁盘占用，由用户确认。
- `rm` 只删 `models_root` 下的**一级受管目录**（`[managed]`）；符号链接、嵌套目录、`..` 都会 2。
  路径存在但不在 `models_root`（如 dev path）→ 2；模型不存在 → 3。非交互必须 `--yes`，否则 2；
  先 `show` + `rm --dry-run` 再确认；`rm` 的 `.status=="present"`。
- 没有 `models cache` 子命令；缓存/模型落点用 `models path` 与 `QASR_HOME` 查看。

## config

```bash
qasr --json config path
qasr --json config get backend
qasr --json config show
qasr --json config show | jq '.config | to_entries[] | select(.value.source != "default")'
qasr config set backend vllm --project        # 先给用户看落点
```

- 优先级（低→高）：内置默认 < `QASR_HOME/config.json` < 项目 `.qasr/config.json` <
  环境变量 < `$QASR_CONFIG` < `--config` < CLI。
- `config show` 的 `.config` 是 `{key: {value, source}}`；`.files` 给出各层路径与是否存在。
- `config set` **立即写盘**；`--dry-run` 会被接受但完全忽略，仍立即写盘，不要用它预演。
  `--home` 是默认落点，`--project` 写到最近的 `.git`/`pyproject.toml` 根下的 `.qasr/config.json`。两者互斥（2）。
- 可用键：`model aligner backend device dtype attn_implementation language context format
  batch_size max_inference_batch_size jobs model_dir offline server timeout retries
  window_seconds window_overlap max_chunk_seconds`。
- token 类键会被拒绝（2）：token 只放环境变量 `QASR_SERVE_TOKEN`。
- 在 macOS 上把 `backend` 写成 `vllm` 会让后续 transcribe 直接 3、stream 4；`align` 始终走
  transformers，不受该 backend 影响。改配置前确认平台。
- `config show` 会脱敏 `server` URL 的 userinfo/敏感 query；仍不要把 token 放进 URL。

相关环境变量：`QASR_HOME`、`QASR_MODEL_DIR`、`QASR_MODEL`、`QASR_BACKEND`、`QASR_DEVICE`、
`QASR_DTYPE`、`QASR_OFFLINE`、`QASR_SERVE_URL`、`QASR_LANGUAGE`、`QASR_CONFIG`、`QASR_SERVE_TOKEN`。

## doctor

```bash
qasr --json doctor
qasr --json doctor --strict
qasr --json doctor | jq '{status, bad: [.checks[] | select(.status != "ok") | {name, status, detail, hint}]}'
```

- 字段：`.status`（`ok|warn|fail`）与 `.checks[]` 的 `name/status/value/detail/hint`。
- 典型 checks：`python qasr platform torch qwen-asr transformers vllm backend device memory
  qasr-home disk soundfile ffmpeg model:1.7b model:0.6b model:aligner`。
- **默认退出 0，即使 `status:"fail"`**；`--strict` 在 warn/fail 时退出 3。
- 按 `hint` 修复；doctor 不会自动 `models pull`，也不会改配置。
- `torch`/`qwen-asr` fail = 缺 `[local]`；`device` warn = 无 torch 或 MPS 不可用；
  `model:*` warn/fail 先看是否 `complete`。

## languages

```bash
qasr --json languages
qasr --json languages | jq '{alignable, forceable: [.forceable[].iso], dialects: (.dialects|length)}'
```

- `forceable` 30 种可强制语言；`alignable` 11 种可对齐语言；
  `dialects` 22 个仅 LID 的中文方言标签；`lid_labels` 是模型输出标签全集。
- 方言标签不能用于 `-l` 强制（4）；`yue` 在 forceable 与 alignable 中都有。

## bench

```bash
qasr --json bench --duration 30 --iterations 3
qasr --json bench a.wav --iterations 5
qasr --json bench --duration 5 --dry-run
```

- 无 INPUT 时用合成音频，`--duration` 默认 10 s；`--iterations` 默认 3。
- 字段在顶层：`rtf load_s audio_s elapsed_s max_new_tokens iterations peak_memory_mb`；**没有 `.metrics`**。
- 只测吞吐/RTF，不承诺准确率；远程 bench 把网络时间计入 RTF（带 warning）。
- `max_new_tokens` 按最长输入时长自动取（`min(16384, max(1024, dur*8+256))`），
  避免 qwen-asr 的 512 默认值把长音频生成提前截断、导致 RTF 只统计了半段工作量。
- `--dry-run` 打印计划，不加载权重。

## completion

```bash
qasr completion zsh | head
qasr completion bash > ~/.qasr-completion.bash
qasr completion fish > ~/.config/fish/completions/qasr.fish
```

- 只打印脚本到 stdout，不落盘；写 home 由用户决定。
- 支持 `bash|zsh|fish`；其他名字 exit 2。

## finetune lint-jsonl

```bash
qasr --json finetune lint-jsonl train.jsonl
qasr --json finetune lint-jsonl train.jsonl --strict --no-check-files
```

- 只校验 Qwen3-ASR SFT JSONL，不导入 torch/datasets、不解码音频、不训练。
- 允许字段：`audio`、`text`、`prompt`；`text` 必须是非空字符串，`audio` 必须是可读路径。
- 若 `text` 形如 `language <label><asr_text>...`，label 必须在 LID 标签内，`<asr_text>` 后不能为空。
- `--strict` 把未知字段、重复 audio 等 warning 升级为 error；`--no-check-files` 跳过文件存在性。
- 有任何 error 或文件不可读 → 4；JSON payload 的 `.valid`/`.errors[]`/`.warnings[]` 给出明细。

## 版本与维护

```bash
qasr --version
qasr --help >/dev/null
for c in transcribe align serve models languages doctor config jobs bench completion finetune; do
  qasr "$c" --help >/dev/null || echo "BROKEN: $c"
done
qasr --json languages | jq -e '.alignable | length == 11'
```

发版时核对：命令/子命令增删、flag 改名、默认值、extra 依赖、平台限制、`schema_version`。
本 skill 与 CLI 同 tag 维护；新 flag 先进本目录，再改 `SKILL.md` 路由表。
