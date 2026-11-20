---
name: qasr-cli
description: >-
  Operate the local `qasr` CLI (qasr-cli 0.3.x) for Qwen3-ASR speech
  recognition and forced alignment: transcribe files, directories, globs,
  URLs and JSONL manifests to txt/json/jsonl/srt/vtt/tsv/md; word, char or
  sentence timestamps; align a known transcript or re-time an SRT; context
  biasing and language/dialect ID; long-audio chunking; resumable batch jobs
  with `jobs` inspection; remote OpenAI-compatible servers; local vLLM;
  streaming from mic/stdin/file on Linux+CUDA; model pull/verify/rm;
  doctor/config/bench/completion; SFT JSONL lint. Use for qasr, Qwen3-ASR,
  Qwen3 ForcedAligner, 语音转写, 音频转文字, 听写, 转录, 字幕, SRT, VTT,
  强制对齐, 时间戳, 热词, 长音频, 批量转写, 断点续跑, 流式转写, 远程 ASR,
  vLLM ASR. Not for TTS, voice cloning, dubbing, translation, speaker diarization, training or ffmpeg-style media conversion.
---

# qasr-cli：Qwen3-ASR 转写 / 对齐 / 字幕

按 `qasr 0.3.x`（0.3.4 实测）编写。先 `qasr --version`；主/次版本不同时，以 `qasr <cmd> --help` 为准。本 skill 只处理“音频 → 文本/时间轴”；TTS、音色克隆、配音、有声书归 `qtts-cli`，翻译、说话人分离、烧字幕、视频剪辑、训练都不在 qasr 范围。

## 安装与最小路径

```bash
uv tool install --python 3.12 "qasr-cli[local]"       # 本机 transformers（转写/对齐）
uv tool install --python 3.12 "qasr-cli[vllm,mic]"    # 仅 Linux+CUDA：serve/stream
uv tool install --python 3.12 qasr-cli                # 远程客户端/运维（无本地推理）
```

Python 需 `>=3.10,<3.13`；宿主默认 3.13+ 时必须显式 `--python 3.12`。仓库内未装进 PATH 时用 `.venv/bin/qasr` 代替下文所有 `qasr`。

```bash
qasr --version
qasr --json doctor | jq '{status, bad: [.checks[] | select(.status != "ok") | {name, status, hint}]}'
qasr --json models list | jq '.models[] | {alias, status, present}'
qasr --json transcribe a.wav -l auto                 # 最小路径：stdout 一个 JSON envelope
qasr --json transcribe a.wav -l zh --timestamps -f srt,vtt -o out/   # 字幕（需完整模型+aligner）
```
`doctor` 不加 `--strict` 时即使 `status:"fail"` 也退出 0；必须读 `.status` / `.checks[]`。
本地推理缺 `torch` / `qwen-asr` 时先装 `[local]`，不要靠 `models pull` 解决。

## 先读三条（硬规则）

1. **联网、下载、外发由用户决定**：`models pull`、`--server`、URL 输入都会联网。先说明来源、体积、
   目标端，再由用户确认；`transcribe`/`align` 自身不会自动下载模型。
2. **覆盖与删除先确认**：`transcribe` 目标已存在会以 4 拒绝；`align` **静默覆盖且没有 `--overwrite`**；
   `models rm`/`config set` 立即生效：前者先 `models show` + `rm --dry-run`，后者先 `config path`，**没有可用 dry-run**（`--dry-run` 被接受但忽略）。
3. **token 只在用户自己的交互 shell**：用户执行 `read -rs QASR_SERVE_TOKEN; export QASR_SERVE_TOKEN`。
   agent 不读取、不打印、不代填，不写命令行前缀赋值、URL userinfo、config、JSON 或日志。

## 心智模型

- 输入（文件/目录/glob/URL/manifest/stdin）→ 后端（本地 transformers、本地 vLLM、remote）→
  文本与时间戳 → stdout 或 `-o` 文件。进度与人类日志只在 stderr。
- 本地 `--jobs` 恒被压为 1，串行；吞吐用 `--batch-size`（vLLM 默认 8，transformers 默认 1）。
  远程 `--jobs 1..32`（默认 4）才并行。
- 时间戳 = ASR + 强制对齐器；必须显式给 11 种可对齐语言（`zh en yue fr de it ja ko pt ru es`），
  写 `-l auto` 会以 4 拒绝（`align` 除外，它的 `-l auto` 合法）。22 个中文方言标签只能由 LID 输出，不能 `-l` 强制；`yue` 例外。
- 机器消费一律 `--json`（`stream` 例外，用 `--emit jsonl`）。`--json` 与 `-f` 的冲突见下。
- `--dry-run` 只证明参数解析与计划，不证明模型存在；体检用 `doctor` / `models verify`。

## 退出码

| code | 含义 | agent 动作 |
|---|---|---|
| 0 | 成功 | 仍读 `.ok`；`doctor` 可能 `status:"fail"` |
| 2 | 用法/配置/manifest schema | 修参数；不重试 |
| 3 | 模型/运行时/后端/下载 | `doctor --json`；查模型完整性、torch、vLLM、网络 |
| 4 | 输入/音频/路径/能力不支持 | 换输入或后端；远程要时间戳属此类 |
| 5 | `align` 覆盖率低于阈值；或 `stream` 达时长上限 | `align` 未写任何文件；读 error envelope |
| 6 | 批量部分失败 | 读 `.errors[]`；仅 `retryable:true` 才可 `--retry` |
| 130 | SIGINT/SIGTERM | 先看 `.qasr/state.jsonl`，再 `--resume` |

`--fail-fast` 或全部失败时取首个错误的码，而不是 6。

## 任务路由

| 意图 | 命令（`qasr` 前缀省略） | 检查 / 下一步 |
|---|---|---|
| 单文件 | `--json transcribe a.wav -l auto` | `.ok` `.result.text` `.result.language_source`；空文本看 `duration_s` |
| 强制语言 | `--json transcribe a.wav -l zh` | `.result.language_source=="forced"`；方言标签不可强制 |
| 粤语 | `--json transcribe a.wav -l yue` | `yue` 可强制、可对齐 |
| 热词 | `--json transcribe a.wav --context "Qwen3" --context-file terms.txt` | 词表每行一个、`#` 注释；多次 `--context` 按空格合并 |
| 字幕 | `--json transcribe a.wav -l zh --timestamps -f srt,vtt -o out/` | `.result.segments[]`；4=语言不可对齐或 chunk>180 |
| 已知文稿 | `--json align a.wav --text-file script.txt -l zh -f srt -o out/` | 顶层 `.language`、`.result.coverage`；先 dry-run（静默覆盖） |
| 重定时 SRT | `--json align a.wav --srt draft.srt -l en -f srt -o out/` | `.result.items[]` 单调；align 必须给语言（`auto` 可） |
| 目录/glob | `--json transcribe ./audio --recursive -f txt,srt -l zh --timestamps -o out/ --resume` | `.summary.*`、`.errors[]`；6 看 `retryable` |
| stdin 清单 | `--json transcribe --stdin-list -o out/ < paths.txt` | 多输入无 `--json`/`--jsonl` 时必须 `--out-dir` |
| 直接 URL | `--json transcribe https://example.com/a.wav -o out/` | 会联网下载；失败 3/4，先确认来源可信 |
| 异构 manifest | `--json transcribe --manifest jobs.jsonl -o out/ --max-chunk-seconds 180 --resume` | schema 字段错 2；timestamp 项缺 `language` 是 4 |
| 续跑 | 原命令加 `--resume`（必要时 `--retry`） | `.summary.skipped` 上升；单文件没有 `.summary`，看 `.skipped` |
| 长音频 | `transcribe long.wav -l zh --max-chunk-seconds 270` | 时间戳固定 ≤180；OOM 降 chunk 或换 0.6b |
| 远程文本 | `--json --server http://HOST:PORT transcribe a.wav --jobs 4` | `.model.backend=="remote"`；时间戳/align/stream 均 4 |
| serve 门禁 | `--json models show 1.7b` → `--json serve 1.7b --dry-run` | `.status=="complete"`；dry-run `.source=="local"` 且 `.path` 非 null，`repo` 会下载 |
| serve 启动 | `tmux new-session -d -s qasr-serve 'qasr serve 1.7b --host 127.0.0.1 --port 18000'` | 仅 Linux+CUDA+`[vllm]`；无内建鉴权；详见 `deploy-and-remote.md` |
| 本地 vLLM | `--json --backend vllm transcribe a.wav --batch-size 8` | `.model.backend=="vllm"`；非 Linux/CUDA → 3 |
| 流式 | `qasr stream --input a.wav --emit jsonl` | `seq` 从 0 连续、`is_final`；仅 Linux+CUDA+vLLM |
| 运维 | `--json doctor` / `models *` / `config *` / `jobs *` / `bench` / `completion` | 见 `references/ops-and-models.md` |
| 微调数据 | `--json finetune lint-jsonl train.jsonl --strict` | 只校验，不训练；exit 4=有错或文件不可读 |

## 后端决策

```text
有 --server 或 QASR_SERVE_URL? → remote
  ├─ 文本 / context / 批量 → --jobs 1..32（默认 4）
  ├─ 要时间戳 / align / stream → 不支持，exit 4
  └─ --max-chunk-seconds >600 → 降为 600 + warning
否则本地
  ├─ 要 stream → 必须 Linux + CUDA + [vllm]，否则 4
  ├─ backend=auto：Linux+CUDA 且 vllm 可导入 → vllm；否则 transformers（含 macOS/MPS/CPU）
  ├─ 显式 vllm 但平台不支持 → transcribe 3；stream 4；align 忽略 vllm、始终 transformers（remote 仍 4）
  └─ dtype=auto：CUDA→bf16，MPS→fp16，CPU→fp32
```

| 能力 | transformers | vllm | remote |
|---|---|---|---|
| 文本 / LID / context | ✓ | ✓ | ✓（context 走 chat，`-l` 不强制） |
| 时间戳 / `align` | ✓ | ✓ 均走 transformers（`--backend` 对 align 无效） | ✗ exit 4 |
| 多文件 | ✓ 串行（jobs=1） | ✓ 串行（jobs=1） | ✓ jobs 1–32 |
| `stream` | ✗ 4 | ✓ 仅 Linux/CUDA | ✗ 4 |
| `--batch-size` | 默认 1 | 默认 8 + `--max-inference-batch-size` | 恒 1 |
| `--gpu-memory-utilization` | ✗ 2 | ✓ (0,1] | ✗ 2 |

显式 `--device cuda:N` 会在 import torch 前设置 `CUDA_VISIBLE_DEVICES`；`--device mps` 配 vLLM 会失败。
`auto` 在 Linux+CUDA 上探测到 vLLM 但导入失败时回退 transformers，JSON 顶层出现 `backend_fallback:true`；
显式 `--backend vllm` 不回退，直接 3。`--server` 与显式 `--backend transformers|vllm` 冲突（2）。

## 输出与破坏性

- `--json`：stdout 恰好一个对象；单输入是 item envelope，多输入是 batch envelope（`.summary`/`.items`/`.errors`）。
- `--jsonl`：仅 `transcribe` 支持，逐条 item 一行，**没有 summary**；`done:` 只在写文件模式的人类 stderr 出现。
- `transcribe` 的 `--json` 与 `-f json/jsonl/tsv/md` 互斥（2）；`-f txt/srt/vtt` 可同用。
  `align` 的 `-f` 可与 `--json` 同用，同样支持 7 种（help 只列 4 种），多格式需 `-o`；全局 `--jsonl` 仍 2。
- `-f srt/vtt` 必须同时给 `--timestamps` 与可对齐语言，否则 4。无 `-o` 且无 `--json/--jsonl` 时多格式是 2；
  `--json` 下多格式合法，**无 `-o`** 只输出 envelope、不落盘，给 `-o` 仍写多个文件（已存在则 4）；`--output` 只接受单输入单格式。
- `transcribe` 目标已存在默认 **拒绝**（4）；用户同意后才 `--overwrite`，批量优先 `--resume` / `--skip-existing`。
- `align` **静默覆盖**旧文件且没有 `--overwrite`：先 `--dry-run` 看 `.plan.outputs[].path`，确认或换新目录。
- 批量 state 在 `<out-dir>/.qasr/state.jsonl`，另有 `manifest.json`、`lock`；不要手工改 `.qasr/`。
- 读 stdout envelope，不要读 sidecar `out/*.json` 的 `.outputs`（先写文件后组装，可能为空/缺失）。

## 硬性禁止

- 长任务（>~10 分钟）与 `serve` 只用 tmux；禁止 `nohup` 与裸 `&`。
- 不把 token 放进 argv、config、state、日志、URL userinfo；不替用户 `read`/`export`。
- 不自动 `models pull` / `models rm`；不把 `serve` 绑到 `0.0.0.0`（除非用户明确要求且有防火墙/反代/TLS）。
- 不向远程求 `--timestamps` / `align` / `stream`；不把 `--gpu-memory-utilization` 给非 vLLM。
- 不把 `doctor` 的 exit 0 当健康；不把 `models list` 的 `present:true` 当完整（看 `status`）。
- 不把 `bench` 当准确率评测；不写 `-f markdown`（用 `md`）；没有 `models cache` 子命令。
- 不声称支持 VAD、说话人分离、翻译、烧字幕、视频抽轨、训练（只有 `finetune lint-jsonl`）。

## 自检

```bash
qasr --json transcribe a.wav | jq -e '.ok == true and .command == "transcribe"'
qasr --json transcribe ./audio -o out/ --resume | jq -e '.summary.total == (.summary.ok + .summary.failed + .summary.skipped)'
qasr --json transcribe a.wav -l zh --timestamps | jq -e '(.result.segments|length) > 0 and ([.result.segments[].start] as $s | $s == ($s|sort)) and all(.result.segments[]; .end >= .start)'
qasr --json align a.wav --text-file t.txt -l zh --coverage-min 0.8 | jq -e '.ok == true and .result.coverage >= 0.8'
qasr --json transcribe a.wav -l zh --timestamps -f srt -o out/ --dry-run | jq -e '.plan.outputs[0].path | endswith(".srt")'
```

其他判据（`schema_version==1`、`.summary.skipped`、`jobs` 落定、`stream.seq`）见 `troubleshooting.md`。

## 参考索引

- `references/commands-core.md` — transcribe/align 全参数、格式矩阵、JSON 字段、字幕断句（7s/42 列/700ms）。
- `references/batch-and-resume.md` — 目录/stdin/URL/manifest、稳定排序去重、job key、state、锁、并发。
- `references/long-audio-and-stream.md` — 270/180/600 阈值、align 窗口、stream 快照与回滚。
- `references/deploy-and-remote.md` — serve 前置与安全、远程能力边界、vLLM、GPU、鉴权。
- `references/ops-and-models.md` — 模型解析/pull/verify/rm、config 层级、doctor、bench、completion、finetune。
- `references/safety.md` — 覆盖/删除/下载/外发/长任务/token 判定表。
- `references/troubleshooting.md` — 按退出码与 `error.kind` 分流、常见假成功。
