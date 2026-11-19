# QASR-SKILL-DESIGN

**状态**：设计稿 v2（Partner A），基于 Partner B review `7a3ec220…b06f` 修订
**事实基线**：`qasr 0.3.0` / tag `b945bb4` / 2026-11-19
**范围**：只设计 Claude Code skill；不修改 `qasr_cli/`，不实现 SKILL.md

---

## 0. 一句话结论

这个 skill 不是"qasr 命令手册"，而是**把用户意图翻译成可验证命令序列的路由器**：默认 `--json` + `doctor`/`--dry-run` 先确认环境与计划；它必须显式写出 qasr 的四条反直觉边界——**远程无时间戳/对齐/流式**、**时间戳需 11 种可对齐语言之一**、**`transcribe` 拒绝覆盖但 `align` 静默覆盖**、**`serve` 可能回退 repo id 并下载且 `--offline` 拦不住**——否则 agent 会稳定踩坑。

---

## 1. skill 定位与触发

### 1.1 三层分工

| 层 | 负责 | 不负责 |
|---|---|---|
| `qwen-asr` 官方 CLI / Python | 模型、`qwen-asr-serve`、SFT 训练 | 输入归一、批量状态、断点、稳定 JSON |
| `qtts align` | 给**自己合成的音频**产字幕（原文即真值） | 任意音频、任意已知文本、独立重定时 |
| `qasr` | 任意音视频 → 文本/时间轴/批量/远程/流式 | TTS、音色、混音、训练 |
| 本 skill | 选对命令、参数、校验字段、失败下一步 | 替代 `--help`、编造未实现能力 |

**与 qtts-cli skill 的边界**：TTS 生产链（配音、克隆、有声书、dub）留在 qtts；只有"已有音频、要转写/对齐/批量"才进入 qasr。两边都有"字幕"，所以 description 必须带 `qasr`、`Qwen3-ASR` 强锚点。

### 1.2 何时触发 / 不触发

| 触发 | 例子 |
|---|---|
| 明确点名 | "用 qasr 转写这个 wav"、"Qwen3-ASR 跑一下" |
| 本机 ASR | 语音转文字、会议录音整理、播客逐字稿、听写 |
| 字幕/时间轴 | 生成 SRT/VTT、给已知文稿打时间轴、重定时现有 SRT |
| 批量/长音频 | 目录转写、断点续跑、1 小时录音、热词偏置 |
| 部署形态 | `qasr serve`、远程 ASR、vLLM 加速、流式麦克风 |
| 运维 | `doctor` 报错、模型半包、`config`、`bench`、`jobs` |

| 不触发 | 应去 |
|---|---|
| 纯 TTS / 音色克隆 / 有声书 / 视频配音 | `qtts-cli` skill |
| 只说“给视频加字幕” | 先问：已有文稿用 `align`，需要转写用 `transcribe`；翻译/烧录交给别的工具 |
| 字幕翻译、音频翻译、说话人分离 | 翻译 / diarization 工具；qasr 只做转写与时间轴 |
| 烧字幕、视频剪辑、视频抽轨 | ffmpeg / NLE；qasr 只产出 SRT/VTT 等文本 |
| 通用音频编辑、降噪、混音、格式转换 | ffmpeg / 专门工具 |
| 训练或微调 ASR 模型 | 官方 `finetuning/`；qasr 只做 `finetune lint-jsonl` |
| 阿里云 DashScope / 第三方云 ASR | qasr 无该 backend，不要硬套 `--server` |
| 只说"处理一下音频"且无转写语义 | 先问清楚，不触发 |

### 1.3 frontmatter 候选

**推荐版**：

```yaml
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
  vLLM ASR. Not for TTS, voice cloning, dubbing or generic audio editing.
---
```

**短版（与 qtts 冲突严重时）**：

```yaml
description: >-
  Use the `qasr` CLI for Qwen3-ASR transcription, SRT/VTT subtitles, forced
  alignment, long audio, resumable batch jobs, remote/vLLM serving and
  streaming. Triggers: qasr, Qwen3-ASR, 语音转写, 听写, 字幕, 强制对齐,
  批量转写. Not for TTS or voice cloning (use qtts-cli).
```

取舍：

- `name: qasr-cli` 与 `qtts-cli` 同构，避免与命令名混淆。
- description 先出现 `qasr` 命令词（最强检索锚点），中文词只补充。
- 不加裸词 "audio"/"subtitle"/"transcription"，避免吸走 ffmpeg/qtts 任务。
- 明确 "Not for TTS"：同族 skill 的负向边界比正向堆词更有效。
- 格式 token 写 `md`，不写 `markdown`（CLI 只认 `md`）。

---

## 2. 文件布局、安装与渐进披露

### 2.1 安装矩阵（首屏必须给）

```bash
uv tool install --python 3.12 "qasr-cli[local]"      # P0：本地 transformers
uv tool install --python 3.12 "qasr-cli[vllm,mic]"   # Linux+CUDA：serve/stream
uv tool install --python 3.12 qasr-cli               # 远程客户端/运维
uv tool install --python 3.12 --from . "qasr-cli[local]"   # 开发期
```

Python `>=3.10,<3.13`；宿主默认 Python 可能是 3.13+，必须显式 `--python 3.12`。
仓库内优先探测 `qasr`，没有则用 `.venv/bin/qasr`（本项目开发环境的 fallback）；两者都没有才安装。

| extra | 提供 | 边界 |
|---|---|---|
| 无 | 远程客户端、models/config/doctor/completion/finetune | 不能本地推理 |
| `local` | `qwen-asr==0.0.6` + `torch>=2.4` | transformers 后端 |
| `vllm` | `qwen-asr[vllm]==0.0.6` | 仅 Linux+CUDA |
| `mic` | `sounddevice` | 仅 `stream --input mic` |
| `hf` | `huggingface_hub` | `models pull --source hf` |
| `modelscope` | `modelscope` | `models pull --source modelscope` |
| `all` | local + mic + hf + modelscope | **不含 vllm** |

### 2.2 目录与渐进披露

```
skills/qasr-cli/                # 仓库内源，随 CLI 版本走
├── SKILL.md                    # ≤ 170 行，80% 任务只读这一层
└── references/
    ├── commands-core.md        # transcribe / align / 输出契约
    ├── batch-and-resume.md     # 目录、stdin、URL、manifest、resume、jobs
    ├── long-audio-and-stream.md# 切块、时间戳、stream 三种输入
    ├── deploy-and-remote.md    # serve、远程客户端、vLLM、GPU、鉴权
    ├── ops-and-models.md       # models/languages/config/doctor/bench/completion/finetune
    ├── safety.md               # 覆盖、删除、token、下载、长任务
    └── troubleshooting.md      # 退出码分流、doctor 读法、能力矩阵
```

比 qtts 多一个 `safety.md`：qasr 的 `align` 会**静默覆盖**、`serve` 可能**未经确认下载**，值得独立成文。

安装（Claude Code 需要发现入口）：

```bash
mkdir -p ~/.agents/skills ~/.claude/skills
ln -sfn "$PWD/skills/qasr-cli" ~/.agents/skills/qasr-cli
ln -sfn ../../.agents/skills/qasr-cli ~/.claude/skills/qasr-cli
# 项目级：
mkdir -p .agents/skills .claude/skills
ln -sfn "$PWD/skills/qasr-cli" .agents/skills/qasr-cli
ln -sfn ../../.agents/skills/qasr-cli .claude/skills/qasr-cli
```

`~/.agents/skills` 是实体目标；`~/.claude/skills` 才是 Claude Code 的发现路径（与 `qtts-cli` 同构）。

### 2.3 SKILL.md 提纲（目标 160 行）

| 节 | 写什么 | 行数 |
|---|---|---|
| frontmatter | 上文 description | 8 |
| 安装 + 最小路径 | `--version` → `doctor --json` → 单文件 | 18 |
| 先读三条 | ① 下载/外发由用户决定 ② 覆盖/删除要确认 ③ token 只在用户 shell | 8 |
| 心智模型 | 输入 → 引擎 → 结果；本地串行/远程并行；时间戳=ASR+aligner | 18 |
| 退出码 | 0/2/3/4/5/6/130 + agent 动作 | 12 |
| 任务路由 | §3 的 26 行表 | 35 |
| 后端决策 | §5 的树 + 能力矩阵 | 16 |
| 输出与破坏性 | stdout 纪律、transcribe/align 覆盖差异、state 目录 | 20 |
| 硬性禁止 | §9 的压缩版 | 14 |
| 自检 | §8 的 5 条 jq | 10 |
| 参考索引 | 每个 ref 一行 | 8 |

### 2.4 references 职责

| 文件 | 职责 | 不写 |
|---|---|---|
| `commands-core.md` | transcribe/align 全参数、格式矩阵、JSON 字段、字幕断句（7s/42 列/700ms、毫秒 half-up） | 批量编排、部署 |
| `batch-and-resume.md` | 输入展开与稳定排序/去重、stdin-list、URL、manifest、job key、state、锁、并发 | 单文件参数 |
| `long-audio-and-stream.md` | 270/180/600 阈值、align `--window-seconds`/overlap、stream 快照与回滚、无 tail 模式 | serve 部署 |
| `deploy-and-remote.md` | serve 前置、远程能力边界、vLLM、GPU、鉴权 | 转写语义 |
| `ops-and-models.md` | 模型解析优先级、pull/verify/rm、config 层级、doctor 字段 | 批量策略 |
| `safety.md` | 覆盖/删除/下载/外发/长任务/token 判定表 | 命令语法复述 |
| `troubleshooting.md` | 按退出码与 `error.kind` 分流、常见假成功 | 教程式背景 |

披露规则：SKILL.md 能独立完成"单文件 + 字幕 + 读错误"；batch/stream/serve/模型损坏才读 ref；每个 ref ≤ 220 行，首行写"何时读我"。

---

## 3. 任务路由表

约定：全局 flag 放 `qasr` 之后、子命令之前；命令 flag 放子命令之后。

- **除 `stream` 外**，机器消费一律 `--json`；`stream` 只能用 `--emit jsonl`。
- `--json` 与 `-f json/jsonl/tsv/md` 冲突（exit 2）；与 `-f txt/srt/vtt` 合法。
- 多输入默认需要 `--out-dir`；`--json`/`--jsonl` 可直接走 stdout（`commands/transcribe.py:299-302`）。
- `--dry-run` 支持 transcribe/align/serve/bench（除 serve 外默认打印 JSON plan）；`stream` 不支持（exit 2）。

| 用户意图 | 推荐命令序列 | 要检查的 JSON | 失败下一步 |
|---|---|---|---|
| 安装 | `uv tool install --python 3.12 "qasr-cli[local]"`；仓库内可先用 `.venv/bin/qasr` | `qasr --version` | 本地推理缺 extra → `3` |
| 首次体检 | `qasr --json doctor` → `models list` | `.status`、`.checks[]`、`.models[].status` | 按 `checks[].hint`；不自动 pull |
| 单文件转写 | `qasr --json transcribe a.wav -l auto` | `.ok`、`.result.text`、`.result.language_source` | 空文本看 `duration_s`/格式；`4` 查输入 |
| 强制语言 | `... -l zh` | `.result.language_source=="forced"` | 22 个方言标签不可强制 |
| 粤语/方言 | `qasr --json transcribe a.wav -l yue` | `.result.language_source=="forced"` | `yue` 可强制可对齐；其他方言只 `auto` |
| 热词/上下文 | `... --context "Qwen3" --context "通义"` 或 `--context-file terms.txt` | `.result.text` 是否命中 | 词表每行一个、`#` 注释；多次 `--context` 按空格合并 |
| 出字幕 | `... -l zh --timestamps -f srt,vtt -o out/` | `.result.segments[]`、stdout `.outputs[]` | `4` 多为语言不可对齐或 chunk>180 |
| 已知文稿对齐 | `qasr --json align a.wav --text-file t.txt -l zh -f srt -o out/` | 顶层 `.language`、`.result.coverage` | **align 静默覆盖**；先 dry-run |
| 重定时 SRT | `qasr --json align a.wav --srt draft.srt -l en -f srt -o out/` | `.result.items[]` 单调性 | `align` 必须给 `--language`（`auto` 可） |
| 目录/glob 批量 | `qasr --json transcribe ./audio --recursive -f txt,srt -l zh --timestamps -o out/ --resume` | `.summary.*`、`.errors[]` | `6` 读 `.error.retryable` |
| stdin 清单 | `qasr --json transcribe --stdin-list -o out/ < paths.txt` | 同上 | 多输入无 `--json` 时必须 `--out-dir` |
| 直接 URL | `qasr --json transcribe https://example.com/a.wav -o out/` | `.input.source`、`.result.text` | 网络失败 `3`；先确认来源可信且会联网（ingress 下载，不外发本地数据） |
| 异构 manifest | `... --manifest jobs.jsonl -o out/ --max-chunk-seconds 180 --resume` | `.summary.skipped`、`.errors[]` | manifest 字段错 `2`；timestamps 项必须有 `language` |
| 批量续跑 | 原命令加 `--resume`（必要时 `--retry`） | `.summary.skipped` 上升 | 单文件无 `.summary`，看 `.skipped` |
| 长音频 | `... --max-chunk-seconds 270` / 时间戳固定 `180` | `.result.warnings[]` | 显存不足降 chunk 或换 0.6b |
| 远程文本 | `qasr --json --server URL transcribe a.wav --jobs 4` | `.model.backend=="remote"` | 连接失败 `3`；时间戳 `4` |
| 远程 + context | 同上加 `--context` | `.result.warnings[]` | 见 `language_not_enforced_on_chat` / `server_ignored_language` |
| 本地 vLLM | `qasr --json --backend vllm transcribe a.wav --batch-size 8` | `.model.backend=="vllm"` | 非 Linux/CUDA `3`（stream `4`） |
| serve 前置 | `models show/verify` → `serve --dry-run` | `models show` 的 `.status=="complete"`；`serve --dry-run` 的 `.source=="local"` | `source:"repo"` 会下载，先问用户 |
| 起服务 | `qasr serve 1.7b --host 127.0.0.1 --port 18000` | tmux 内看启动日志 | 仅 Linux+CUDA+`[vllm]`；serve 无鉴权；`--dry-run` 可跨平台但不证明能启动 |
| 流式 | `qasr stream --input a.wav --emit jsonl` | `seq` 连续、`is_final` | 仅 Linux+CUDA+vLLM；`--json` 会 `2` |
| 性能基线 | `qasr --json bench --duration 30 --iterations 3` | `rtf/load_s/audio_s/elapsed_s/peak_memory_mb` | 只报吞吐，不承诺准确率 |
| 模型运维 | `models list/verify/show/path/pull/rm` | `.status`、`.fingerprint` | 半包用 `pull --resume` |
| 配置 | `config path/get/set/show` | `.source` 判断生效层 | `config set` 立即写盘，无 dry-run |
| 任务查看 | `qasr --json jobs list --limit 20` → `jobs show ID` | `.counts`、`.first_error` | 目录被删则 `state:"missing"` |
| Shell 补全 | `qasr completion zsh`（bash/fish 同） | stdout 非空 | shell 名写错 `2` |
| 微调数据 | `qasr --json finetune lint-jsonl train.jsonl --strict` | `.valid`、`.errors[]` | 只校验，不训练 |

---

## 4. 黄金路径（可直接复制）

标注：`[dry-run]` 先用计划验证；`[model]` 需本地模型；`[linux-cuda]` 仅 Linux+CUDA；`[remote]` 需可达服务；`[network]` 会联网。本机 macOS 默认未装 torch，先按 4.1 安装。

### 4.1 安装

```bash
uv tool install --python 3.12 "qasr-cli[local]"     # [install]
qasr --version
```

仓库内若未把工具装进 PATH，用 `.venv/bin/qasr` 代替；两者都没有才安装。

### 4.2 首次体检

```bash
qasr --json doctor | jq '{status, bad: [.checks[] | select(.status != "ok") | {name, status, hint}]}'
qasr --json models list | jq '.models[] | {alias, status, present, path}'
qasr --json languages | jq '{alignable, forceable: [.forceable[].iso]}'
```

`doctor` 不加 `--strict` 时即使 `status:"fail"` 也退出 0；必须读 `.status`。

### 4.3 单文件转写

```bash
qasr --json transcribe a.wav -l auto                          # [model]
qasr --json transcribe a.wav -l zh --context "Qwen3" --context "通义千问"  # [model]
```

`language_source` 在 transcribe 的 `.result.language_source`：`auto`（LID）、`forced`（指定）、`server`（远程）。

### 4.4 字幕（SRT / VTT）

```bash
qasr --json transcribe a.wav -l zh --timestamps -f srt,vtt -o out/   # [model]
qasr --json transcribe a.wav -l yue --timestamps -f srt -o out/      # [model] 粤语可强制/可对齐
```

11 种可对齐语言：`zh en yue fr de it ja ko pt ru es`。`--timestamps --language auto` → 4。
`-f srt,vtt` 与 `--json` 合法；`-f json` 与 `--json` 会 2（见 §3）。
`-f srt/vtt` 必须同时给 `--timestamps`，否则以 4 拒绝。

### 4.5 已知文本对齐（注意覆盖）

```bash
qasr --json align a.wav --text-file script.txt -l zh -f srt -o out/ --dry-run   # [dry-run]
qasr --json align a.wav --text-file script.txt -l zh -f srt -o out/             # [model]
qasr --json align a.wav --srt draft.srt -l en -f srt -o out/ --coverage-min 0.8 # [model]
```

`align` **没有 `--overwrite`**：目标存在即静默覆盖。先看 `.plan.outputs[].path`，取得用户确认或换新目录。
`align` 必须给 `--language`（可 `auto`）；`language_source` 枚举是 `explicit|script|lid_probe`。

### 4.6 目录 / glob 批量 + resume

```bash
qasr --json transcribe ./audio --recursive -f txt,srt -l zh --timestamps \
  -o out/ --resume --skip-existing        # [model]
```

本地 `--jobs` 会被强制为 1；并行靠 `--batch-size`（vLLM 默认 8，transformers 默认 1）。
中断后用同一命令复跑；state 在 `<out-dir>/.qasr/state.jsonl`。
`--resume` 按 job key（audio SHA + model/aligner fingerprint + 规范化请求参数 + backend + `qasr_version` + 相对输出路径）判断，不是“目录存在就跳过”；`--skip-existing` 只跳过已存在且非空的输出。`--retry` 不能与 `--fail-fast` 同用（2）。

### 4.7 manifest（异构参数）

```jsonl
{"id":"ep01","audio":"audio/ep01.wav","language":"zh","formats":["txt"]}
{"id":"ep02","audio":"https://example.com/ep02.wav","language":"zh","formats":["srt"],"timestamps":true}
```

```bash
qasr --json transcribe --manifest jobs.jsonl -o out/ --max-chunk-seconds 180 --resume   # [network][model]
```

前置：每条 `timestamps:true` 的记录必须自带 `language`；per-item timestamps **不会**把 CLI 默认 270 改成 180，必须显式 `--max-chunk-seconds 180`。
**不要**用 CLI 级 `--timestamps` 兜底：它会把所有未显式写 `timestamps` 的记录切到时间戳模式（显式 `false` 仍是 false），缺可对齐 `language` 的记录直接 exit 4。

### 4.8 stdin 清单 / 直接 URL

```bash
printf '%s\n' a.wav b.wav | qasr --json transcribe --stdin-list -o out/ --resume   # [model]
qasr --json transcribe https://example.com/a.wav -o out/                          # [network][model]
```

多输入不带 `--json`/`--jsonl` 时必须 `--out-dir`。

### 4.9 长音频

```bash
qasr --json transcribe long.wav -l zh --max-chunk-seconds 270 -o out/                  # [model]
qasr --json transcribe long.wav -l zh --timestamps --max-chunk-seconds 180 -f srt -o out/  # [model]
qasr --json align long.wav --text-file long.txt -l zh -f srt -o out/                  # [model]
```

阈值：无时间戳默认 270s；有时间戳默认/上限 180s；远程上限 600s。单文件长音频 OOM 时降 `--max-chunk-seconds` 或换 0.6b，降 `--batch-size` 无效。
0.3.0 没有 VAD；静音检测/按说话人切块不在能力范围。`align` 自带 180s 窗口（`--window-seconds` 10–180，`--window-overlap` 0–10），长音频通常不需要手动切块。

### 4.10 远程客户端

Token 由用户在自己的交互 shell 里设置（`read -rs QASR_SERVE_TOKEN; export QASR_SERVE_TOKEN`）；agent 不读取、不打印、不代填，也不在命令行写前缀赋值（会留在 history）。

```bash
qasr --json --server http://127.0.0.1:18000 transcribe a.wav --jobs 4   # [remote]
qasr --json --server http://127.0.0.1:18000 transcribe a.wav --context "Qwen3"  # [remote]
```

远程不支持 `--timestamps`、`--granularity`（非 sentence）、`align`、`stream`，发请求前以 4 拒绝。`context` 走 `/v1/chat/completions`，`-l` 不被强制；读 `.result.warnings[]` 确认是否出现 `language_not_enforced_on_chat` / `server_ignored_language`。不传 `--model` 时客户端读 `/v1/models` 的第一个 id；JSON `.model.id` 是实际使用的 id。

### 4.11 本地 vLLM

```bash
qasr --json --backend vllm transcribe a.wav --batch-size 8                    # [linux-cuda][model]
qasr --json --backend vllm --gpu-memory-utilization 0.5 transcribe a.wav      # [linux-cuda][model]
```

`--batch-size` 对 vLLM 同样生效（默认 8）；`--max-inference-batch-size` 是模型内部批。`--gpu-memory-utilization` 只允许 vLLM，传给它后端会 exit 2（不是静默忽略）。
共享 GPU 上显式给 `--gpu-memory-utilization 0.3–0.6`，不要用默认占满；这是 vLLM 显存预留比例，不是硬上限。

### 4.12 serve（先门禁，再启动）

```bash
qasr --json models show 1.7b | jq -e '.status == "complete"'      # [model]
qasr --json serve 1.7b --dry-run | jq -e '.source == "local"'     # [dry-run]
tmux new-session -d -s qasr-serve 'qasr serve 1.7b --host 127.0.0.1 --port 18000'   # [linux-cuda]
tmux capture-pane -pt qasr-serve | tail -20
```

`source:"repo"` 表示本地缺失/半包，`qwen-asr-serve` 启动后会下载；`--offline` 不约束子进程。serve 没有内建鉴权，`QASR_SERVE_TOKEN` 只被客户端读取。
`qasr serve` 不注入 `--host`，vLLM 自己的默认是 `0.0.0.0`；**必须显式 `--host 127.0.0.1`**，除非用户明确要求且已有防火墙/反代/TLS。
`serve --dry-run` 只证明 argv 与模型解析，macOS 也能跑；实际启动仍需 Linux+CUDA+`[vllm]`。未知 flag 放在 MODEL 之后会原样透传给 `qwen-asr-serve`。

### 4.13 stream

```bash
qasr stream --input a.wav --emit jsonl        # [linux-cuda]
qasr stream --input mic --emit text           # [linux-cuda][mic]
arecord -q -f S16_LE -r 16000 -c 1 | qasr stream --input stdin --pcm s16le --emit jsonl  # [linux-cuda]
```

`stream` 拒绝 `--json`/`--dry-run`/URL/目录（2/4）；`--max-stream-seconds` 默认 3600，到点以 5 结束；快照是累计文本，可能被回滚。
`--input FILE` 读到 EOF 即结束，不是 `tail -f`；要跟管道增长请用 `--input stdin`。

### 4.14 bench

```bash
qasr --json bench --duration 30 --iterations 3    # [model]
qasr --json bench a.wav --iterations 5            # [model]
```

字段是 `rtf/load_s/audio_s/elapsed_s/iterations/peak_memory_mb`，**没有 `.metrics`**。

### 4.15 models

```bash
qasr --json models list
qasr --json models show 1.7b
qasr --json models verify 1.7b
qasr --json models rm 1.7b --dry-run             # [managed] 先看计划，无需 --yes；dev path 会 2
# 用户确认后：
qasr models rm 1.7b --yes
qasr models pull 0.6b --source hf --resume       # [network]，先问用户
```

`rm` 只删 `QASR_HOME/models` 下的受管目录；dev 路径会被拒绝。
没有 `models cache` 子命令；缓存与落点看 `models path` / `QASR_HOME`。
`--model` 接受 alias（`1.7b`/`0.6b`）、本地路径或 HF repo id；`aligner` 只用于 `--aligner`。`--model-dir` / `QASR_MODEL_DIR` 改搜索根。`models show` 的 `.fingerprint`（`sha256:...`）可用于跨机器核对同一权重，也是 resume job key 的一部分。

### 4.16 config

```bash
qasr --json config path
qasr --json config get backend
qasr --json config show | jq '.config | to_entries[] | select(.value.source != "default")'
# 仅在用户确认落点后：
qasr config set backend vllm --project
```

优先级：默认 < `QASR_HOME/config.json` < 项目 `.qasr/config.json` < 环境变量 < `$QASR_CONFIG` < `--config` < CLI。`config set` 立即写盘，**没有 dry-run**；在 macOS 上把 backend 写成 vllm 会让后续命令直接 3。

### 4.17 jobs

```bash
qasr --json jobs list --limit 20
qasr --json jobs show JOB_ID
qasr --json jobs show out/          # 已登记的 out-dir 才可作查找键
```

`.state` 只有 `present|missing`；`.counts` 是 `pending/running/done/failed/skipped`。

### 4.18 finetune lint

```bash
qasr --json finetune lint-jsonl train.jsonl
qasr --json finetune lint-jsonl train.jsonl --strict --no-check-files
```

不导入 torch/datasets，不解码音频；exit 4 表示有错或文件不可读。

### 4.19 completion

```bash
qasr completion zsh | head                  # 先看 stdout 非空、内容合理
# 写 home 属于用户决定；用户确认落点后再重定向：
mkdir -p ~/.config/fish/completions
qasr completion zsh  > ~/.qasr-completion.zsh
qasr completion bash > ~/.qasr-completion.bash
qasr completion fish > ~/.config/fish/completions/qasr.fish
source ~/.qasr-completion.zsh
```

`qasr completion` 只打印脚本，不落盘；shell 名写错 exit 2。

---

## 5. 后端 / 设备决策树

```
有 --server 或 QASR_SERVE_URL?
├─ 是 → backend=remote
│   ├─ 纯文本/context/批量 → --jobs 1..32（默认 4）
│   ├─ context → chat/completions，-l 不强制（读 warnings）
│   ├─ 要时间戳 / align / stream → 不支持，exit 4
│   └─ --max-chunk-seconds >600 → 降为 600 + warning
└─ 否 → 本地
    ├─ 要 stream → 必须 Linux + CUDA + [vllm]，否则 4
    ├─ backend=auto
    │   ├─ Linux + CUDA + vllm 可导入 → vllm
    │   └─ 其他（含 macOS/MPS、CPU）→ transformers
    ├─ 显式 vllm 但平台不支持 → transcribe/align 3；stream 4；不自动降级
    └─ dtype=auto：CUDA→bf16，MPS→fp16，CPU→fp32
```

| 能力 | transformers | vllm | remote |
|---|---|---|---|
| 文本 / LID | ✓ | ✓ | ✓ |
| context | ✓ | ✓ | ✓（走 chat，`-l` 不强制） |
| 时间戳 | ✓ | ✓（aligner 走 transformers） | ✗ 4 |
| `align` | ✓ | ✓ | ✗ 4 |
| 多文件 | ✓（串行，jobs 强制 1） | ✓（串行，jobs 强制 1） | ✓（jobs 1–32） |
| stream | ✗ 4 | ✓（仅 Linux/CUDA） | ✗ 4 |
| 微批 | `--batch-size`（默认 1） | `--batch-size`（默认 8）+ `--max-inference-batch-size` | 恒 1 |
| `--gpu-memory-utilization` | ✗ 2 | ✓ (0,1] | ✗ 2 |

显式 `--device cuda:N` 在 import torch 前设置 `CUDA_VISIBLE_DEVICES`；`--device mps` 配 vLLM 直接失败。
`auto` 在 Linux+CUDA 上探测到 vLLM 但导入失败时会回退 transformers，JSON 顶层出现 `backend_fallback:true`；显式 `--backend vllm` 不回退，直接 3。

**离线与模型缺失**：`transcribe`/`align` 不会自动下载，缺模型 → 3；`serve` 是例外（见 §4.12）；`--offline` 对 serve 子进程不生效。

**主要风险**

| 风险 | 触发条件 | 规避 |
|---|---|---|
| 依赖钉死冲突 | `qwen-asr` 要求 `transformers==4.57.6`、`vllm==0.14.0` | 用 `uv tool` 隔离；不要把 `[vllm]` 装进 macOS 主环境 |
| MPS 能力有限 | macOS/MPS，flash-attn 不可用，fp16 显存/算子限制 | 用 `backend=auto`；长音频降 chunk 或换 0.6b；必要时 CPU 慢跑 |
| 长音频显存/时长 | 单文件数十分钟，transformers 串行 | 先 `--dry-run` 看窗口；显式 270/180；OOM 降 chunk |
| 时间戳精度 | 跨 chunk 英文粘词、aligner 180s 上限、coverage 低 | 选 11 种可对齐语言；读 `warnings`/`coverage`；必要时重切 |
| 模型半包 | `.aria2`/`.incomplete`/safetensors 缺失 | `models verify`；`pull --resume`；serve 前必须 complete |
| 流式不可用 | macOS、远程、非 vLLM | 明确降级到 `transcribe`（无实时回滚） |

Qwen3-ASR README 提到 aligner “up to 5 minutes”，而 qasr 的实现常量是 180s；skill 一律以 qasr 的 180s 为准。

---

## 6. 安全与破坏性操作

**Token（客户端）**

- `QASR_SERVE_TOKEN` 只从环境读，客户端使用；`qasr serve` 本身**不校验**。
- 由用户在自己的交互 shell 执行 `read -rs QASR_SERVE_TOKEN; export QASR_SERVE_TOKEN`；agent 不读取、不打印、不代填。
- 禁止把 token 写进命令行前缀赋值、URL userinfo、argv、config、JSON、state、日志；这些都会留在 history 或泄漏。

**下载与外发**

- `transcribe`/`align`：不会自动下载；缺失/半包 → exit 3。
- `serve`：本地缺失/半包会回退 repo id，`qwen-asr-serve` 启动后可能下载；`--offline` 拦不住。
- serve 前必须 `models show`/`verify` 为 `complete`；否则先问用户是否 `models pull`。
- `models pull`、远程 `--server`、URL 输入都会联网；先说明体积/来源/外发目标，由用户确认。

**覆盖与删除**

| 操作 | 默认 | 需要 |
|---|---|---|
| `transcribe -o DIR / --output` | 目标存在即 exit 4 | 用户同意后才 `--overwrite`；批量优先 `--resume`/`--skip-existing` |
| `align -o / --output` | **静默覆盖**（无 `--overwrite`） | 先 `--dry-run` 看 `.plan.outputs[].path`，确认或换新目录 |
| `models rm` | 只删受管目录 | 先 `models show`/`rm --dry-run`，用户同意后 `--yes` |
| `config set` | 立即写盘 | 先 `config path` 给用户看落点；项目级 `--project` |

**长任务（只允许 tmux）**

- 超过 ~10 分钟的任务一律 tmux；**禁止 nohup 与裸 `&`**。
- serve 是常驻进程，必须在 tmux 里跑。

```bash
tmux new-session -d -s qasr-job 'qasr --jsonl transcribe ./audio -o out/ --resume > qasr.jsonl 2> qasr.err'
tmux capture-pane -pt qasr-job | tail -20
tmux attach -t qasr-job
```

- 远程长任务同样只在 tmux 内运行；重连后用 `jobs show` 恢复上下文，不重复提交。
- 中断（130）后先看 `.qasr/state.jsonl`，再决定 `--resume`。

**服务暴露**

- `serve` 无内建鉴权；`qasr serve` 不注入 `--host`，vLLM 默认是 `0.0.0.0`。
- 必须显式 `--host 127.0.0.1`；不要绑 `0.0.0.0`，除非用户明确要求且已有网络层保护（防火墙/反代/TLS）。

**执行边界**

- 不替用户跑 `models pull`、`models rm`、`serve` 公网绑定、`finetune` 训练。
- 不修改 `<out-dir>/.qasr/` 下的 state/lock/manifest。
- 输出路径不要指向输入、模型目录或 `<out-dir>/.qasr/`；manifest 的 `output` 必须是相对 `--out-dir` 的路径。

---

## 7. 失败处理与退出码

| code | 含义 | agent 动作 |
|---|---|---|
| 0 | 成功 | 仍要读 `.ok`；`doctor` 可能 `status:"fail"` |
| 2 | 用法/配置/manifest schema | 读 `error.message`/`hint` 修参数；不重试 |
| 3 | 模型/运行时/后端/下载 | `doctor --json`；查模型完整性、torch、vLLM、网络 |
| 4 | 输入/音频/路径/能力不支持 | 换输入或后端；远程要时间戳属此类，不重试 |
| 5 | `align` 覆盖率低于 `--coverage-min`；或 `stream` 达 `--max-stream-seconds` | `align` **未写任何文件**；`stream` 无文件产物；读 error envelope |
| 6 | 批量部分失败 | `.errors[]`；按 `retryable` 决定 `--retry` |
| 130 | SIGINT/SIGTERM | `--resume`；检查锁与中间产物 |

规则：

- **5 的边界**：`align` 在 `_write_outputs` 之前抛错，磁盘无结果、`.outputs` 不存在；`stream` 不写文件（`--emit jsonl` 会在 stdout 留终止错误快照，`--emit text` 只进 stderr）；`transcribe` 没有 quality gate，不会返回 5。README 的 "subtitle quality gate" 是未实现文案。
- **fail-fast**：`--fail-fast` 取第一个错误的码，而不是 6。
- **partial**：非 fail-fast 且部分成功 → `ok:false` + 6；`.summary.ok` 可能 >0。
- **all-fail**：非 fail-fast 且一个都没成功 → 首个错误的码，不是 6。
- **retryable**：只对 `error.retryable==true` 且用户同意时重试；`--retry` 只重试一次。
- 远程超时：先 `--timeout` 调大，再降 `--jobs`；不盲目加 `--retries`。

---

## 8. 验证与自检

```bash
# 1) 单输入 stdout 恰好一个 JSON 对象且 ok
qasr --json transcribe a.wav | jq -e '.ok == true and .command == "transcribe"'

# 2) 批量账本自洽
qasr --json transcribe ./audio -o out/ --resume \
  | jq -e '.summary.total == (.summary.ok + .summary.failed + .summary.skipped)'

# 3) 时间戳非空、单调且 end>=start（自包含，不依赖 sidecar）
qasr --json transcribe a.wav -l zh --timestamps \
  | jq -e '(.result.segments | length) > 0
           and ([.result.segments[].start] as $s | $s == ($s | sort))
           and all(.result.segments[]; .end >= .start)'

# 4) 对齐覆盖率（阈值显式，先判 ok）
qasr --json align a.wav --text-file t.txt -l zh --coverage-min 0.8 \
  | jq -e '.ok == true and .result.coverage >= 0.8'

# 5) 计划与产物一致（srt 需要 timestamps + 可对齐语言）
qasr --json transcribe a.wav -l zh --timestamps -f srt -o out/ --dry-run \
  | jq -e '.plan.outputs[0].path | endswith(".srt")'
```

其他判据：

- **schema**：`schema_version==1`；错误对象必有 `kind/code/message/retryable/details`。
- **resume**：批量复跑 `.summary.skipped` 上升；**单文件**没有 `.summary`，看 `.skipped`。
- **jobs**：`.state=="present"` 且 `.counts.pending==0`、`.counts.running==0` 即全部落定。五类之和等于 `.job.total` **只在没有 spec 校验失败时**成立：`total` 含 `spec_errors`，而它们不写 state（`commands/transcribe.py:532-534`）。
- **dry-run**：只证明解析/计划，不证明模型存在（`--model /nope` 也 `ok:true`）；体检用 `doctor`/`models verify`。
- **sidecar**：`out/*.json` 里的 `outputs` 可能为 `[]`/缺失（先写文件后组装 payload）；以 stdout envelope 为准。
- **jsonl**：`--jsonl` 只有逐条 item 行，没有 summary；summary 只在人类 stderr。不要把最后一行当批次汇总。
- **stream**：`seq` 从 0 连续递增，最后一条 `is_final:true`；累计文本允许被更短文本替换。
- 进度与三方日志在 stderr；`2>/dev/null` 不应改变 stdout 的 JSON。

---

## 9. 反模式

| 反模式 | 后果 | 正确做法 |
|---|---|---|
| `--json` 配 `-f json/jsonl/tsv/md` | exit 2 | 机器输出走 `--json`；文件格式用 `-f txt/srt/vtt` |
| 以为 `align` 也拒绝覆盖 | 静默毁旧文件 | align 无 `--overwrite`；先 dry-run + 用户确认 |
| 把 token 写进命令行前缀/URL | 进 shell history/日志 | 用户自己 `read -rs`/`export`；agent 不接触 |
| 解析人类输出/进度条 | 格式一变就错 | `--json`；stream 用 `--emit jsonl` |
| 远程求时间戳/对齐/流式 | exit 4 | 本地跑，或远程只取文本 |
| 远程 `--context` 以为 `-l` 生效 | 语言可能不符 | 读 `.result.warnings[]` |
| 缺模型时自动 `models pull` | 未经同意的下载 | 先 `models list`，由用户决定 |
| 以为 `serve --offline` 不下载 | 启动时拉 repo | serve 前 `models verify`；`source:"repo"` 即停下确认 |
| 以为 `serve` 有鉴权 | 裸奔 | 显式 `--host 127.0.0.1`；serve 无内建 auth |
| nohup / 裸 `&` 跑长任务 | 违反硬规则、难恢复 | tmux only；`capture-pane` 查看 |
| 1 小时音频 CPU 盲跑 | 数小时且可能 OOM | 先 dry-run 看窗口，或先跑 60s 样本 |
| 长音频 OOM 降 `--batch-size` | 单文件不经过 batching，无效 | 降 `--max-chunk-seconds` 或换 0.6b |
| `--timestamps --max-chunk-seconds 270` | exit 4 | 时间戳固定 ≤180 |
| `-f srt/vtt` 不给 `--timestamps` | exit 4 | 同时给 `--timestamps` 与可对齐 `-l` |
| 22 个方言标签用 `-l` | exit 4 | 只 `auto`；`yue` 例外，可强制 |
| 把 `doctor` exit 0 当健康 | 假成功 | 读 `.status` / `.checks[]` |
| macOS 装 `[vllm]` / 跑 stream | 不可用 | 仅 Linux+CUDA |
| `--gpu-memory-utilization` 配 transformers | exit 2 | 仅 vLLM |
| 读 `out/*.json` 的 `.outputs` | 可能为空 | 读 stdout envelope |
| 用 `.metrics` 读 bench | 字段不存在 | 顶层 `rtf/load_s/elapsed_s/...` |
| `-f markdown` | 无效 token | 用 `md` |
| 对空音频断言失败 | 零样本成功但空文本 | 看 `duration_s` 与 `.result.text` |

---

## 10. 维护

**版本绑定**

- SKILL.md 开头："按 `qasr 0.3.x` 编写；先 `qasr --version`，主/次版本不同则核对 `qasr <cmd> --help`。"
- 仓库内 skill 与 CLI 同 tag 发布。
- CLI 新增 flag：先改 `references/commands-*.md`，再改 SKILL.md 路由表。

**事实核对（每次发版）**

```bash
qasr --help >/dev/null || echo "BROKEN: root"
for c in transcribe align serve stream models languages doctor config jobs bench completion finetune; do
  qasr "$c" --help >/dev/null || echo "BROKEN: $c"
done
qasr --json languages | jq -e '.alignable | length == 11'
qasr --json doctor >/dev/null; echo "doctor exit=$? (may be 0 even when status=fail)"
```

**安装位置**

- 源：仓库 `skills/qasr-cli/`（review 跟代码一起走）。
- 实体：`~/.agents/skills/qasr-cli` 符号链接。
- Claude Code 发现入口：`~/.claude/skills/qasr-cli -> ../../.agents/skills/qasr-cli`。
- 项目级：`<repo>/.agents/skills/qasr-cli`（实体）与 `<repo>/.claude/skills/qasr-cli`（发现入口）。
- PyPI wheel 不打 skill。

**触发词评估（rubric 规模）**

1. 53 条 prompt：正向核心（qasr/Qwen3-ASR/语音转写/字幕/对齐/批量/流式/长音频/远程/vLLM）与高风险负向（TTS、克隆、ffmpeg、训练、云 API、"给视频配字幕"）。
2. 新会话只给 description，记录是否加载。
3. 门槛：正向核心 100% 触发；高风险负向 0 误触发；与 qtts-cli 的混淆单独统计。
4. 每次改 description 都重跑。

---

## 附录 A：CLI 事实核对（供 B 复核）

已实际执行（无推理、无下载）：

```text
qasr --version / --help
各子命令 --help
qasr --json doctor / models list|show|path|verify / jobs list / config show|path|get
qasr transcribe|align|serve|bench --dry-run
qasr stream --input x --json ; qasr stream --dry-run
```

关键偏差（均已写入正文）：

1. `-f` help 只列 5 种，实际 `tsv/md` 可用（`render.py:12`）。
2. `transcribe/align/bench --dry-run` 默认 JSON plan；`serve --dry-run` 默认纯文本，需 `--json`。
3. `doctor` 默认退出 0，即使 `status:"fail"`；`--strict` 才 3。
4. `--dry-run` 不校验模型存在（`--model /nope` 仍 `ok:true`）。
5. `stream --dry-run` 与 `stream --json` 均 exit 2。
6. `transcribe` 的 `-f json/jsonl/tsv/md` + `--json` 冲突；`-f txt/srt/vtt` 合法，`align -f json --json` 也合法。
7. `transcribe --output` 只接受单格式；多格式必须 `--out-dir`。
8. `transcribe` 拒绝覆盖；**`align` 静默覆盖且无 `--overwrite`**（`align.py:450-489`）。
9. `jobs list --limit N`；`jobs show` 接受 job id 或已登记 out-dir。
10. `config show` 的 `.config` 是 `{value, source}` 映射。
11. `align` 永远要求 `--language`（可 `auto`）。
12. 全仓只有 `align.py:392-415`（coverage）与 `stream.py:365-371`（cap）抛 `QualityError`；`transcribe` 不会 5；README 的 "subtitle quality gate" 未实现。
13. `serve` 的 `resolve_serve_model` 对缺失/半包回退 repo id（`serve.py:57-72`）；`--offline` 不约束子进程。
14. manifest 的 per-item `timestamps` 不改变 CLI 默认 chunk（`commands/transcribe.py:377-379`），需显式 180。
15. 远程 `context` 走 chat，警告 `language_not_enforced_on_chat` / `server_ignored_language`（`remote.py:648-652`）。
16. `bench` 无 `.metrics`；顶层 `rtf/load_s/audio_s/elapsed_s/iterations/peak_memory_mb`。
17. vLLM 的 `--batch-size` 默认 8（`commands/common.py:107-109`）。
18. `transcribe -f json` 的 sidecar `outputs` 为空（`transcribe.py:1377-1383`），以 stdout 为准。
19. `--gpu-memory-utilization` 非 vLLM 后端 → exit 2（`commands/transcribe.py:366-375`）。
20. 单文件 `--resume` 返回 item envelope，没有 `.summary`。
21. `qasr serve` 不注入 `--host`；`qwen-asr-serve`/vLLM 默认 `0.0.0.0`，示例必须显式 `--host 127.0.0.1`。
22. `align` 默认 `--window-seconds 180`、`--window-overlap 3`，长音频自动窗口化；README 的 “up to 5 minutes” 与 qasr 的 180s 常量冲突，以 180s 为准。

## 附录 B：不写进 skill

- 完整 `--help` 抄写；内部 Python API；训练流程；性能数字承诺；任何真实 token/内网地址/个人路径。

## 附录 C：修订对照表

### BLOCKER

| # | 改在哪 | 证据 |
|---|---|---|
| 1 | §3 远程行、§4.10、§6 Token、§9 | 删除所有命令行前缀赋值与 URL userinfo 占位；`remote.py:35-38` 只读环境 |
| 2 | §6 长任务、§4.12 | 只保留 tmux；禁止 nohup/裸 `&` |
| 3 | §7、§3 字幕行、§8 | 5 只保留 align coverage / stream cap；`align.py:392-415`、`stream.py:365-371` |
| 4 | §4.5、§6 覆盖表、§9 | align 无 `--overwrite`；`align.py:450-489` 直接 `atomic_write_bytes` |
| 5 | §4.12、§6 下载、§9、附录 A#13 | `serve.py:57-72` 回退 repo；`--offline` 不约束子进程 |

### MAJOR

| # | 改在哪 | 证据 |
|---|---|---|
| 6 | §3 批量/manifest、§4.4/4.6、§8#5 | `commands/transcribe.py:303-312`；改 `-f txt,srt` |
| 7 | §4.7 | manifest 补 `language`；显式 180；`commands/transcribe.py:377-379,561-565` |
| 8 | §2.1、§4.1 | `pyproject.toml:10`；补 `[local]` 与 extras 表 |
| 9 | §3 stdin/URL 行、§4.8 | `transcribe.py:299-302`；URL 需网络 |
| 10 | §3 远程 context、§4.10、§5 | `remote.py:648-652` |
| 11 | §3 serve、§4.12、§9 | `serve.py:20-24,127-142` |
| 12 | §3 方言、§4.4、§9 | `languages.py`：`yue`/Cantonese 在 forceable+alignable |
| 13 | §2.2、§10 | `~/.claude/skills/qtts-cli` 链接模式 |
| 14 | §6 Token/服务暴露、§4.12 | serve 无 token 校验；不传 `--host` 时 vLLM 默认 0.0.0.0，必须显式 127.0.0.1 |
| 15 | §4.11、§5、§9 | `commands/transcribe.py:366-375` → 2 |

### MINOR

| # | 改在哪 | 证据 |
|---|---|---|
| 1 | §3 bench、§4.14 | `schemas/bench.json` 无 `.metrics` |
| 2 | §1.3、§9 | `render.py:12` 是 `md` |
| 3 | §5、§4.11 | `commands/common.py:107-109` 默认 8 |
| 4 | §10 核对脚本 | 去掉空元素，先单独 `qasr --help` |
| 5 | §10 触发评估 | 扩到 53 条，正向 100%、高风险负向 0 |
| 6 | §3 续跑、§8 | `transcribe.py:705-717` 单文件无 `.summary` |
| 7 | §4.15/4.16 | rm 先 `--dry-run`；config set 无 dry-run、先 `config path` |
| 8 | §4.5 | align `language_source` = `explicit/script/lid_probe` |
| 9 | §8#4 | 显式 `--coverage-min 0.8` 且先判 `.ok` |
| 10 | §3 开头、§9 | 明确 stream 例外 |
| 11 | §8、§3 | sidecar `outputs` 为空，读 stdout |
| 12 | §3 长音频、§9 | 单文件不经过 batching，降 chunk/换模型 |
| 追加 | §3/§9 | `-f md` 是文件格式；冲突只限 json/jsonl/tsv/md |

### 审核外补正（Partner A 自查）

| 事实 | 改在哪 | 证据 |
|---|---|---|
| `qasr serve` 不注入 `--host`，vLLM 默认 `0.0.0.0` | §4.12、§6、§9、附录 A#21 | `serve.py` 仅在显式 host 时追加；qwen-asr/vLLM 默认绑定 |
| `--jsonl` 没有 summary，summary 只在 stderr | §8 | `transcribe.py:689-704` 只拼 item 行 |
| `--context` 可重复，按空格合并 | §4.3 | `cli.py:127` `action="append"`；`commands/transcribe.py:115-128` |
| 0.3.0 无 VAD；`stream` 无 tail 模式 | §4.9、§4.13 | 全仓无 VAD；`--input FILE` 读到 EOF |
| 没有 `models cache` 子命令 | §4.15 | `models --help` 只有 list/path/verify/show/rm/pull |
| `align` 自动窗口化，默认 180/3 | §4.9、附录 A#22 | `commands/align.py:79-98`；`align.py:380-388` |
| `auto` 的 vLLM 导入失败会回填 `backend_fallback` | §5 | `commands/common.py:89-105` |
| `--resume` 的 job key 含 `backend`、`qasr_version`、相对输出路径，不含输出 SHA | §4.6 | `jobs.py:99-120` 的 `job_key()` |
