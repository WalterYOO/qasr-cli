# qasr-cli 设计方案（v0.3）

> 状态：Partner A 构思稿 + Partner B 独立审核已完成，审核意见已全部修入本版。
> 目标：`uv tool install` 安装，`qasr` 命令调用，完整覆盖 Qwen3-ASR 能力。

---

## 0. 一句话定位

`qasr` 是**面向脚本的本地 ASR 运行时 + 批处理/对齐前端**：把官方 `qwen-asr` 的识别、LID、上下文偏置、长音频、时间戳、强制对齐、流式与 OpenAI 兼容服务，收敛成一套稳定、可恢复、可组合的 Unix 命令；不是官方 demo 的复刻，也不是训练框架。

目标用户：播客/视频转写、会议与访谈归档、TTS 数据质检、服务器批量任务、需要 SRT/VTT/JSON 的自动化脚本。

**已实机验证**（macOS M4 Pro / MPS / fp16）：`Qwen3-ASR-0.6B` + `Qwen3-ForcedAligner-0.6B` 加载 6.3s，短音频推理 4.6s，中文转写与 13 个字级时间戳均正确。依赖组合 `qwen-asr 0.0.6 / transformers 4.57.6 / torch 2.15.0 / Python 3.12`。

---

## 1. 与相邻工具的分工

| 工具 | 负责 | 不负责 |
|---|---|---|
| **qasr** | 多文件/目录/URL 批处理、上下文偏置、字幕、长音频、模型管理、统一 JSON 契约、脚本化 | 浏览器 UI、训练框架、重写 OpenAI server |
| `qwen-asr-demo*` | 交互演示 | 生产批处理 |
| `qwen-asr-serve` | `vllm serve` 包装、OpenAI 兼容接口 | 本地模型管理、批处理状态 |
| `qtts align` | TTS 链路里的"已知文本对齐"，带 lexicon、字幕 QC、recipe | 通用 ASR/批处理 |

`qasr align` 与 `qtts align` 有重叠，但职责不同：词表映射、TTS 项目配置、recipe 继续归 qtts；qasr 是通用工具，**不依赖 qwen-tts-cli**。未来 qtts 可在"无 lexicon、无项目配置"时委托 qasr，但不反向依赖。

### 明确不做（反目标）

- 说话人分离（模型不支持）、翻译（vLLM 的 `translate` task_type 只是强制输出语言标签，不是真翻译）、GUI、插件系统。
- 训练框架与数据管线；P2 最多做 `finetune lint-jsonl`。
- 把长音频"伪流式"包装成流式；不支持的组合直接报 capability 错误。
- 原生异步 job daemon；`state.jsonl` + `--resume` 已覆盖 CLI 场景。

---

## 2. 能力与语言契约

### 2.1 能力 → 命令

| 能力 | 官方入口 | qasr 入口 | 备注 |
|---|---|---|---|
| 单/多音频转写 | `transcribe(audio, ...)` | `qasr transcribe INPUT...` | 文件/目录/glob/URL/manifest |
| 语种识别（LID） | `language=None` | 默认 `--language auto` | 输出原始语言标签 |
| 提示语种 | `language="Chinese"` | `--language zh` | **prompt 偏置 + 标签覆盖**，不是识别保证 |
| 30 语言 | `SUPPORTED_LANGUAGES` | `qasr languages` | 可强制语言表 |
| 22 中文方言 | README 声明 | `qasr languages` | 仅 LID 可输出，**不可强制** |
| 上下文偏置 | `context="..."` | `--context` / `--context-file` | manifest 支持逐条 |
| 长音频切块 | 库内 1200s / 180s | 默认沿用；P1 自管 | 见 §8、§10 |
| 批量推理 | `audio=[...]`, `max_inference_batch_size` | `--batch-size` | qasr 微批 + 逐条错误隔离 |
| 时间戳 | `return_time_stamps=True` + aligner | `--timestamps` | P0 要求显式 11 语言之一 |
| 独立强制对齐 | `Qwen3ForcedAligner.align` | `qasr align` | 11 语言；P0 单窗 ≤180s |
| 流式 | `init/streaming/finish` | `qasr stream`（P2） | 仅 vLLM；累计快照 |
| OpenAI 兼容服务 | `qwen-asr-serve` | `qasr serve` + `--server`（P1） | 远程模式无需本地权重 |
| 微调 SFT | `finetuning/qwen3_asr_sft.py` | 不纳入 | P2 仅 lint |
| 复读修复/输出解析 | `parse_asr_output` | 始终启用 | 记录 `text_transform` |

### 2.2 三张语言表（必须分开）

1. **可强制语言（30）**：`SUPPORTED_LANGUAGES`。`--language` 只接受这些（含 ISO/别名映射）。
2. **LID 可输出标签（30 + 22 方言）**：模型可能返回方言名；**不校验**，原样写入 JSON `language`。
3. **可对齐语言（11）**：Chinese、English、Cantonese、French、German、Italian、Japanese、Korean、Portuguese、Russian、Spanish。`--timestamps` 与 `qasr align` 只接受这些。

方言表（展示用，不可强制）：Anhui、Dongbei、Fujian、Gansu、Guizhou、Hebei、Henan、Hubei、Hunan、Jiangxi、Ningxia、Shandong、Shaanxi、Shanxi、Sichuan、Tianjin、Yunnan、Zhejiang、Cantonese (Hong Kong)、Cantonese (Guangdong)、Wu、Minnan。

### 2.3 强制语种的真实语义

`language` 的实现只是 prompt 末尾追加 `language X<asr_text>`，并在 `parse_asr_output` 中把语言标签覆盖成用户值。qasr 必须：

- 文档写"提示语种/标签覆盖"，不写"保证识别为 X"。
- 输出前清理 text 中可能残留的 `<asr_text>` 或 `language X` 前缀。
- `language_source` 字段区分 `auto|forced`。

---

## 3. 命令表面

### 3.1 全局

```
qasr [GLOBAL] <command> [ARGS]

GLOBAL:
  --config PATH        JSON 配置层（合并，不是替换）
  --model NAME|PATH    别名 1.7b|0.6b 或路径/repo（默认 1.7b）
  --aligner auto|none|PATH
  --backend auto|transformers|vllm|remote
  --server URL         远程 OpenAI 兼容服务（隐含 backend=remote）
  --device auto|cpu|mps|cuda|cuda:N
  --dtype auto|bf16|fp16|fp32
  --attn-implementation auto|sdpa|flash_attention_2|eager   # transformers only
  --model-dir PATH
  --offline            禁止联网/下载
  --json               stdout 单个 JSON 对象
  --jsonl              stdout 逐行 JSON
  --dry-run            只解析与规划，不加载模型
  --quiet / --verbose / --log-level
  --version
```

采用 argparse（不引入 Typer/Click），沿用 qtts 的 `ArgumentParser.error() → UsageError`，保证参数错误也能输出 JSON 错误对象。

**`--json` / `--jsonl` / `--format` 的关系**：

- `--json`：单条结果输出单个 JSON 对象；批量输出一个批量 envelope。
- `--jsonl`：每个 item 一行 JSON（含错误行）。
- `--format json/jsonl` 是**文件输出**格式；与 `--json/--jsonl` 互斥，同时出现 → 用法错误。
- 多格式 + stdout → 用法错误，必须 `--out-dir`。

### 3.2 子命令树

```
qasr transcribe INPUT... [--manifest FILE]
qasr align AUDIO (--text TEXT | --text-file FILE | --srt FILE)
qasr models list|pull|path|verify        # P0
qasr models show|rm                      # P1
qasr languages
qasr doctor [--server URL]
qasr config show|path                    # P0
qasr config get|set                      # P1

# P1 — 运行时、部署与长音频
qasr serve [MODEL]
qasr jobs list|show
qasr bench

# P2 — 流式与生态
qasr stream --input mic|stdin|file
qasr completion bash|zsh|fish
qasr finetune lint-jsonl
```

**不设 `qasr batch` 子命令。** 批量就是 `transcribe` 的默认形态，复杂场景用 `--manifest`。理由：避免两套执行器、两套状态机、两套 JSON schema。

### 3.3 `qasr transcribe`

```
qasr transcribe a.wav b.mp3 ./podcasts/ "https://example.com/x.wav"
qasr transcribe --manifest jobs.jsonl --out-dir out --resume
qasr transcribe a.wav --format txt,json,srt --timestamps --language zh \
    --context-file hotwords.txt --model 1.7b --batch-size 1 --out-dir out --resume
```

参数：

```
-f, --format LIST        txt,json,jsonl,srt,vtt（P1 加 tsv,md）；逗号组合
-o, --out-dir DIR        输出目录
    --output PATH|-      单输出快捷方式
-l, --language auto|zh|en|yue|...
    --context TEXT       可重复
    --context-file PATH  每行一个词/短语；# 开头注释
    --timestamps         P0 要求 --language 显式为 11 语言之一
    --granularity sentence|word|char   默认 sentence
    --batch-size N       微批大小；默认 local=1, vllm/remote=8
    --max-inference-batch-size N       模型构造参数；默认 = batch-size
    --max-new-tokens N   auto | 整数；默认 auto（见 §10）
    --jobs N             仅远程模式有效；本地 >1 给 warning 并强制 1
    --recursive
    --manifest FILE
    --resume / --overwrite
    --skip-existing      # P1
    --retry              # P1，仅 retryable 错误
    --continue-on-error  默认开启
    --fail-fast
    --no-progress
```

行为契约：

- **stdout 只放结果**：单条 `txt` 只有文本；`--json` 一个对象；`--jsonl` 一行一个。进度/警告/第三方日志全部 stderr。
- 多输入默认 `jsonl`；显式 `--format txt --out-dir` 时按文件名写盘。
- 空音频（`language None<asr_text>`）输出空文本、`language: null`，不算失败。
- `--timestamps` 时 `txt` 仍只输出文本；时间信息走 `json/srt/vtt`。
- 单条输入且未指定 `--out-dir/--output` → stdout；多输入未指定 `--out-dir` → 用法错误。
- `--dry-run` 输出解析后的输入列表、模型/后端/设备、将产生的输出文件，不加载权重。

### 3.4 `qasr align`

```
qasr align interview.wav --text-file transcript.txt --language zh -f srt,json -o out/
qasr align a.wav --text "甚至出现交易几乎停滞的情况。" --language zh --granularity char
qasr align a.wav --srt existing.srt --language en --format json
```

- `--language` **P0 必填**，且必须是 11 种对齐语言之一。
- `--language auto`：P1 才提供。P0 若传入，尝试脚本判定（Han→Chinese，Kana→Japanese，Hangul→Korean，Cyrillic→Russian）；Latin 及其他脚本 → 报错要求显式语言（避免把法语/德语误判成英语）。
- P0 单窗 ≤180s。超过 → 退出 4，提示 `--window-seconds`（P1）。180s 与包内 `MAX_FORCE_ALIGN_INPUT_SECONDS` 一致，避免与时间戳路径不一致。
- 默认 `json`；`srt/vtt` 需要 cue 切分（见 §8）。
- `--coverage-min 0.5`：对齐覆盖参考文本的比例低于阈值 → 退出 5；`--coverage-min 0` 关闭。
  - 分子：对齐 items 规范化后与参考文本匹配的字符数；分母：参考文本规范化后的字符数。
  - 规范化：NFKC、去空白、去标点；用单调 `SequenceMatcher` 匹配。
- 不允许 `--server`：服务端无对齐端点，capability error。
- 输出：`--output`/`--out-dir` 与 transcribe 同规则。

### 3.5 `qasr stream`（P2）

```
qasr stream --input mic
qasr stream --input stdin --pcm s16le --sample-rate 16000 --channels 1 --emit jsonl
qasr stream --input file.wav --realtime
```

- 仅 `backend=vllm`（本地）；远程不支持。需要 `[vllm,mic]` 才能用麦克风。
- `--emit jsonl` 每行是**累计快照**，不是 token delta：

```json
{"schema_version":1,"seq":3,"text":"今天天气","language":"Chinese","is_final":false}
```

原因：qwen-asr 每个 chunk 会回滚最后若干 token 重新解码，文本是"可修订的累计值"，无法安全做增量 diff。

- 参数：`--chunk-ms 2000`、`--unfixed-chunks 2`、`--unfixed-tokens 5`、`--max-stream-seconds`（默认 3600，防 O(n²)）。
- 人类模式用 `\r` 覆盖当前行，结束时换行。

### 3.6 `qasr models` / `languages`

```
qasr models list [--json]
qasr models pull 1.7b --source hf|modelscope [--revision REV]
qasr models path 1.7b
qasr models verify 1.7b [--full]
qasr models show 1.7b        # P1
qasr models rm 0.6b          # P1；只删 QASR_HOME/models 下、需确认
qasr languages [--json]
```

别名表：

| 别名 | 仓库 | 用途 |
|---|---|---|
| `1.7b` | `Qwen/Qwen3-ASR-1.7B` | 默认识别 |
| `0.6b` | `Qwen/Qwen3-ASR-0.6B` | 快速/高吞吐 |
| `aligner` | `Qwen/Qwen3-ForcedAligner-0.6B` | 时间戳/对齐 |

解析优先级：

1. `--model` 显式路径或 repo id；
2. `QASR_MODEL` / `QASR_MODEL_DIR`；
3. 项目 `.qasr/config.json`；
4. `QASR_HOME/models`；
5. 开发期兼容路径：`./models`、`../Qwen3-ASR/models`（仅存在时）；
6. 在线模式交给 HuggingFace/ModelScope 下载。

默认 **1.7B**，不静默降级。24GB M4 Pro 跑 1.7B fp16 + aligner 可行（需 P0 集成测试验证）；要速度显式 `--model 0.6b`。

### 3.7 `qasr serve` / 远程（P1）

```
qasr serve 1.7b --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.8
qasr serve --dry-run
qasr transcribe a.wav --server http://gpu-box:8000
qasr doctor --server http://gpu-box:8000
```

- `serve` 需要 `[vllm]` extra；`os.execvp` 转发到 `qwen-asr-serve`，不自管进程；未知参数透传。
- 远程客户端：无 context 时走 `/v1/audio/transcriptions`（multipart）；有 context 时走 `/v1/chat/completions`（system message + audio_url/base64），再用 `parse_asr_output` 解析。
- 远程语言参数用 ISO 码映射；服务端拒绝时重试一次不带 language，并给 warning。
- 远程不支持时间戳/对齐；流式也不支持（qwen-asr serve 不暴露 chunk 级回滚语义）。请求时 capability error。
- `QASR_SERVE_TOKEN` 作为 Bearer token，只从环境变量读。
- `--timeout 300`、`--retries 2`；4xx→退出 4，5xx/超时→退出 3（retryable）。

### 3.8 `qasr doctor`

- 检查：Python/qasr 版本；`torch`/`qwen-asr`/`vllm` 可导入性与版本；`transformers` 是否 4.57.6；设备/显存/内存；模型目录与就绪状态、磁盘余量；soundfile/libsndfile、ffmpeg；`--server` 可达性与 `/v1/models`。
- 结论 `ok|warn|fail`；默认退出 0，`--strict` 在 warn/fail 时退出 3。
- `--json` 输出 `{schema_version, ok, status, checks:[{name,status,value,detail,hint}]}`。

### 3.9 `qasr config`

- 路径：`--config` > `$QASR_CONFIG` > 项目向上查找 `.qasr/config.json` > `$QASR_HOME/config.json`。
- `config show` 显示合并后的有效配置与每项来源；`config path` 打印实际文件；`config get/set` 为 P1。
- 键：`model`、`aligner`、`backend`、`device`、`dtype`、`attn_implementation`、`language`、`context`、`format`、`batch_size`、`max_inference_batch_size`、`jobs`、`model_dir`、`offline`、`server`。
- 优先级（低→高）：内置默认 < `QASR_HOME/config.json` < 项目 `.qasr/config.json` < 环境变量 < `--config` < CLI。相对路径相对配置文件解析。
- 环境变量：`QASR_MODEL`、`QASR_MODEL_DIR`、`QASR_BACKEND`、`QASR_DEVICE`、`QASR_DTYPE`、`QASR_OFFLINE`、`QASR_SERVE_URL`、`QASR_SERVE_TOKEN`、`QASR_CONFIG`。

---

## 4. 架构

```
qasr_cli/
  __init__.py            # __version__
  __main__.py            # python -m qasr_cli
  cli.py                 # argparse、分发、JSON/错误出口
  errors.py              # QasrError + 退出码
  config.py              # 配置合并、平台目录
  catalog.py             # 模型别名/扫描/指纹（纯 Python）
  inputs.py              # 文件/目录/glob/URL/manifest（纯 Python）
  audio.py               # 探测/时长/格式检查；不 import torch
  languages.py           # 30 可强制 / 52 LID / 11 对齐 三张表（纯 Python）
  runtime.py             # 懒加载 qwen_asr/torch，能力探测
  backends.py            # Backend 协议 + Transformers/VLLM/Remote
  transcribe.py          # 编排：输入 → 微批 → 结果 → 输出
  align.py               # 已知文本对齐
  long_audio.py          # 纯函数：窗口规划、文本锚定、时间合并（P1）
  subtitles.py           # cue 构建、SRT/VTT 渲染、舍入（纯 Python）
  render.py              # txt/json/jsonl（纯 Python）
  jobs.py                # state.jsonl、原子写、resume、job_key（纯 Python）
  doctor.py
  commands/              # 每个子命令一个薄适配层
schemas/                 # 所有 --json 输出的 JSON Schema
```

核心抽象：

```python
class Backend(Protocol):
    name: str
    def capabilities(self) -> Capabilities: ...
    def transcribe(self, items: list[AudioItem], opts: AsrOptions) -> list[AsrResult]: ...
    def align(self, items: list[AudioItem], texts: list[str],
              languages: list[str], opts: AlignOptions) -> list[AlignResult]: ...
    def close(self) -> None: ...
```

硬约束：

- `cli.py`、`config.py`、`catalog.py`、`inputs.py`、`audio.py`、`languages.py`、`long_audio.py`、`subtitles.py`、`render.py`、`jobs.py`、`doctor.py` **不得 import torch/qwen_asr/vllm**。
- `qasr --help` 不加载任何模型依赖；`--dry-run`、`models list`、配置校验可纯 CPU 运行。
- 运行时模块懒 import；`--offline` 必须在 import transformers **之前**设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，并向 `from_pretrained` 传 `local_files_only=True`。

### 4.1 微批与错误隔离（修正）

`transcribe(list)` 会先 `normalize_audios` 解码整个 list，`max_inference_batch_size` 是模型构造参数，不能控制输入解码内存。qasr 必须：

1. 先逐条校验路径/URL、格式、时长，不把坏文件放进 batch。
2. 按 `--batch-size` 把 **已校验** 的 item 分组，每组调用一次 `transcribe(group)`。
3. 组失败时降级为逐条调用，定位坏 item；成功项保留，失败项写 `error`，继续其他组。
4. `--max-inference-batch-size` 单独传给模型构造，与输入分组解耦。
5. 本地默认 `batch-size=1`（内存优先）；vLLM/remote 默认 8。

---

## 5. 后端策略

| 能力 | transformers | vLLM | remote |
|---|---|---|---|
| 批量转写 | ✅ | ✅ | ✅（客户端并发） |
| 上下文偏置 | ✅ | ✅ | chat 端点 ✅；transcription 端点 ❌ |
| 时间戳 | ✅（需 aligner） | ✅（aligner 仍走 HF） | ❌ |
| 独立 align | ✅ | ✅ | ❌ |
| 流式 | ❌ | ✅（单流） | ❌ |
| macOS/MPS | ✅ | ❌ | ✅ |
| Linux/CUDA | ✅ | ✅ | ✅ |

### 5.1 三种运行形态与 GPU 的关系

"有没有 GPU"和"要不要远程"是两个独立问题。qasr 有三种形态：

| 形态 | 典型命令 | 模型跑在哪 | CLI 所在机器需要 GPU 吗 | 适用场景 |
|---|---|---|---|---|
| **本地 transformers** | `qasr transcribe a.wav` | 本机 | 不需要（MPS/CPU 可跑） | Mac 单机、无 GPU 服务器 |
| **本地 vLLM** | `qasr --backend vllm transcribe ...` | 本机 | 需要 Linux/CUDA | 就在 GPU 机器上跑、要高吞吐/流式 |
| **远程客户端** | `qasr --server http://gpu:8000 transcribe ...` | 远端常驻服务 | 不需要 | Mac 调 GPU 服务器、多客户端共享、服务器已在跑服务 |

关键点：

- **有 GPU ≠ 必须远程。** 如果你人就在 GPU 机器上，本地 vLLM 少一层网络、少一个常驻服务，最简单。
- **远程解决的是"CLI 机器 ≠ GPU 机器"**：MacBook 调 Linux GPU box；多人/多任务共享一个常驻服务；服务器上已经跑着 vLLM。
- **没有 GPU 也能完整用离线能力**：transformers + MPS/CPU。实测 0.6B 在 M4 Pro 上稳态 RTF 0.056（1 小时音频约 3.4 分钟）。
- **远程的能力边界**：服务端 API 不返回时间戳、不暴露流式回滚语义，所以远程只做转写（P1 再补 context）；对齐/流式仍需本地。
- **远程的代价**：需要服务端常驻、网络、音频上传；服务端版本与 qasr 客户端要约定 API 契约。

由于三波功能连续实现，远程不是"要不要做"，而是**固定在 P1 的运行时/部署功能组里实现**，不与 P0 的单机核心链路交叉。P0 先把本地 transcribe/align/批处理/模型校验做扎实，P1 再引入 serve、remote、vLLM 与长音频。

`--backend auto`：

1. 指定 `--server` → `remote`；
2. `stream` → 必须 `vllm`；Linux+CUDA+已安装才可用，否则退出 4；
3. Linux+CUDA+vLLM 可导入 → `vllm`；
4. macOS → `transformers` + MPS；
5. 其他 → `transformers` + CPU。

`auto` 模式下非流式任务若 vLLM 不可用可回退 transformers，但 JSON 记录 `"backend_fallback": true` + warning。显式 `--backend vllm` 不可用 → 退出 3，附安装命令与平台说明。

设备/dtype：CUDA→bf16，MPS→fp16，CPU→fp32。MPS 不默认 bf16；失败时明确报错并提示 `--dtype fp32`，不静默回退 CPU。

---

## 6. 输出契约

### 6.1 stdout / stderr

- stdout：只有结果（文本、JSON、字幕）。
- stderr：日志、进度、警告、模型下载提示、第三方 banner。
- 调用 qwen_asr 时用 `stdout_to_stderr()` 包裹 fd 1，防止第三方库污染结果。

### 6.2 通用 envelope

所有 `--json` 输出都有：

```json
{"schema_version": 1, "ok": true, "command": "transcribe"}
```

错误：

```json
{
  "schema_version": 1,
  "ok": false,
  "command": "transcribe",
  "error": {
    "kind": "capability",
    "code": 4,
    "message": "timestamps require an explicit alignment language",
    "hint": "pass --language zh|en|yue|fr|de|it|ja|ko|pt|ru|es",
    "retryable": false,
    "details": {}
  }
}
```

### 6.3 transcribe 单条

```json
{
  "schema_version": 1,
  "ok": true,
  "command": "transcribe",
  "input": {"id": "a", "source": "a.wav", "duration_s": 12.34, "sha256": "..."},
  "model": {
    "id": "Qwen/Qwen3-ASR-1.7B",
    "backend": "transformers",
    "device": "mps",
    "dtype": "fp16",
    "fingerprint": "sha256:..."
  },
  "result": {
    "text": "今天天气不错。",
    "language": "Chinese",
    "language_source": "auto",
    "segments": [{"start": 0.12, "end": 0.34, "text": "今", "kind": "char"}],
    "timestamp_source": "forced_aligner",
    "text_transform": "qwen-asr-repetition-fix-v1",
    "warnings": []
  },
  "metrics": {"load_ms": 0, "infer_ms": 830, "wall_ms": 900, "rtf": 0.067},
  "outputs": [{"format": "srt", "path": "out/a.srt", "sha256": "..."}]
}
```

- `rtf = infer_ms / (duration_s * 1000)`。
- `timestamp_source`: `forced_aligner | null`。
- `segments.kind`: `sentence | word | char`。

### 6.4 transcribe 批量

只有 **多于一条输入** 或使用 `--manifest` 时使用批量 envelope：

```json
{
  "schema_version": 1,
  "ok": false,
  "command": "transcribe",
  "summary": {"total": 2, "ok": 1, "failed": 1, "skipped": 0},
  "items": [{"id": "a", "ok": true, "result": {}}],
  "errors": [{"id": "b", "error": {"kind": "input", "code": 4, "message": "..."}}]
}
```

`--jsonl` 每行一个 item（成功或失败），失败行带 `"ok": false`；最后不额外输出 summary（summary 只走 stderr 人类输出）。

### 6.5 align

```json
{
  "schema_version": 1,
  "ok": true,
  "command": "align",
  "input": {"source": "a.wav", "duration_s": 12.34},
  "language": "Chinese",
  "result": {
    "items": [{"text": "甚", "start": 0.4, "end": 0.72}],
    "coverage": 0.98,
    "warnings": []
  },
  "metrics": {"load_ms": 0, "infer_ms": 200}
}
```

### 6.6 dry-run / doctor / languages / models

- `dry-run`：`{schema_version, ok, command, plan:{inputs:[], model:{}, backend, device, dtype, outputs:[]}}`。
- `doctor`：`{schema_version, ok, status, checks:[{name,status,value,detail,hint}]}`。
- `languages`：`{schema_version, ok, forceable:[{name,iso,aliases}], lid_labels:[...], alignable:[...], dialects:[...]}`。
- `models list`：`{schema_version, ok, models:[{alias,repo,path,present,size_bytes,fingerprint,capabilities}]}`。

所有 schema 存 `schemas/*.json`，CI 用 `jsonschema` 做快照校验。

### 6.7 退出码（沿用 qtts）

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | 参数、配置、manifest schema 错误 |
| 3 | 模型/运行时/后端/下载/OOM |
| 4 | 输入、音频、路径、能力不支持 |
| 5 | 对齐覆盖率/字幕质量门禁失败 |
| 6 | 批量部分失败 |
| 130 | SIGINT/SIGTERM |

### 6.8 字幕规则

- 内部统一整数毫秒；aligner 秒值只在入口转换一次。
- SRT `HH:MM:SS,mmm`；VTT `HH:MM:SS.mmm` + `WEBVTT` 头。
- half-up 舍入；保证 `end >= start + 1ms`、单调、不越过音频时长。
- cue 断句：优先标点（`。！？!?；;`），其次 ≥700ms 停顿；单 cue **≤7s 且 ≤42 显示宽度**（CJK 计 2）；英文不拆单词。
- `--granularity`：
  - `word`：aligner 原样 items（中文按字、英文按词、日/韩按各自分词）。
  - `char`：CJK token 拆字并按比例分配时间；拉丁 token 拆字符（近似）。文档标注"近似"。
  - `sentence`：按标点/停顿聚合成 cue；仅影响字幕与 JSON segments。
- 无时间戳却请求 `srt/vtt` → capability error，提示 `--timestamps` 或先 `qasr align`。

---

## 7. 批处理、作业与 resume 契约

### 7.1 manifest（JSONL，严格）

每行一个对象，未知字段报错：

```json
{"id":"a1","audio":"/data/a.wav","language":"zh","context":"交易 停滞","formats":["srt","json"],"output":"a1","timestamps":true,"granularity":"sentence"}
{"id":"a2","audio":"https://example.com/b.mp3","language":"auto","formats":["txt"]}
{"id":"a3","audio_base64":"data:audio/wav;base64,...","language":"en"}
```

规则：

- `audio` 与 `audio_base64` 二选一，必填。
- `id` 可选；默认 `sha256(audio 字符串)[:12]`；必须唯一，重复 → 退出 2。
- `output` 相对 `--out-dir`；拒绝绝对路径、`..`、NUL；默认等于 sanitized `id`。
- 逐条可覆盖 `language`、`context`、`formats`、`timestamps`、`granularity`；CLI 提供默认值。
- `--manifest` 与位置参数互斥。
- `--out-dir` 下生成 `.qasr/{state.jsonl,manifest.json,lock}`。

### 7.2 状态机与断点续跑

- 状态：`pending | running | done | failed | skipped`。
- `state.jsonl` 每行：`schema_version,id,job_key,status,outputs[],error{},started_at,finished_at`。
- 原子写：同目录临时文件 + `fsync` + `rename`；`state.jsonl` 每次 append 并 `fsync`。
- 启动时获取 `.qasr/lock`（PID）；已有活跃锁 → 退出 2。
- 启动时把遗留 `running` 恢复为 `pending`。
- `job_key = sha256(canonical(request) + audio_sha256 + model_fingerprint + aligner_fingerprint + backend + qasr_version + output_rel)`。
- 优先级：
  1. `--overwrite`：总是重跑，成功后原子替换旧输出；
  2. `--resume`：`done` 且 key 一致、输出存在且 SHA 一致 → `skipped`；否则重跑；
  3. 默认：重跑。
- 失败时保留旧输出；只有成功才 rename 覆盖。
- 部分失败退出 6；全部失败取首个错误码。

### 7.3 并发

- 本地 `--jobs` 强制 1（多进程复制权重，24GB 易 OOM）；用户传 >1 时 warning。
- 本地并发靠 `--batch-size` 微批（§4.1）。
- 远程 `--jobs` 默认 4，线程池；`--fail-fast` 时取消未开始任务。

### 7.4 输入与输出路径

- 输入展开稳定排序、绝对路径去重；目录只扫 wav/mp3/flac/m4a/ogg/opus/webm/mp4；符号链接不递归防环。
- 目录输入镜像相对路径到 `--out-dir`；拒绝逃逸。
- URL 先下载到 `QASR_HOME/cache/url/<sha256>.<ext>`（超时 60s、最大 2GB、可 `--no-cache`），再按本地文件处理；job key 用内容 SHA。
- 输出文件名：`<sanitized id>.<ext>`；冲突时默认报错，`--overwrite` 才覆盖；manifest 可显式 `output` 区分。

---

## 8. 长音频与对齐

### P0（明确边界）

- 转写：交给库内切块。**ASR 目标 1200s，实际可能因 ±5s 能量边界搜索略超；时间戳目标 180s，实际可能到 ~185s。** qasr 不承诺硬上限，文档写明。
- `transcribe --timestamps`：**必须显式指定 11 种对齐语言之一**。`--language auto --timestamps` → 退出 4，提示先跑无时间戳转写拿到语言，或直接指定。原因：库内把每 chunk 的 LID 结果直接传给 aligner，不校验；qasr 拿不到 per-chunk 语言，无法安全补救。
- `align`：单窗 ≤180s；超过 → 退出 4，提示 P1 的 `--window-seconds`。
- 英文跨 chunk 粘词是已知风险：库内 `''.join()`，qasr 无法定位边界；P0 文档声明，P1 自管切块修复。

### P1

- `long_audio.py`：纯函数窗口规划（核心区 + 重叠上下文，静音边界），每窗一次 ASR 得局部文本，单调匹配把参考文本归属到窗口，再 `align()`，最后合并并强制时间单调。
- 参考 qtts `long_align.py`（默认 270s 窗口、3s 重叠、覆盖率 0.5），但不 import qtts。
- 转写侧 `--max-chunk-seconds` 自管切块，用于进度、内存上限与断点续跑；切块后硬裁剪。

---

## 9. 打包、安装与发布

### 9.1 依赖分层（采纳审核意见，base 保持轻量）

```toml
[project]
name = "qasr-cli"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = ["numpy>=1.24", "soundfile>=0.12"]

[project.optional-dependencies]
local = ["qwen-asr==0.0.6", "torch>=2.4"]
vllm = ["qwen-asr[vllm]==0.0.6"]          # 内含 vllm==0.14.0
mic = ["sounddevice>=0.4"]
hf = ["huggingface_hub>=0.24"]
modelscope = ["modelscope>=1.15"]
all = ["qasr-cli[local,mic,hf,modelscope]"]
dev = ["pytest>=8", "ruff", "mypy", "jsonschema"]

[project.scripts]
qasr = "qasr_cli.cli:main"
```

安装矩阵：

```bash
uv tool install --python 3.12 "qasr-cli[local]"      # 主推：本地推理
uv tool install --python 3.12 "qasr-cli[vllm,mic]"   # Linux GPU：高吞吐 + 流式
uv tool install qasr-cli                              # 仅远程客户端/模型管理（轻量）
uv tool install --from . "qasr-cli[local,dev]"        # 开发
```

- base 不装 torch/qwen-asr；`qasr doctor` 在缺少运行时时给出精确修复命令。
- `uv tool install` 默认 Python 3.14 不满足 `<3.13`，文档必须写 `--python 3.12`（uv 会自动下载）。
- `qwen-asr` 会带入 `transformers==4.57.6`、gradio、flask；安装体积无法避免，qasr 不重复 pin transformers。
- 实测组合：qwen-asr 0.0.6 / transformers 4.57.6 / torch 2.15.0 / Python 3.12 / macOS MPS。
- `torch>=2.4` 不设上界，避免挡住各平台 wheel；CI 记录实测版本，升级 torch 需重跑 1.7B MPS 与 Linux vLLM 集成测试。
- vLLM 使用 `qwen-asr[vllm]==0.0.6` 钉的 0.14.0，不追 nightly；`[vllm]` 已隐含本地运行所需依赖；Linux CI 验证。
- TestPyPI 发布：`--index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple`，否则依赖解析不到。

### 9.2 模型目录

模型绝不放进 uv tool 的 venv（upgrade 会丢），统一放 `QASR_HOME/models`：

```
QASR_HOME（默认平台数据目录）
  models/
  cache/
  config.json
  jobs/
macOS: ~/Library/Application Support/qasr
Linux: $XDG_DATA_HOME/qasr 或 ~/.local/share/qasr
```

`models pull` 写 `source.json`（repo、revision、文件清单、哈希），支持 `--resume`。指纹：小文件内容 SHA-256；大权重记录文件名/大小/首尾 1MiB 哈希；`--full` 才全量。

**完整性校验（P0 必须）**：本机实测 `Qwen3-ASR-1.7B/` 是未下完的半包——残留 `model-00001-of-00002.safetensors.aria2`，`AutoModel.from_pretrained` 报 `SafetensorError: incomplete metadata`。这类失败必须在加载前被 `models verify` / `doctor` 拦住：

- 检查 `.aria2`、`.incomplete`、`.part` 残留；
- 按 `model.safetensors.index.json` 校验每个 shard 是否存在、大小是否匹配；
- 尝试解析每个 safetensors 头部（不加载权重）；
- 输出 `complete | incomplete | missing`，并给出 `qasr models pull <alias> --resume`。

发布：`uv build` + PyPI trusted publishing，先 TestPyPI；CI 在干净 Python 3.12 环境验证 `uv tool install --from . "qasr-cli[local]"`、`qasr --version`、`qasr doctor --json`。

---

## 10. qwen-asr 的坑与对策（已按审核修正）

| 坑 | 事实 | qasr 对策 |
|---|---|---|
| **180s vs README 5min** | `MAX_FORCE_ALIGN_INPUT_SECONDS=180`；README 写 5 分钟 | 文档写 180s；`align` P0 也按 180s 拒绝超限 |
| **切块不是硬上限** | ±5s 能量边界搜索，时间戳 chunk 可达 ~185s | 文档写"目标值"；P1 自管切块硬裁剪 |
| **时间戳路径不校验语言** | LID 结果直接传给 aligner；非 11 语言会静默产生错误时间 | P0 要求 `--timestamps` 显式 11 语言之一；`auto` 直接拒绝 |
| **时间戳拼接无法补救** | 库内 `''.join()`，无 chunk 边界/warning | P0 文档声明英文边界风险；P1 自管切块 |
| **`transcribe(list)` 全解码** | 先 `normalize_audios` 再切块；`max_inference_batch_size` 不控输入内存 | qasr 逐条校验 + 微批 + 组失败降级逐条 + 逐条错误隔离 |
| **强制语种是标签覆盖** | prompt 追加 `language X<asr_text>`；forced 时直接返回用户语言 | 文档写"提示/覆盖"；清理残留 tag；`language_source` 区分 |
| **30 语言 ≠ 52** | `SUPPORTED_LANGUAGES` 只有 30；方言只在 README | 三张表分开；`--language` 只校验 30；JSON `language` 原样输出 |
| **默认 `max_new_tokens=512`** | transformers 512；vLLM 4096；1200s chunk 必截断 | `auto = min(16384, max(1024, ceil(duration_s*8)+256))`；文档说明极端长音频仍建议切分；不承诺截断 warning（高层 API 拿不到 finish_reason） |
| **流式 O(n²)** | 每 chunk 重喂 `audio_accum`；vLLM-only；不支持 batch/时间戳 | `--max-stream-seconds` 默认 3600；`--chunk-ms`；macOS 拒绝 |
| **流式 finish 回滚不同** | `max(1, len-k)` 且无替换字符检查 | 尾包单独测试 |
| **复读修复会改文本** | `detect_and_fix_repetitions(threshold=20)` | 记录 `text_transform`，可追溯 |
| **依赖钉死** | transformers 4.57.6、vllm 0.14.0、Python <3.13 | `doctor` 一等检查；安装文档写 `--python 3.12` |
| **MPS 算子兼容** | bf16 不稳 | 默认 fp16；`--dtype fp32` 兜底 |
| **vLLM spawn** | 多进程可能重复执行入口 | serve 用 exec 转发；入口 guard；Linux CI 验证 |
| **远程接口未实测** | `/v1/audio/transcriptions` 的 language/context 行为需实机确认 | 先做 P1；失败时降级/报错，不假设 |
| **本地模型可能是半包** | 实测 1.7B 残留 `.aria2`，safetensors 头损坏 | `models verify`/`doctor` 加载前检查 shard/`.aria2`/头；`pull --resume` |

---

## 11. 测试与 CI

| 层 | 内容 | 依赖 |
|---|---|---|
| 单元 | 参数解析、三张语言表、模型别名、配置优先级、输入展开、manifest、resume key、SRT/VTT 舍入、cue 切分、路径安全、窗口规划 | 纯 Python |
| 契约 | 所有 `schemas/*.json` 快照、错误→退出码、stdout 纯净性 | 纯 Python |
| 模拟 | `FakeBackend` 注入确定性文本/时间戳；本地 HTTP mock 测 remote；崩溃注入测 resume | 纯 Python |
| 集成 | macOS 0.6B+MPS 冒烟；**1.7B+aligner MPS 内存/可行性**；短音频时间戳；Linux NVIDIA vLLM batch/流式/显存降级；11 语言对齐 | `RUN_INTEGRATION=1` + 模型 |

CI：Linux + macOS 跑纯 CPU 测试，不下载模型；`ruff` + `mypy`（核心无运行时模块）+ `pytest`；构建 wheel 后在干净环境 `uv tool install --from . "qasr-cli[local]"` 并跑 `qasr doctor --json`。额外测试第三方库 stdout 污染（导入/加载模型时）。

---

## 12. 实现波次与功能组合

> **P0/P1/P2 是按依赖与功能内聚划分的实现波次，不是发布节奏。** 三波连续实现、连续交付；每一波结束都能独立验收，不做"先发一个残缺版本再补"的取舍。

### 12.1 功能组合总览

| 功能组 | 内容 | 波次 |
|---|---|---|
| 核心识别 | transcribe、LID、context、语言、时间戳（显式语言） | P0 |
| 对齐与字幕 | align、cue 切分、SRT/VTT、覆盖率门禁 | P0 |
| 模型与诊断 | models list/pull/path/verify、languages、doctor、config show/path | P0 |
| 输出与批处理 | JSON schemas、manifest、state、resume、lock、原子写、逐条错误隔离 | P0 |
| 运行时扩展 | vLLM 离线、serve、remote 客户端、doctor --server | P1 |
| 长音频 | 时间戳/对齐窗口化、ASR 锚定、单调合并 | P1 |
| 高级批处理 | --stdin-list、--skip-existing、--retry、tsv/md、jobs、bench、models show/rm、config get/set | P1 |
| 流式 | stream（vLLM）、mic、累计快照、尾包 | P2 |
| 生态与平台 | finetune lint、completion、多 GPU 分片、视频抽轨、watch、Windows | P2 |

### 12.2 P0 — 单机核心闭环

**目标**：一个人在 Mac/CPU/单卡机器上，装完就能可靠地转写、对齐、批量、续跑。

范围：

1. `transcribe`：单/多文件、目录、glob、URL、`--manifest`；txt/json/jsonl/srt/vtt；`--context`/`--language`。
2. `--timestamps`（显式 11 语言）；`align`（显式语言，≤180s）。
3. `models list/pull/path/verify`、`languages`、`doctor`、`config show/path`。
4. transformers 后端（CUDA/MPS/CPU）、模型解析、离线模式、微批与逐条错误隔离。
5. `state.jsonl` + `--resume` + 原子写 + lock + 稳定 JSON/退出码。
6. 单元/契约/模拟测试 + macOS/Linux CPU CI + 1.7B MPS 集成测试。

**验收**：干净环境 `uv tool install --python 3.12 "qasr-cli[local]"` → `qasr doctor` 全绿 → `qasr models pull 1.7b` → 转写一个目录并 `--resume` 续跑 → JSON/SRT 通过 schema 校验；半包模型被 `verify` 拦住并给出 `pull --resume`。

### 12.3 P1 — 批量、部署与长音频

**目标**：跨机器、长时间、可运维。

范围：

1. `serve`（薄包装 `qwen-asr-serve`）+ `--server` 远程客户端 + `doctor --server`。
2. vLLM 离线批量后端；远程并发 `--jobs`；远程 context 走 chat completions。
3. 长音频时间戳/对齐窗口化（180s/270s 窗口 + ASR 锚定 + 单调合并）。
4. `models show/rm`、`config get/set`、`--stdin-list`、`--skip-existing`、`--retry`、tsv/md、`jobs list/show`、`bench`。
5. `align --language auto`（脚本判定 zh/ja/ko/ru，Latin 要求显式）。

**验收**：Mac 客户端通过 `--server` 转写 GPU 服务器上的 1 小时音频；带时间戳的 1 小时音频窗口化后时间单调、覆盖率达标；kill -9 后 `--resume` 只重跑未完成项；远程 4xx/5xx/超时映射到正确退出码。

### 12.4 P2 — 流式与生态

**目标**：实时输入与周边工作流。

范围：

1. `stream`（vLLM/CUDA，累计快照）+ `--mic` + 尾包测试。
2. `finetune lint-jsonl`（只校验/规范化 SFT 数据，不训练）。
3. `completion bash|zsh|fish`。
4. 多 GPU 分片（每卡一个子进程，`CUDA_VISIBLE_DEVICES` 隔离）。
5. 视频容器直接输入（ffmpeg 抽轨）、目录 watch、Windows 支持。

**验收**：Linux+CUDA 上麦克风实时出累计文本；`qasr stream --emit jsonl` 的 `seq` 单调、`is_final` 只出现一次；多 GPU 分片结果与单卡一致；Windows 至少通过 `qasr doctor` 与 CPU 转写。

### 12.5 明确不做（与波次无关）

训练框架、数据管线、说话人分离、翻译、原生异步 server、插件系统、默认结果缓存。这些不因为"三波全做"而进入范围。

---

## 13. 审核记录与修订

### Partner A（共同构思）主要贡献

- 定位为"脚本化 ASR 运行时 + 批处理前端"，不设独立 `batch` 子命令，避免双执行器。
- `QASR_HOME` 模型目录、`state.jsonl` + `--resume`、job key、原子写。
- 流式输出用**累计快照**而非 delta。
- 列出 qwen-asr 的 12 个坑（180s、切块超限、拼词、LID、对齐语言、512 token、全解码、流式 O(n²) 等）。

### Partner B（独立审核）结论：PASS WITH CHANGES

已修入本版的阻塞/重要项：

1. **BLOCKER**：`--timestamps` + auto LID 无法通过高层 API 做 per-chunk 校验 → P0 强制显式 11 语言。
2. **BLOCKER**：无法在库外补空格/记 chunk warning → 删除该承诺，P0 声明风险，P1 自管切块。
3. 180s/300s 矛盾 → P0 `align` 统一 ≤180s，P1 窗口化。
4. `--batch-size` 不能控内存/错误隔离 → 逐条校验 + 微批 + 组失败降级逐条。
5. 强制语种语义夸大 → 改为"prompt 偏置 + 标签覆盖"。
6. `align auto` 误判 → Latin 等脚本要求显式语言。
7. 30 语言 vs 52 → 三张表分开。
8. `max_new_tokens auto` 无定义 → 给出公式，删除不可实现的截断 warning。
9. manifest/resume 契约缺失 → 补完整 schema、状态机、优先级、锁、原子写、路径安全。
10. 输出契约只覆盖 transcribe → 补 align/dry-run/doctor/languages/models schema。
11. 批处理路径/id 不安全 → sanitize、拒绝 `..`、冲突检测、内容哈希。
12. 安装风险 → base 改轻量，`[local]`/`[vllm]` 分层，写 `--python 3.12`，补 1.7B MPS 测试与 TestPyPI 说明。

---

## 14. 已确认决策

1. **安装策略**：base 轻量（numpy + soundfile），本地推理走 `[local]` extra；主推 `uv tool install --python 3.12 "qasr-cli[local]"`。
2. **不设 `batch` 子命令**：批量统一走 `transcribe --manifest/--out-dir`；以后若有强需求再加 `qasr batch` 别名。
3. **远程模式**：固定在 P1 的运行时/部署功能组实现，P0 不涉及。原因见 §5.1：
   - 有 GPU 但就在 GPU 机器上跑 → 本地 vLLM，更简单；
   - CLI 机器 ≠ GPU 机器（Mac 调 Linux box）、或多客户端共享 → 远程；
   - 没有 GPU → 本地 transformers + MPS/CPU 已够用。
4. **默认模型 1.7B**：质量优先；`--model 0.6b` 为快速模式。实测 0.6B 在 M4 Pro 上稳态 RTF 0.056；1.7B 本机因半包未能实测，实施前先 `qasr models pull 1.7b --resume`。绝不静默降级。
5. **时间戳强制显式语言**：P0 只接受 11 种对齐语言之一；`--language auto --timestamps` 退出 4。P1 加"前 30s 探测 + 强制"的 auto 路径。

### 残余开放项（不阻塞设计）

- 1.7B 本机目录是半包（`.aria2` + safetensors 头损坏），实施第一步先修复/重下；`models verify` 已列为 P0。
- 远程 `/v1/audio/transcriptions` 的 language/context 行为需实机验证；P1 实现时用 `doctor --server` + 失败降级兜底。
- 是否把 DashScope API 作为第四种 backend：当前不在范围，若需要再单独立项。
