# qasr-cli P0 实现任务书

> 依据：`DESIGN.md` v0.3。P0 只做**单机核心闭环**；`serve/remote/vLLM/stream/长音频窗口化` 属 P1/P2，本阶段不实现。
> 每一波（M1–M4）结束都要能独立验收；M4 完成后交给独立 Codex 审查/测试。

---

## 0. P0 目标

在 Mac（MPS）和 Linux（CPU/CUDA）上，用户执行：

```bash
uv tool install --python 3.12 "qasr-cli[local]"
qasr doctor
qasr models pull 1.7b
qasr transcribe ./audio/ --recursive --format txt,json,srt --timestamps --language zh --out-dir out --resume
qasr align a.wav --text-file transcript.txt --language zh --format srt,json
```

即可完成转写、时间戳、独立对齐、批量、断点续跑，并得到稳定 JSON/SRT 输出。

**P0 不包含**：`serve`、`--server`、vLLM 后端、`stream`、`>180s` 的对齐窗口化、`--stdin-list`、`--skip-existing`、`--retry`、`tsv/md`、`models show/rm`、`config get/set`、`jobs list/show`、`bench`、`completion`、`finetune`、多 GPU、视频抽轨、watch、Windows。

---

## 1. 工程约束

- Python `>=3.10,<3.13`；开发与 CI 用 **3.12**。
- 包名 `qasr-cli`，模块 `qasr_cli`，入口 `qasr = qasr_cli.cli:main`。
- 使用 **uv** 管理环境与依赖；不要用 pip 手工装到系统 Python。
- base 依赖只允许 `numpy>=1.24`、`soundfile>=0.12`；`qwen-asr`/`torch` 放 `[local]` extra；测试依赖放 `[dev]`。
- CLI 用 **argparse**；不要引入 Typer/Click。
- 允许新增小型纯 Python 依赖，但必须先说明理由；默认不新增。
- 所有 `--json` 输出存 `schemas/*.json`，测试用 `jsonschema` 校验。
- 不修改 `DESIGN.md`；发现矛盾先报告。

### 开发环境

```bash
cd /Users/walter/workspace/QwenLM/qwen-asr-cli
uv venv --python 3.12 .venv
uv pip install -e ".[local,dev]"
```

若下载 torch 太慢，可复用已装好运行时的解释器做集成测试：

```bash
/Users/walter/workspace/QwenLM/qwen-tts-cli/.venv/bin/python -m pytest ...
```

但**单元测试必须能在没有 torch 的环境跑通**。

### 本地模型

```
/Users/walter/workspace/QwenLM/Qwen3-ASR/models/Qwen3-ASR-0.6B          # 完整
/Users/walter/workspace/QwenLM/Qwen3-ASR/models/Qwen3-ForcedAligner-0.6B # 完整
/Users/walter/workspace/QwenLM/Qwen3-ASR/models/Qwen3-ASR-1.7B          # 半包！用于验证 models verify
```

集成测试用 `QASR_TEST_MODEL_DIR` 指向上面目录；不要提交模型或音频。

---

## 2. 交付结构

```
qasr_cli/
  __init__.py          # __version__
  __main__.py          # python -m qasr_cli
  cli.py               # argparse 树、全局 flag、异常→退出码
  errors.py            # QasrError / UsageError / InputError / ModelError / CapabilityError
  config.py            # 配置合并、QASR_HOME、平台目录
  languages.py         # 30 可强制 / 52 LID / 11 对齐 + ISO/别名
  catalog.py           # 模型别名、路径解析、verify、pull、fingerprint
  inputs.py            # 文件/目录/glob/URL、sanitize、稳定排序、去重
  audio.py             # 时长/采样率/格式探测；不 import torch
  runtime.py           # 懒加载 qwen_asr/torch、能力探测
  backends.py          # Backend 协议 + TransformersBackend
  transcribe.py        # 编排：输入 → 微批 → 结果 → 输出
  align.py             # 已知文本对齐
  subtitles.py         # cue 构建、SRT/VTT、舍入（纯 Python）
  render.py            # txt/json/jsonl（纯 Python）
  jobs.py              # manifest、state.jsonl、resume、lock、原子写
  doctor.py
  commands/            # transcribe/align/models/languages/doctor/config
schemas/               # transcribe-single/batch/error/align/doctor/languages/models/dry-run
tests/
  unit/                # 纯 Python
  contract/            # schema/退出码/stdout 纯净性
  integration/         # 需要模型，RUN_INTEGRATION=1
pyproject.toml
README.md
```

**硬约束**：`cli.py`、`config.py`、`languages.py`、`catalog.py`、`inputs.py`、`audio.py`、`subtitles.py`、`render.py`、`jobs.py`、`doctor.py` 不得 import `torch`/`qwen_asr`/`vllm`。

---

## 3. 里程碑与任务

### M1 — 骨架、契约、模型与诊断

**T1.1 项目骨架**
- `pyproject.toml`：name/version/requires-python/依赖/extras/entry point/setuptools 配置。
- `qasr_cli/__init__.py`、`__main__.py`、`cli.py`、`errors.py`。
- `qasr --version`、`qasr --help` 在**无 torch** 环境可运行；启动不 import torch。
- `errors.py` 实现退出码 0/2/3/4/5/6/130 与 `to_dict()`。

**T1.2 配置**
- `QASR_HOME`：macOS `~/Library/Application Support/qasr`，Linux `$XDG_DATA_HOME/qasr` 或 `~/.local/share/qasr`。
- 配置优先级：内置默认 < `QASR_HOME/config.json` < 项目 `.qasr/config.json`（向上查找）< 环境变量 < `--config` < CLI。
- `qasr config show [--json]`、`qasr config path`。
- 环境变量：`QASR_MODEL`、`QASR_MODEL_DIR`、`QASR_BACKEND`、`QASR_DEVICE`、`QASR_DTYPE`、`QASR_OFFLINE`、`QASR_CONFIG`。

**T1.3 三张语言表**
- 30 可强制语言 + ISO 码 + 别名（zh/en/yue/.../mk）。
- 22 方言（仅 LID 展示，不可强制）。
- 11 对齐语言。
- `--language` 校验只针对 30；方言名 → 退出 4，提示用 `auto`。
- `qasr languages [--json]`。

**T1.4 模型目录与校验**
- 别名 `1.7b`/`0.6b`/`aligner` → repo。
- 解析顺序：`--model` 路径/repo → env → 项目配置 → `QASR_HOME/models` → `./models`/`../Qwen3-ASR/models`（存在时）→ 在线。
- `qasr models list/path/verify`；`pull` 可用 `--source hf|modelscope`，无依赖时给安装提示。
- `verify` 必须检查：`.aria2/.incomplete/.part` 残留、`model.safetensors.index.json` shard 存在与大小、safetensors 头可解析；输出 `complete|incomplete|missing`；半包给 `pull --resume`。
- 指纹：小文件 sha256；大权重 size+首尾 1MiB；结果缓存到 `QASR_HOME/cache/fingerprints.json`。

**T1.5 doctor**
- 检查 Python/qasr/torch/qwen-asr/transformers/vllm、设备（cuda/mps/cpu）、内存/显存、模型状态、磁盘、soundfile/ffmpeg。
- `--json` schema；`ok|warn|fail`；默认退出 0，`--strict` 在 warn/fail 时退出 3。

**M1 验收**
- 无 torch 环境：`qasr --help`、`qasr --version`、`qasr languages --json`、`qasr config show --json`、`qasr models verify 1.7b --json` 均通过。
- `models verify 1.7b` 正确报 `incomplete`；`0.6b` 报 `complete`。
- `doctor --json` 通过 schema。

---

### M2 — 输入、运行时、基础转写

**T2.1 输入展开**
- 文件、目录（`--recursive`）、glob（由 shell 展开，qasr 也处理字面 `*`）、URL。
- 目录只扫 wav/mp3/flac/m4a/ogg/opus/webm/mp4；符号链接不递归防环。
- 稳定排序、绝对路径去重、`id` 生成；拒绝 `..` 逃逸。
- URL 下载到 `QASR_HOME/cache/url/<sha256>.<ext>`，超时 60s，最大 2GB；`--offline` 禁止未缓存 URL。

**T2.2 音频探测**
- `audio.py` 用 soundfile 读时长/采样率/声道；损坏文件给 `InputError`。
- 不 import torch。

**T2.3 运行时**
- `runtime.py` 懒 import `qwen_asr`/`torch`；`--offline` 在 import 前设置 `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`，并传 `local_files_only=True`。
- `backends.py` 定义 `Capabilities`、`Backend` 协议、`TransformersBackend`。
- 设备/dtype：cuda→bf16、mps→fp16、cpu→fp32；`--dtype` 覆盖。
- 模型构造：`from_pretrained(path, dtype=..., device_map=..., max_inference_batch_size=..., max_new_tokens=auto)`。
- `max_new_tokens auto = min(16384, max(1024, ceil(duration_s*8)+256))`，按本次最大音频时长计算。
- 模型缓存：进程内按 `(path, device, dtype, aligner_path)` 复用；纯文本转写不加载 aligner。

**T2.4 微批与错误隔离**
- 逐条校验后再分组；默认 local `batch-size=1`。
- 组失败 → 逐条重试定位；失败项写 error，继续其他组。
- `--max-inference-batch-size` 单独传给模型构造。

**T2.5 transcribe（基础）**
- `qasr transcribe INPUT...`；`--language`、`--context`/`--context-file`、`--model`、`--aligner`、`--batch-size`、`--max-new-tokens`、`--out-dir`、`--output`、`--format txt,json,jsonl`、`--recursive`、`--offline`、`--dry-run`、`--json`、`--jsonl`、`--quiet`。
- `--json` 单条 schema；`--jsonl` 每行一个 item；批量 `--json` 用一个 envelope。
- stdout 只放结果；所有第三方输出走 stderr（`stdout_to_stderr`）。
- 空音频输出空文本、`language: null`，不算失败。

**M2 验收**
- `qasr transcribe /tmp/qasr_asr_zh.wav --language zh --json` 返回正确中文 text/language。
- `qasr transcribe` 多文件 → jsonl 每行一个结果。
- 一个坏文件 + 一个好文件 → 好文件成功、坏文件 error，退出 6。
- `qasr --help` 仍不 import torch。

---

### M3 — 时间戳、字幕、对齐

**T3.1 时间戳**
- `--timestamps` 仅接受显式 11 对齐语言；`--language auto --timestamps` → 退出 4，提示。
- 构造模型时传 `forced_aligner` 与 `forced_aligner_kwargs`。
- 结果 `segments` 输出 `{text,start,end,kind}`；`timestamp_source: forced_aligner`。
- 清理 text 中残留 `<asr_text>` / `language X` 前缀。

**T3.2 字幕**
- `subtitles.py`：cue 切分（标点优先、≥700ms 停顿）、`≤7s` **且** `≤42` 显示宽度（CJK 计 2）、英文不拆词。
- SRT `HH:MM:SS,mmm`；VTT `HH:MM:SS.mmm` + `WEBVTT`。
- 内部整数毫秒；half-up；`end >= start+1ms`、单调、不超音频时长。
- `--granularity sentence|word|char`；`word` 原样 items，`char` 拆 CJK 并按比例分时，`sentence` 聚合 cue。
- 无时间戳请求 srt/vtt → capability error。

**T3.3 align**
- `qasr align AUDIO (--text | --text-file | --srt)`。
- `--language` 必填且为 11 语言之一；`auto` → 脚本判定 zh/ja/ko/ru，Latin 等报错（P1 才放宽）。
- 单窗 ≤180s；超限退出 4。
- `--format json,srt,vtt`；`--output/--out-dir`。
- `--coverage-min 0.5`：规范化后单调匹配，低于阈值退出 5；`0` 关闭。
- `--timestamps` 与 `align` 都不允许 `--server`（P0 无 remote，直接不实现该 flag）。

**M3 验收**
- `qasr transcribe /tmp/qasr_asr_zh.wav --timestamps --language zh -f srt -o out/` 生成合法 SRT。
- `qasr align /tmp/qasr_asr_zh.wav --text "甚至出现交易几乎停滞的情况。" --language zh --format json` 返回 13 个 items，时间单调。
- 覆盖率低于阈值 → 退出 5。
- SRT/VTT golden 测试通过。

---

### M4 — 批处理、恢复、文档、收尾

**T4.1 manifest**
- JSONL 严格 schema：`id/audio/audio_base64/language/context/formats/output/timestamps/granularity`；未知字段报错。
- `audio` XOR `audio_base64`；`id` 唯一；`output` 相对 out-dir、拒绝绝对路径/`..`/NUL。
- `--manifest` 与位置参数互斥。

**T4.2 state/resume/lock**
- `state.jsonl` 字段与状态机按 DESIGN §7.2。
- 原子写（temp+fsync+rename）、输出 SHA、`.qasr/lock`。
- `--resume`：`done` + key 一致 + 输出存在/SHA 一致 → skip；否则重跑。
- `--overwrite` 总是重跑；`running` 启动时恢复 `pending`。
- 部分失败退出 6；全部失败取首个错误码。

**T4.3 输出命名**
- 单输入：`<sanitized id>.<ext>`；目录输入镜像相对路径；URL 用内容哈希。
- 冲突默认报错，`--overwrite` 覆盖；多格式 + stdout → 用法错误。

**T4.4 README 与 schema**
- README：安装（`--python 3.12`）、模型准备、5 个 quickstart 示例、输出/退出码、已知限制（180s、方言、英文边界、半包）。
- `schemas/*.json` 完整；contract 测试校验。

**T4.5 自测**
- `uv run pytest` 全绿（纯 CPU）。
- `RUN_INTEGRATION=1` + `QASR_TEST_MODEL_DIR` 跑 0.6B 集成测试。
- `uv tool install --from . "qasr-cli[local]"` 后 `qasr --version`、`qasr doctor --json`。
- 提交每个里程碑，最后提交 P0。

**M4 验收**
- 干净环境安装成功；`qasr doctor` 全绿。
- `qasr transcribe ./samples --recursive --format txt,json,srt --out-dir out --resume` 可重复执行；kill -9 后续跑只处理未完成项。
- `qasr models verify 1.7b` 正确拦截半包。

---

## 4. 关键契约速查

### 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | 参数/配置/manifest schema |
| 3 | 模型/运行时/下载/OOM |
| 4 | 输入/音频/路径/能力不支持 |
| 5 | 对齐覆盖率/字幕质量门禁 |
| 6 | 批量部分失败 |
| 130 | SIGINT/SIGTERM |

### JSON envelope

```json
{"schema_version": 1, "ok": true, "command": "transcribe"}
```

错误：`{schema_version, ok:false, command, error:{kind,code,message,hint,retryable,details}}`。

### stdout/stderr

- stdout：结果（文本/JSON/字幕）。
- stderr：进度、warning、第三方日志。
- `--json` 与 `--jsonl` 互斥；`--format json` 是文件输出，与 `--json` 互斥。

### 语言

- 可强制：30；方言：22（仅展示）；对齐：11。
- 强制语言是 prompt 偏置 + 标签覆盖，不是识别保证。

---

## 5. 测试要求

| 层 | 必须覆盖 |
|---|---|
| 单元 | 参数解析、三张语言表、配置优先级、模型别名/verify、输入展开/路径安全、cue 切分/舍入、job key/resume、JSON 渲染 |
| 契约 | 所有 schema 快照、错误→退出码、stdout 纯净性、`--help` 不 import torch |
| 模拟 | `FakeBackend` 注入固定文本/时间戳；坏文件隔离；崩溃注入 resume |
| 集成 | 0.6B 短音频 text/json/srt；align 13 items；1.7B verify 半包；`RUN_INTEGRATION=1` 才跑 |

音频 fixture：单元测试用代码生成的 1s 静音/正弦 WAV；集成测试用 `/tmp/qasr_asr_zh.wav`（若存在）或 `QASR_TEST_AUDIO`。

---

## 6. 提交与报告

- 每个里程碑一次 commit，Conventional Commits。
- 最终报告必须包含：
  1. 完成/未完成任务对照；
  2. 测试命令与结果（含失败输出）；
  3. 已知限制与偏差；
  4. 关键文件路径；
  5. 供 Codex 审查的入口（如何跑、看哪些文件）。
- 不修改 `DESIGN.md`；不实现 P1/P2 内容。
