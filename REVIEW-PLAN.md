# qasr-cli P0 独立审查与测试计划

> 审查者：Codex Partner（独立于实现）
> 依据：`DESIGN.md` v0.3、`P0-TASKS.md`
> 阶段：Phase 1 只产出计划与黑盒测试骨架；Phase 2 才读实现、跑测试、出报告。
> 被测对象：`qasr` CLI（`qasr_cli.cli:main`），不接受“import 实现再断言”的白盒测试作为验收证据。

---

## 0. 结论口径（先看这里）

| 结论 | 含义 |
|---|---|
| **PASS** | P0 验收项全部满足；无 BLOCKER/MAJOR；对抗用例无数据损坏、路径逃逸、静默错误 |
| **PASS WITH CHANGES** | P0 主链路可用，但存在必须在本波次内修掉的 MAJOR，或验收项有带条件的偏差 |
| **FAIL** | 存在 BLOCKER：错误结果被当成功、路径逃逸、stdout 污染、半包模型绕过校验、断点续跑丢/错数据、退出码契约破坏 |

判定不看“测试数量”，只看：**契约是否稳定、失败是否可诊断、坏输入是否被隔离、恢复是否可信**。

---

## 1. 审查范围与非范围

### 1.1 范围内（P0）

1. `transcribe`：单/多文件、目录、glob、URL、`--manifest`；`txt/json/jsonl/srt/vtt`；`--context`/`--context-file`；`--language`。
2. `--timestamps`：仅显式 11 种对齐语言；`--language auto` 必须拒绝。
3. `align`：`--text`/`--text-file`/`--srt`；显式语言；≤180s；覆盖率门禁。
4. `models list/pull/path/verify`、`languages`、`doctor`、`config show/path`。
5. transformers 后端（CUDA/MPS/CPU）、模型解析、离线模式、微批、逐条错误隔离。
6. `manifest` 严格校验、`state.jsonl`、`--resume`、`--overwrite`、原子写、lock、job key、输出 SHA。
7. JSON schema、退出码、stdout/stderr 分离、无 torch 环境启动。
8. 安装与文档：`uv tool install --python 3.12 "qasr-cli[local]"`、README 的 P0 限制声明。

### 1.2 非范围（P0 不判 FAIL，但“悄悄实现了一半”要记录）

- `serve`、`--server`、remote 客户端、vLLM 后端、`stream`、mic。
- `--stdin-list`、`--skip-existing`、`--retry`、`tsv/md`、`models show/rm`、`config get/set`、`jobs list/show`、`bench`、`completion`、`finetune`。
- 长音频自管切块、`--window-seconds`、`align --language auto` 的完整 P1 放宽（P0 只做脚本判定兜底）。
- 训练/翻译/说话人分离/GUI。

> P1/P2 命令或参数在 P0 出现时，预期是 `exit 2`（未知参数/命令），而不是半成品行为。若已实现但行为不完整，记为 MINOR/MAJOR 并注明“越界实现”。

---

## 2. 审查依据与冲突处理

优先级：`P0-TASKS.md` 的 P0 验收 > `DESIGN.md` v0.3 > `qwen-asr` 实际行为。

- `DESIGN.md` 与 `P0-TASKS.md` 冲突时，以 P0 任务书为准，但必须在报告中列出冲突。
- `DESIGN.md` 未写死的细节，不按“实现方便”倒推；列入 §8 歧义清单，Phase 2 与实现者确认后再定 PASS/FAIL。
- **Phase 1 协调补充（错误优先级）**：当请求同时存在“输入无效”和“模型不可用”时，必须先报输入错误（exit 4），不得让模型解析/加载错误（exit 3）覆盖它。该条已写入 `test_input_error_precedes_model_resolution`。
- **Phase 1 协调补充（输出通道与格式）**：多输入 + `--json`/`--jsonl` 是显式输出通道，合法并输出 batch envelope；只有完全没有 `--out-dir`/`--output`/`--json`/`--jsonl` 时才 exit 2。`--format json|jsonl` 与 `--json|--jsonl` **无条件互斥**，即使提供了 `--out-dir`。
- 上游事实以 `/Users/walter/workspace/QwenLM/Qwen3-ASR/` 为准：
  - `SUPPORTED_LANGUAGES` 30 项；
  - `MAX_FORCE_ALIGN_INPUT_SECONDS = 180`；
  - `MAX_ASR_INPUT_SECONDS = 1200`；
  - `parse_asr_output` 对 `language None<asr_text>` 返回空文本、空语言；
  - 强制语言是“prompt 偏置 + 标签覆盖”，不是识别保证。

---

## 3. 风险清单（按严重度）

### 3.1 BLOCKER（出现即 FAIL）

| ID | 风险 | 触发条件 | 影响 | 覆盖测试 |
|---|---|---|---|---|
| B1 | 半包/损坏模型在 `from_pretrained` 才爆炸 | 合成模型目录：残留 `.aria2`、索引 shard 缺失、safetensors 头截断 | 用户看到 `SafetensorError`/OOM/长时间卡住，而不是 `verify` 的可操作提示；P0 验收直接失败 | `MOD-04..08` |
| B2 | `--timestamps --language auto` 未被拒绝 | 默认语言 auto + timestamps | 库把 per-chunk LID 直接喂 aligner，非 11 语言时产生**错误但看似成功**的时间戳 | `TS-01`、`TS-02`、`MAN-11` |
| B3 | stdout 被日志/进度/第三方输出污染 | `--json`/`--jsonl` + 模型加载/警告 | 管道消费方 `json.loads(stdout)` 失败；批处理不可脚本化 | `PUR-01..06`、`INT-PUR-01` |
| B4 | resume 只按 id 判断，不校验 job key/输出 SHA | 改语言/模型/context 后 `--resume` | 静默复用旧结果，输出与请求不一致（数据正确性问题） | `INT-RES-02..03` |
| B5 | manifest `output` 路径逃逸 | `../x`、绝对路径、NUL、符号链接 | 写到 `--out-dir` 外，覆盖用户文件 | `MAN-04..08`、`ADV-10` |
| B6 | `kill -9` 后 lock 永久占用 | 上次进程被 SIGKILL，`.qasr/lock` 残留 | 后续 `--resume` 永远 exit 2，P0 的“可恢复”承诺失效 | `INT-RES-04`、`MAN-14` |
| B7 | >180s 的 `align` 未在加载模型前拒绝 | 181s 音频 | aligner 内部报错/截断/静默错误时间；退出码不是 4 | `AL-06..07`、`ADV-11` |
| B8 | 坏文件拖垮整批 | 一个坏 WAV + 多个好文件，`--batch-size>1` | 好文件也失败，或整批 exit 3/异常回溯 | `INT-BAT-01`、`TR-05` |
| B9 | 空音频被当成失败 | 合法 0 样本 WAV / 静音 | 与 DESIGN §3.3 “空文本、language null、不算失败”冲突 | `INT-EMP-01` |
| B10 | 解析失败仍返回 ok=true | 模型输出无 `<asr_text>` / 空串 / 异常标签 | 空文本被包装成成功，用户无法区分“静音”和“解析 bug” | `INT-TR-01`、`INT-EMP-01` |

### 3.2 MAJOR（不阻塞主链路，但 P0 内应修）

| ID | 风险 | 影响 | 覆盖测试 |
|---|---|---|---|
| M1 | 批量全部失败返回 6 而不是首个错误码 | 脚本无法区分“部分失败”和“全失败” | `PUR-04`、`MAN-13` |
| M2 | 方言名被 `--language` 接受 | 与“方言不可强制”契约冲突，输出语言标签不可信 | `LANG-04`、`LANG-06` |
| M3 | SRT/VTT 时间不单调、`end<start+1ms`、越过音频时长 | 播放器/质检失败 | `INT-SUB-01..03` |
| M4 | cue 超 7s 或 42 显示宽度；英文拆词 | 字幕不可读，验收 golden 失败 | `INT-SUB-02` |
| M5 | 强制语言残留 `language X` / `<asr_text>` 前缀 | 文本脏，下游 TTS/字幕错 | `INT-TR-02` |
| M6 | 配置优先级错误（env 覆盖 `--config`、项目配置不向上查找） | 用户显式参数被忽略 | `CFG-03..06` |
| M7 | 目录递归遇符号链接环不终止/重复 | 卡死、重复转写、内存上涨 | `ADV-09` |
| M8 | `--resume` 不跳过已完成项 | 1 小时目录全量重跑，失去 P0 核心价值 | `INT-RES-02` |
| M9 | 原子写不是 temp+fsync+rename | kill 时留下半截 JSON/SRT，resume 误判 | Phase 2 代码审查 + `INT-RES-04` |
| M10 | 纯文本转写也加载 aligner | 内存翻倍，24GB M1/M4 上 OOM 风险 | `INT-TR-03`（metrics/内存观察） |
| M11 | `--dry-run` 仍下载 URL / 加载模型 | 离线规划失败，dry-run 失去意义 | `TR-12`、`ADV-08` |
| M12 | doctor 默认退出码被 warn/fail 改变 | CI 误报；`--strict` 语义失效 | `DOC-02..04` |
| M13 | `--jobs>1` 本地未警告/未强制 1 | 多进程复制权重导致 OOM | `TR-11` |
| M14 | 错误 JSON 缺少 `kind/code/hint/retryable/details` | 机器调用方无法分支 | `PUR-02`、`PUR-06` |

### 3.3 MINOR

| ID | 风险 |
|---|---|
| m1 | `--version` 格式不稳定、无 `qasr` 前缀 |
| m2 | `--help` 文案缺 P0 子命令或退出码说明 |
| m3 | 未知语言退出码在 2/4 间摇摆 |
| m4 | `models path` 缺失时退出码不统一（3/4） |
| m5 | 空目录输入返回 0 且空结果 |
| m6 | schema 版本号不是整数 1 |
| m7 | `metrics.rtf` 缺失或分母用错 |
| m8 | stderr 中文/英文混排、warning 无前缀 |
| m9 | `--quiet` 未抑制非错误日志 |
| m10 | README 未声明 180s、方言不可强制、英文跨 chunk 粘词、半包 |

---

## 4. 黑盒测试矩阵

调用约定：所有测试通过 `subprocess` 执行 `qasr`（或 `python -m qasr_cli`），隔离 `QASR_HOME`，清空 `QASR_*` 环境变量，cwd 使用临时目录，避免命中开发期 `../Qwen3-ASR/models` 回退路径。

### 4.1 CLI 表面与启动（无 torch）

| ID | 命令 | 输入 | 期望 stdout | 期望 stderr | 退出码 | JSON 字段 |
|---|---|---|---|---|---|---|
| CLI-01 | `qasr --version` | — | 单行含版本号 | 空 | 0 | — |
| CLI-02 | `qasr --help` | — | 含 transcribe/align/models/languages/doctor/config | 空 | 0 | — |
| CLI-03 | `python -m qasr_cli --version` | — | 同 CLI-01 | 空 | 0 | — |
| CLI-04 | `qasr --help`（import blocker） | `torch/qwen_asr/vllm` 导入即失败并记录 | 正常 help | 不得有运行时回溯 | 0 | 导入日志无三个模块 |
| CLI-05 | `qasr --json bogus` | 未知命令 | 单个 JSON | 可为空 | 2 | `ok=false, error.kind=usage, code=2` |
| CLI-06 | `qasr --json transcribe` | 缺位置参数 | 单个 JSON | 可为空 | 2 | `command=transcribe`，error 同上 |
| CLI-07 | `qasr --json --jsonl languages` | 互斥 flag | 单个 JSON | 可为空 | 2 | error.kind=usage |
| CLI-08 | `qasr serve` / `qasr stream` | P1 命令 | — | 用法错误 | 2 | — |
| CLI-09 | `qasr transcribe --skip-existing x.wav` | P1 flag | — | 用法错误 | 2 | — |
| CLI-10 | `qasr --device quantum transcribe x.wav` | 非法枚举 | — | 用法错误 | 2 | — |
| CLI-11 | `qasr --dtype int8 transcribe x.wav` | 非法枚举 | — | 用法错误 | 2 | — |
| CLI-12 | `qasr --config missing.json config show` | 配置文件不存在 | — | 明确路径 | 2 | — |
| CLI-13 | `qasr --config bad.json config show` | 非法 JSON | — | 解析错误 | 2 | — |
| CLI-14 | `qasr --json languages` | 正常 | 单个 JSON | 空/警告 | 0 | `schema_version=1, ok=true` |
| CLI-15 | `qasr --model /nope transcribe --json a.wav` | 有效音频 + 模型缺失 | 单 JSON | 模型错误 | 3 | `ok=false, error.code=3` |
| CLI-16 | `qasr --model <合成完整目录> transcribe --json missing.wav` | 无效音频 + 模型路径存在 | 单 JSON | 输入错误优先 | 4 | `ok=false, error.code=4` |

### 4.2 三张语言表

| ID | 命令 | 输入 | 期望 | 退出码 |
|---|---|---|---|---|
| LANG-01 | `qasr languages --json` | — | `forceable` 30 项 | 0 |
| LANG-02 | 同上 | — | `alignable` 11 项，且 ⊆ forceable | 0 |
| LANG-03 | 同上 | — | `dialects` 22 项；`lid_labels` 52 项 | 0 |
| LANG-04 | 同上 | — | `Sichuan` 在 dialects/lid，不在 forceable | 0 |
| LANG-05 | 同上 | — | `zh→Chinese`、`en→English`、`yue→Cantonese`；条目含 name/iso/aliases | 0 |
| LANG-06 | `qasr --dry-run transcribe a.wav --language Sichuan` | 方言强制 | 错误提示用 `auto` | 4 |
| LANG-07 | `qasr --dry-run transcribe a.wav --language zh` | ISO | 成功规划 | 0 |
| LANG-08 | `qasr --dry-run transcribe a.wav --language Chinese` | 规范名 | 成功规划 | 0 |
| LANG-09 | `qasr --dry-run transcribe a.wav --language klingon` | 未知 | 明确不支持 | 2 或 4 |
| LANG-10 | `qasr align a.wav --text hi --language th --model /nope` | 可强制但不可对齐 | 对齐语言错误 | 4 |
| LANG-11 | `qasr align a.wav --text 你好 --language auto --model /nope` | 中文脚本判定 | 通过语言门，进入模型解析 | 3 |
| LANG-12 | `qasr align a.wav --text hello --language auto --model /nope` | Latin 自动判定 | 要求显式语言 | 4 |

### 4.3 配置

| ID | 场景 | 期望 |
|---|---|---|
| CFG-01 | `config path` | stdout 单行，路径以 `config.json` 结尾；退出 0 |
| CFG-02 | `config show --json` | 含 model/backend/device/dtype/offline 等有效值；默认 model=1.7b、backend=auto |
| CFG-03 | `QASR_HOME/config.json` 设 model=0.6b | show 为 0.6b |
| CFG-04 | 在 CFG-03 基础上 `QASR_MODEL=1.7b` | show 为 1.7b（env 覆盖 home） |
| CFG-05 | 再加 `--config cfg.json`（model=aligner） | show 为 aligner（`--config` 覆盖 env） |
| CFG-06 | 再加 CLI `--model 0.6b` | show 为 0.6b（CLI 最高） |
| CFG-07 | 项目 `proj/.qasr/config.json`，cwd=`proj/sub` | 向上查找到并生效 |
| CFG-08 | `QASR_CONFIG=…` | 被当作 `--config` 同层 |
| CFG-09 | `QASR_OFFLINE=1/0` | show 为 true/false |
| CFG-10 | `QASR_OFFLINE=maybe` | 退出 2 |
| CFG-11 | 配置含非法 backend/device/dtype | 退出 2 |
| CFG-12 | 相对 model_dir | 相对配置文件所在目录解析（若 show 暴露路径） |

### 4.4 模型与 verify

测试模型目录名假设为 repo basename（`Qwen3-ASR-1.7B`），与 qtts 的目录扫描惯例一致；Phase 2 若实现采用别的命名，先记为契约歧义，不算实现 bug。

| ID | 场景 | 期望 |
|---|---|---|
| MOD-01 | 空 `QASR_HOME` 下 `models list --json` | 3 个别名均 `present=false`；退出 0；纯 JSON |
| MOD-02 | `models path 1.7b`，目录存在 | stdout 单行绝对路径；退出 0 |
| MOD-03 | `models path 1.7b`，目录不存在 | 明确缺失；退出 3 或 4 |
| MOD-04 | 构造完整 model 目录后 `models verify 1.7b --json` | `complete` |
| MOD-05 | 完整目录 + `.aria2` 残留 | `incomplete`，hint 含 `pull` 与 `--resume` |
| MOD-06 | index 引用的 shard 缺失 | `incomplete` |
| MOD-07 | safetensors 头截断/非法 | `incomplete`，不得回溯 |
| MOD-08 | 目录不存在 | `missing` |
| MOD-09 | verify/list 在 import blocker 下 | 不尝试 import torch/qwen_asr/vllm |
| MOD-10 | 重复 verify | 指纹缓存可用；输出一致（不要求性能断言） |
| MOD-11 | 合成半包目录（复制真实 `config/tokenizer/vocab`，截断 shard 或加 `.aria2`） | `incomplete` + `pull --resume`；不依赖真实 1.7B 状态 |
| MOD-12 | `models pull` 无 hf/modelscope 依赖 | 退出 3，hint 给安装命令（环境相关，可 skip） |

构造的 safetensors 必须是合法头：`8 字节小端头长 + JSON 头 + 数据`；截断用例只保留前 4 字节或声明超长头。

### 4.5 doctor

| ID | 场景 | 期望 |
|---|---|---|
| DOC-01 | `doctor --json` | `{schema_version,ok,status,checks[]}`；每项含 name/status/value |
| DOC-02 | 默认（允许 warn/fail） | 退出 0 |
| DOC-03 | `doctor --json --strict` + import blocker | 退出 3 |
| DOC-04 | 缺 torch/qwen-asr | hint 含 `qasr-cli[local]` 或精确安装命令 |
| DOC-05 | `doctor --json --server http://x` | P0 未实现该 flag → 2 |
| DOC-06 | stdout 纯度 | 整个 stdout 可被一次 `json.loads` |

### 4.6 transcribe：输入、规划、错误

| ID | 命令/输入 | 期望 stdout | 期望 stderr | 退出码 |
|---|---|---|---|---|
| TR-01 | 不存在的文件 | — | 明确路径 | 4 |
| TR-02 | 0 字节 `.wav` | — | 坏音频 | 4 |
| TR-03 | 截断 WAV / 文本改名 `.wav` | — | 坏音频 | 4 |
| TR-04 | `.txt` 显式输入 | — | 不支持格式 | 4 |
| TR-05 | 坏文件 + 好文件（好文件需模型） | JSONL/批量 envelope | 逐条错误 | 6 |
| TR-06 | 目录 + 2 个文件 + 无任何输出通道 | — | 要求 out-dir/output/json | 2 |
| TR-06b | 2 个空 WAV + `--json`（无 out-dir） | batch envelope | — | 0 |
| TR-07 | `--output` + 多输入 | — | 用法错误 | 2 |
| TR-08 | `--format txt,json` + stdout | — | 多格式需 out-dir | 2 |
| TR-09 | `--format tsv` | — | P1 格式 | 2 |
| TR-10 | `--json` + `--format json` | — | 互斥 | 2 |
| TR-11 | `--jobs 2` 本地 + dry-run | 规划 | 含 warning（强制 1） | 0 |
| TR-12 | URL + `--dry-run` | plan 含 URL | 无下载副作用 | 0 |
| TR-13 | URL + `--offline`（未缓存） | — | 离线/未缓存 | 4 |
| TR-14 | `--batch-size 0` / `-1` | — | 用法错误 | 2 |
| TR-15 | `--max-new-tokens 0` | — | 用法错误 | 2 |
| TR-16 | `--max-new-tokens auto --dry-run` | 规划成功 | — | 0 |
| TR-17 | 同文件重复两次 + dry-run | plan inputs 去重为 1 | — | 0 |
| TR-18 | 字面 glob `dir/*.wav` + dry-run | 展开为实际文件 | — | 0 |
| TR-19 | 目录递归 + 符号链接环 | 有限时间结束，不重复 | — | 0 |
| TR-20 | dry-run 完整参数 | `plan.inputs/model/backend/device/dtype/outputs` | — | 0 |
| TR-21 | 0 采样空 WAV + `--model /nope` | `text=""`, `language=null`, warning 含 empty audio | — | 0 |

### 4.7 时间戳与字幕门禁

| ID | 命令/输入 | 期望 | 退出码 |
|---|---|---|---|
| TS-01 | `transcribe wav --timestamps`（默认 auto） | 要求显式 11 语言 | 4 |
| TS-02 | `transcribe wav --timestamps --language auto` | 同上，hint 给 11 语言 | 4 |
| TS-03 | `transcribe wav -f srt`（无 timestamps） | capability error，hint `--timestamps`/align | 4 |
| TS-04 | 同上 `-f vtt` | 同上 | 4 |
| TS-05 | `--granularity bogus` | 用法错误 | 2 |
| TS-06 | `--timestamps --language zh --dry-run` | 规划成功，不加载模型 | 0 |
| TS-07 | `--timestamps --language th --dry-run` | Thai 不可对齐 | 4 |

### 4.8 align

| ID | 命令/输入 | 期望 | 退出码 |
|---|---|---|---|
| AL-01 | 无 `--text/--text-file/--srt` | 用法错误 | 2 |
| AL-02 | 同时 `--text` 与 `--text-file` | 互斥 | 2 |
| AL-03 | 无 `--language` | 必填 | 2 |
| AL-04 | `--language th` | 非 11 对齐语言 | 4 |
| AL-05 | `--text-file missing.txt` | 输入错误 | 4 |
| AL-06 | 181s WAV | 超 180s，hint 提 window/P1 | 4 |
| AL-07 | 179.9s WAV + `--model /nope` | 通过时长门，进入模型解析 | 3 |
| AL-08 | `--coverage-min 1.5` / `-0.1` | 用法错误 | 2 |
| AL-09 | `--server` | P0 未实现 | 2 |
| AL-10 | `-f json,srt` 输出到 stdout | 多格式需 out-dir | 2 |
| AL-11 | 畸形 SRT | 明确解析错误 | 4（允许 2，需报告） |
| AL-12 | `--language auto` + 中文文本 | 通过语言门 | 3（配 `--model /nope`） |
| AL-13 | `--language auto` + 英文文本 | 要求显式 | 4 |

### 4.9 manifest 与批处理

| ID | 场景 | 期望 | 退出码 |
|---|---|---|---|
| MAN-01 | 2 条合法记录 + `--dry-run --json` | plan inputs=2，id 保留 | 0 |
| MAN-02 | 重复 id | 明确重复 | 2 |
| MAN-03 | 未知字段 | 明确字段名 | 2 |
| MAN-04 | `audio` 与 `audio_base64` 同时存在 | 二选一错误 | 2 |
| MAN-05 | 两者都不存在 | 必填错误 | 2 |
| MAN-06 | `output="../x"` | 拒绝逃逸 | 2 |
| MAN-07 | `output="/abs/x"` | 拒绝绝对路径 | 2 |
| MAN-08 | `output="a\u0000b"` | 拒绝 NUL | 2 |
| MAN-09 | `formats="txt"`（字符串） | schema 错误 | 2 |
| MAN-10 | `formats=["tsv"]` | P1 格式 | 2 |
| MAN-11 | `timestamps=true, language="auto"` | capability error | 4 |
| MAN-12 | `--manifest` + 位置参数 | 互斥 | 2 |
| MAN-13 | 两条都指向坏文件 + `--json` | `ok=false`，summary total=2/failed=2，退出码取首个错误=4 | 4 |
| MAN-14 | 活跃 `.qasr/lock`（PID 存活） | 明确“已有进程” | 2 |
| MAN-15 | 无 `--out-dir` 的 manifest | 用法错误 | 2 |
| MAN-16 | manifest 文件不存在 | 输入错误 | 4 |
| MAN-17 | 某行非法 JSON | 带行号 | 2 |
| MAN-18 | 非法 base64 | 明确拒绝（输入或 schema 错误） | 2 或 4 |
| MAN-19 | `--resume` 无 out-dir | 用法错误 | 2 |

### 4.10 stdout / stderr / JSON 契约

| ID | 场景 | 期望 |
|---|---|---|
| PUR-01 | `languages --json` | stdout 整体是一个 JSON 值，前后无字符 |
| PUR-02 | `--json transcribe missing.wav` | stdout 单 JSON；错误对象含 `schema_version,ok,command,error{kind,code,message,hint,retryable,details}` |
| PUR-03 | `--jsonl transcribe bad1 bad2 -o out` | 恰好 2 行 JSON，无 summary 行；无其他文本 |
| PUR-04 | 同 PUR-03 退出码 | 4（全部失败取首个错误码） |
| PUR-05 | `--help` | stderr 为空 |
| PUR-06 | argparse 错误 + `--json` | stdout 单 JSON（不是仅 stderr） |
| PUR-07 | `models list --json` | stdout 纯 JSON；缺失目录警告只在 stderr |
| PUR-08 | `config path` | stdout 单行路径，无日志 |

### 4.11 集成（需 `RUN_INTEGRATION=1` + 模型）

环境：

```bash
export RUN_INTEGRATION=1
export QASR_TEST_MODEL_DIR=/Users/walter/workspace/QwenLM/Qwen3-ASR/models
export QASR_TEST_AUDIO=/tmp/qasr_asr_zh.wav       # 可选
export QASR_TEST_DEVICE=mps                        # 可选
```

| ID | 场景 | 期望 |
|---|---|---|
| INT-TR-01 | 0.6B + 中文短音频 `--language zh --json` | `text` 非空且含预期中文；`language=Chinese`；`language_source=forced` |
| INT-TR-02 | 同上检查文本 | 无 `<asr_text>`、无 `language X` 前缀 |
| INT-TR-03 | 无 timestamps 转写 + `--aligner <空目录>` | 仍成功，证明纯文本转写不加载 aligner |
| INT-TS-01 | `--timestamps --language zh --json` | `segments` 非空、start≤end、单调；`timestamp_source=forced_aligner` |
| INT-SUB-01 | `-f srt -o out` | 合法 SRT；时间单调；`end≥start+1ms`；不超时长 |
| INT-SUB-02 | cue 规则 | 单 cue ≤7s 且显示宽度 ≤42（CJK=2）；英文不拆词 |
| INT-SUB-03 | `-f vtt` | 以 `WEBVTT` 开头；时间用 `.` |
| INT-AL-01 | 已知 13 字中文 + align | items=13；单调；coverage≥0.5 |
| INT-AL-02 | 覆盖率门禁 | 故意低覆盖 + `--coverage-min 0.9` → 5 |
| INT-EMP-01 | 0 样本/静音 WAV | exit 0；`text=""`；`language=null` |
| INT-BAT-01 | 1 好 + 1 坏 | exit 6；好项 ok，坏项 error；好项输出存在 |
| INT-RES-01 | 同参数跑两次 `--resume` | 第二次 skipped=1、exit 0、输出 SHA 不变 |
| INT-RES-02 | 改 `--language`/context 后 `--resume` | job key 变化 → 重跑，不复用旧结果 |
| INT-RES-03 | 手工篡改已完成输出 | SHA 不一致 → 重跑 |
| INT-RES-04 | 运行中 `kill -9` 后 `--resume` | 清理死锁；done 不重跑；最终 exit 0 |
| INT-RES-05 | 运行中 SIGINT | exit 130；state 不留 running；resume 可继续 |
| INT-LOCK-01 | 两个进程同 out-dir | 第二个 exit 2；第一个正常完成 |
| INT-PUR-01 | 真模型 `--json` | stdout 仍是单 JSON，第三方日志不污染 |
| INT-OFF-01 | `--offline` + 本地模型 | exit 0；无联网依赖 |

---

## 5. 对抗用例（重点）

| ID | 用例 | 为什么危险 | 期望 |
|---|---|---|---|
| ADV-01 | 半包模型：`.aria2` + shard 缺失 + 头损坏 | 上游会抛 `SafetensorError`，且可能下载/卡住 | `verify` 在加载前给 `incomplete` + `pull --resume`，exit 3 |
| ADV-02 | 坏音频：0 字节、截断头、文本改名、目录名 `.wav` | 解码器异常可能拖垮整批 | 单条 error；批量继续；exit 4/6 |
| ADV-03 | 空音频：合法 0 样本 WAV | 上游返回 `language None<asr_text>` | 成功，空文本，`language=null` |
| ADV-04 | 路径逃逸：`../`、绝对、NUL、`out` 是符号链接 | 覆盖 out-dir 外文件 | 全部 exit 2/4，不写文件 |
| ADV-05 | 重复 id / 重复输入路径 | 状态行覆盖、输出冲突 | 重复 id exit 2；重复输入去重 |
| ADV-06 | 并发锁：两进程同 out-dir | 状态文件互相覆盖 | 第二个 exit 2 |
| ADV-07 | `kill -9` 续跑 | lock 残留、running 悬挂、半截输出 | 死锁可恢复；running→pending；输出 SHA 校验 |
| ADV-08 | 180s 边界：179.9 / 180 / 180.1 / 181 | 上游常量 180，README 写 5 分钟 | 180.1/181 在加载前 exit 4；179.9 进入模型 |
| ADV-09 | 方言/不支持语言：Sichuan、Thai(timestamps)、auto(timestamps) | 静默错误时间戳 | 明确 exit 4 + hint |
| ADV-10 | stdout 污染：模型加载日志、transformers warning、进度条 | JSON 管道断裂 | stdout 只有结果 |
| ADV-11 | 无 torch 环境启动：`--help/--version/--dry-run/doctor/models verify` | 轻量安装用户直接崩 | 不 import torch/qwen_asr/vllm；doctor 给安装提示 |
| ADV-12 | 符号链接环、深目录、同名文件 | 递归卡死/重复 | 有限时间、去重、稳定排序 |
| ADV-13 | SRT 半包/时长为 0/负时间 | 渲染崩溃或非法时间 | 归一化，`end≥start+1ms` |
| ADV-14 | 配置循环/损坏 JSON/未知键 | 启动即崩 | exit 2，指明文件与键 |
| ADV-15 | 磁盘满 / 只读 out-dir | 半截文件 | 明确错误，不破坏旧输出（Phase 2 手工） |

---

## 6. 与 `P0-TASKS.md` 一一对应的验收 checklist

### M1 — 骨架、契约、模型与诊断

- [ ] T1.1 `pyproject.toml` 名称/版本/Python 约束/extras/entry point 正确。
- [ ] T1.1 `qasr --version`、`qasr --help` 无 torch 可运行。
- [ ] T1.1 启动不 import torch/qwen_asr/vllm。
- [ ] T1.1 退出码 0/2/3/4/5/6/130 与 `to_dict()` 完整。
- [ ] T1.2 `QASR_HOME` macOS/Linux 路径正确。
- [ ] T1.2 配置优先级五层完整。
- [ ] T1.2 `config show/path` 可用。
- [ ] T1.2 七个环境变量生效。
- [ ] T1.3 30/22/11 三张表与 ISO/别名。
- [ ] T1.3 `--language` 只接受 30；方言 exit 4。
- [ ] T1.3 `languages --json`。
- [ ] T1.4 三个别名与解析顺序。
- [ ] T1.4 `models list/path/verify`、`pull --source`。
- [ ] T1.4 verify 检查 `.aria2/.incomplete/.part`、index shard、safetensors 头。
- [ ] T1.4 `complete|incomplete|missing` + `pull --resume` hint。
- [ ] T1.4 指纹规则与缓存。
- [ ] T1.5 doctor 检查项、`--json`、默认 0、`--strict` 3。
- [ ] **M1 验收**：无 torch 五个命令通过。
- [ ] **M1 验收**：合成半包目录报 `incomplete`，真实 0.6B 报 `complete`。

### M2 — 输入、运行时、基础转写

- [ ] T2.1 文件/目录/glob/URL 展开；稳定排序、绝对路径去重。
- [ ] T2.1 目录扩展名白名单；符号链接不递归。
- [ ] T2.1 `id` 生成；`..` 逃逸拒绝。
- [ ] T2.1 URL 缓存、60s 超时、2GB 上限、offline 拒绝未缓存。
- [ ] T2.2 soundfile 探测；坏音频 `InputError`；不 import torch。
- [ ] T2.3 懒加载；offline 环境变量在 import 前设置；`local_files_only=True`。
- [ ] T2.3 `Capabilities`/`Backend`/`TransformersBackend`。
- [ ] T2.3 设备/dtype 默认值；`--dtype` 覆盖。
- [ ] T2.3 `max_new_tokens auto` 公式；按本次最大时长。
- [ ] T2.3 进程内模型缓存；纯文本不加载 aligner。
- [ ] T2.4 逐条校验后分组；组失败降级逐条。
- [ ] T2.4 `--max-inference-batch-size` 与分组解耦。
- [ ] T2.5 transcribe 参数面完整。
- [ ] T2.5 单条/批量 JSON、JSONL 形状。
- [ ] T2.5 stdout 只放结果，第三方输出重定向 stderr。
- [ ] T2.5 空音频成功、空文本、`language=null`。
- [ ] **M2 验收**：中文短音频 text/language 正确。
- [ ] **M2 验收**：多文件 jsonl 每行一个结果。
- [ ] **M2 验收**：一坏一好 → 好成功、坏 error、exit 6。
- [ ] **M2 验收**：`--help` 不 import torch。

### M3 — 时间戳、字幕、对齐

- [ ] T3.1 `--timestamps` 只接受显式 11 语言；auto exit 4。
- [ ] T3.1 传 `forced_aligner` 与 kwargs。
- [ ] T3.1 `segments{text,start,end,kind}`；`timestamp_source`。
- [ ] T3.1 清理 `<asr_text>` / `language X`。
- [ ] T3.2 cue 标点/700ms 停顿；≤7s、≤42 宽度；英文不拆词。
- [ ] T3.2 SRT/VTT 格式；整数毫秒；half-up；单调；`end≥start+1ms`；不超时长。
- [ ] T3.2 `sentence|word|char`。
- [ ] T3.2 无时间戳请求 srt/vtt → capability error。
- [ ] T3.3 `align` 三种文本来源。
- [ ] T3.3 语言必填/11 语言/auto 脚本判定。
- [ ] T3.3 ≤180s，超限 exit 4。
- [ ] T3.3 输出格式与 `--output/--out-dir`。
- [ ] T3.3 `--coverage-min` 规范化匹配与 exit 5/0 关闭。
- [ ] T3.3 P0 不提供 `--server`。
- [ ] **M3 验收**：中文 SRT 合法。
- [ ] **M3 验收**：align 返回 13 items 且单调。
- [ ] **M3 验收**：低覆盖 exit 5。
- [ ] **M3 验收**：SRT/VTT golden。

### M4 — 批处理、恢复、文档、收尾

- [ ] T4.1 manifest 字段严格；未知字段报错。
- [ ] T4.1 `audio` XOR `audio_base64`；id 唯一。
- [ ] T4.1 `output` 相对、拒绝绝对/`..`/NUL。
- [ ] T4.1 `--manifest` 与位置参数互斥。
- [ ] T4.2 state 字段与状态机。
- [ ] T4.2 原子写 temp+fsync+rename；state append+fsync。
- [ ] T4.2 `.qasr/lock`；活跃锁 exit 2；running→pending。
- [ ] T4.2 job_key 组成完整。
- [ ] T4.2 overwrite/resume/默认优先级。
- [ ] T4.2 失败保留旧输出；成功才 rename。
- [ ] T4.2 部分失败 6；全失败取首错。
- [ ] T4.3 单输入/目录镜像/URL 哈希命名。
- [ ] T4.3 冲突默认报错；overwrite 覆盖。
- [ ] T4.3 多格式 + stdout → 用法错误。
- [ ] T4.4 README 安装/模型/5 quickstart/输出/退出码/限制。
- [ ] T4.4 `schemas/*.json` 完整，contract 测试通过。
- [ ] T4.5 `uv run pytest` 纯 CPU 全绿。
- [ ] T4.5 0.6B 集成测试通过。
- [ ] T4.5 `uv tool install --from . "qasr-cli[local]"` 后 version/doctor 通过。
- [ ] **M4 验收**：干净环境安装 + doctor 全绿。
- [ ] **M4 验收**：目录 `--resume` 可重复；kill -9 续跑只处理未完成项。
- [ ] **M4 验收**：`models verify` 拦截合成半包。

---

## 7. 判定标准

### PASS

- §6 全部勾选，或未勾选项有书面、非阻塞、经确认的偏差。
- 无 BLOCKER；MAJOR 为 0，或全部在本波次修复并有回归测试。
- `review_tests/` 非集成部分全绿；集成部分在本机模型上全绿。
- stdout 纯度、退出码、路径安全、resume 四项抽查无例外。

### PASS WITH CHANGES

- 主链路（transcribe/align/models verify/resume）可用。
- 存在 1–3 个 MAJOR：不造成错误结果/数据损坏，但违反契约或验收项。
- 或存在环境性偏差（例如 1.7B 本机无法实测），但有替代证据。
- 输出必须附：最小修复建议、复现命令、回归测试名。

### FAIL

- 任一 BLOCKER：
  - 错误时间戳/错误文本被 `ok=true` 返回；
  - 路径逃逸或覆盖 out-dir 外文件；
  - `--json` stdout 不可解析；
  - 半包模型绕过 verify 并在加载期崩溃；
  - `kill -9` 后无法 resume，或 resume 复用不一致结果；
  - 批量坏文件导致好文件失败；
  - 空音频被判失败。
- 或 P0 核心验收项缺失 ≥2 个且无替代路径。

---

## 8. 规格歧义与 Phase 2 需确认项

这些不先判 FAIL；Phase 2 先问实现者/看 README，再决定是否记为 MINOR。

1. **多输入文件输出的默认格式**：DESIGN 说“多输入默认 jsonl”，但这是 stdout 形态；写盘时默认 `txt` 还是 `jsonl` 未写死。
2. **`models verify 1.7b` 的目录命名**：`QASR_HOME/models/<repo basename>` 还是 `<alias>`；本计划按 qtts 惯例假设 basename。
3. **`align --language auto` 的 P0 状态**：§3.4 说 P1 才提供，但又给了 P0 脚本判定兜底；以兜底行为测试。
4. **`align` 的时长检查与模型解析顺序**：本计划假设“先拒绝 >180s，再解析模型”，否则离线用户会先看到模型错误。
5. **陈旧 lock（PID 已死）**：DESIGN 只写“活跃锁 exit 2”，未写陈旧锁清理；kill -9 验收隐含必须清理。
6. **未知语言的退出码**：2（用法）还是 4（输入）；测试接受二者，报告需统一建议。
7. **`models path` 缺失时退出码**：3 还是 4；测试接受二者。
8. **manifest 中 `audio_base64` 是否必须 data URI**：裸 base64 是否接受未写死。
9. **空目录输入**：0 + 空结果，还是 4；测试接受二者但要求有明确 summary。
10. **`--dry-run` 是否允许 URL 下载**：本计划按“不下载”测试。
11. **`--jobs` 在 P0 是否暴露**：DESIGN 参数表有，P0 范围无远程；测试只要求“>1 时 warning 并强制 1”。
12. **`--no-progress` / `--continue-on-error` / `--fail-fast` 是否 P0**：P0-TASKS 未列；不强制，缺失不算 FAIL。
13. **JSON 键顺序/浮点格式**：不做字节级 golden，只做结构与数值容差。
14. **`metrics.load_ms/infer_ms` 在缓存命中时的语义**：只要求存在且非负。
15. **SIGTERM 是否也映射 130**：DESIGN 写 SIGINT/SIGTERM，测试以 SIGINT 为主。

---

## 9. Phase 2 执行手册

```bash
cd /Users/walter/workspace/QwenLM/qwen-asr-cli

# 1) 纯 CPU / 无 torch 部分
uv run pytest review_tests -m "not integration" -q

# 2) 仓库自带测试
uv run pytest -q

# 3) 集成（本机模型）
RUN_INTEGRATION=1 \
QASR_TEST_MODEL_DIR=/Users/walter/workspace/QwenLM/Qwen3-ASR/models \
QASR_TEST_AUDIO=/tmp/qasr_asr_zh.wav \
uv run pytest review_tests -m integration -q

# 4) 安装形态验证
uv build
uv tool install --python 3.12 --from . "qasr-cli[local]"
qasr --version && qasr doctor --json
```

若 `.venv` 尚未就绪，可用已有运行时：

```bash
QASR_PYTHON=/Users/walter/workspace/QwenLM/qwen-tts-cli/.venv/bin/python \
  /Users/walter/workspace/QwenLM/qwen-tts-cli/.venv/bin/python -m pytest review_tests -q
```

证据要求：每条失败附

1. 完整命令（含 env）；
2. 退出码；
3. stdout/stderr 原文（截断到关键段）；
4. 期望 vs 实际；
5. 最小修复建议（不改实现）。

---

## 10. `review_tests/` 文件清单

| 文件 | 覆盖 |
|---|---|
| `conftest.py` | subprocess runner、QASR_HOME 隔离、WAV/manifest/model 夹具、import blocker |
| `helpers.py` | JSON/JSONL/SRT 解析、WAV 生成、假模型目录构造 |
| `test_m1_cli_surface.py` | CLI-01..14 |
| `test_m1_languages.py` | LANG-01..12 |
| `test_m1_config.py` | CFG-01..12 |
| `test_m1_models.py` | MOD-01..11 |
| `test_m1_doctor.py` | DOC-01..06 |
| `test_m2_inputs_errors.py` | TR-01..20 |
| `test_m2_stdout_contract.py` | PUR-01..08 |
| `test_m3_timestamps_align.py` | TS-01..07、AL-01..13 |
| `test_m4_manifest_resume.py` | MAN-01..19、ADV-04..06 |
| `test_adversarial_audio.py` | ADV-02、ADV-08、ADV-12、ADV-13 |
| `test_integration_p0.py` | INT-TR/TS/SUB/AL/EMP/BAT/RES/OFF |
| `test_integration_kill_lock.py` | INT-RES-04/05、INT-LOCK-01 |
| `README.md` | 运行方式、环境变量、gating 说明 |

所有测试只通过 `subprocess` 调 `qasr`；不 import `qasr_cli` 任何实现模块。
