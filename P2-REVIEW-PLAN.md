# P2 独立审查计划（stream / completion / finetune lint）

审查者：独立 Codex reviewer（只审不改实现）
基线：`v0.2.0`（P1 终审 `6a0f524`），P2 契约 `P2-TASKS.md`（`522a130`，
含 §7 对 Q1–Q16 的裁决；`d38697a` 的 GPU 透传要求仍有效）
设计依据：`DESIGN.md` v0.3 §2.1/§3.5/§5/§6/§8/§11/§12
审查套件：`review_tests_p2/`（只通过 `subprocess` 调 `python -m qasr_cli`）

---

## 1. 范围与非范围

### 1.1 审查范围

| 块 | 内容 |
|---|---|
| S1 | `qasr stream` 参数矩阵、默认值、互斥、能力错误、退出码 |
| S2 | file / stdin PCM / mic 三种源；s16le 成帧、立体声下混、残字节、EOF、real-time |
| S3 | 累计快照、`seq`、`is_final`、最终快照、失败错误行、人类模式、`--max-stream-seconds` |
| S4 | stdout 纯净、无 state/jobs、无 token 泄漏、无 torch/vllm/sounddevice 启动 |
| C | `completion bash|zsh|fish` 语法、覆盖、静态性、`--json` 冲突 |
| F | `finetune lint-jsonl` 全部字段/前缀/方言/严格模式/文件检查/schema |
| 全局 | 新 schema、pyproject extras、README 命令面、无 GPU 能力提示 |

### 1.2 非范围

- 远程流式；流式时间戳/对齐/翻译/说话人分离。
- 多卡/分布式 serve、视频、目录 watch、Windows 专用路径、常驻 daemon。
- 训练执行、数据集构建、模型下载/转换。
- GPU/vLLM 真机行为与 1h 音频：由协调者 gpu06 提供证据，本套件只做
  `gpu` 标记的占位/跳过，不冒充独立 GPU 证据。

---

## 2. 契约裁决（`P2-TASKS.md` §7，commit `522a130`）

> Q1–Q16 已全部采纳。以下为审查测试采用的最终口径；测试 docstring 标注对应
> Q 编号，避免把裁决后的行为再次当作歧义。

| ID | 最终裁决 | 审查落点 |
|---|---|---|
| Q1 | cap 只发一条 `ok:false,is_final:true,error.kind="quality"`，`seq` 下一号；无前置行也发；全流 `is_final` 仅一次。 | `test_max_stream_seconds_*`、schema 校验 |
| Q2 | `stream` 与全局 `--json` **无条件互斥**（任意位置/emit）→ 2。 | `test_json_is_unconditionally_incompatible_with_stream` |
| Q3 | completion 必须覆盖根命令与每个顶层子命令 `--help` 的全部 long option；脚本含全部子命令名。 | `test_completion_covers_all_long_options_from_help` |
| Q4 | 不引入运行时 fake 注入；stream 成功路径以 gpu06 黑盒为准；Mac 覆盖 usage/能力/PCM 成帧/非成功路径。 | `-m gpu` 用例 + 本机 simulated harness（见 §5 说明） |
| Q5 | strict 下每条 warning 追加同 `field` error，`warnings` 保留；`valid` 只算无 error 行；重复 audio 同样升级。 | `test_*_strict_*` |
| Q6 | 残字节按 `channels×2` 帧判断，不足一帧丢弃并 warning。 | `test_stereo_residual_uses_frame_bytes_*` |
| Q7 | 目录/glob/URL → 4；`-` → 2；重复 `--input` argparse 后者覆盖。 | `test_bad_input_forms_*`、`test_repeated_input_uses_last_value` |
| Q8 | stdin 专属参数用于 file/mic → 2；mic 采集块固定 100ms。 | `test_stdin_only_flags_*`、mic `blocksize==1600` |
| Q9 | realtime 在每个完整 chunk 后睡 `chunk_ms`；首块不预睡、不补睡、EOF 尾块不睡。 | `test_realtime_*` |
| Q10 | `audio_s=累计采样/16000`，3 位小数；partial 只在最终快照。 | `test_audio_s_is_rounded_*`、`test_partial_tail_*` |
| Q11 | 错误行独立 oneOf，只要求 6 个字段；`seq` 下一号；`error` 复用 `error.json`。 | `test_midstream_error_line_*` |
| Q12 | `path` 原样；重复按 resolve 后路径；symlink 指向普通文件即可；JSON 错误 `field="json"`；summary 含计数。 | finetune 用例 |
| Q13 | 末尾恰一个 `\n`；zsh/fish 缺失只跳过语法检查；多余参数 → 2。 | completion 用例 |
| Q14 | 先能力（→4）后模型（→3）；显式 vllm 不可用统一 → 4。 | `test_*_capability_*` |
| Q15 | 进入状态机后 SIGINT 先最终快照再 130；未进入走普通 envelope；TTY `\r` + 末尾 `\n`，非 TTY 只打印最终文本。 | `test_sigint_*`、`test_human_*` |
| Q16 | `stream.json` 用 oneOf；finetune item 的 `line/field/message` 必备。 | schema 用例 |

### 2.1 仍存在的实现风险（非任务书歧义）

- **Q4 的本机可测性**：实现没有（也不应有）fake 注入开关，而
  `require_vllm()` 在 macOS 直接拒绝。审查套件用子进程 `PYTHONPATH` 注入
  fake `torch/vllm/qwen_asr` + `sitecustomize` 模拟 Linux，从而在 Mac 覆盖
  PCM 成帧/下混/快照状态机；这些用例标记 `simulated`，**不作为 gpu06 证据**，
  也不修改实现。若协调者不接受平台模拟，可将这批用例改为 `gpu` 并在服务器执行。
- **`--gpu-memory-utilization` 透传**：`d38697a` 明确要求共享 GPU 上必须传给
  `VLLMBackend`；fake harness 通过 `Qwen3ASRModel.LLM` 的 kwargs 验证。
- **`--device cuda:N` 映射**：要求映射为 `CUDA_VISIBLE_DEVICES=N`；fake harness
  在模型初始化时记录该环境变量。

---

## 3. 风险清单（按严重度）

### 3.1 BLOCKER

当前无已确认 BLOCKER。Q4 已裁决：stream 成功路径以 gpu06 为准；本机
`simulated` harness 只做补充证据，不作为 PASS 的独立 GPU 依据。

### 3.2 MAJOR

1. **cap 路径**（Q1）：最容易出现 `is_final` 两次、seq 断裂或 stdout 半行。
2. **`--json` 无条件互斥**（Q2）：任意位置/emit 都必须 2。
3. **completion 完整性**（Q3）：实现可能漏掉 P1/P2 新 flag；用 `--help` 动态
   提取对照。
4. **能力/模型优先级**（Q14）：显式 vllm 不可用必须是 4，不能是 3。
5. **strict warning 表示**（Q5）：`ok/valid/errors/warnings` 必须自洽。
6. **立体声残帧**（Q6）：必须按 4 字节帧判断，不能把 2 字节当半个样本。
7. **GPU 参数透传**（`d38697a`）：`--gpu-memory-utilization` 与 `cuda:N` 映射
   在共享 GPU 上不可退化。

### 3.3 MINOR

- `--input` 2/4 映射、stdin 专属参数与 file/mic 的组合、realtime 睡眠边界、
  `audio_s` 舍入、错误行 schema、finetune 路径/重复/symlink、completion 末尾
  换行、能力/模型优先级、Ctrl-C/TTY 细节、schema 变体。

---

## 4. 黑盒测试矩阵

### 4.1 S1 参数与能力

| 用例 | 输入 | 期望 stdout/JSON | 退出 |
|---|---|---|---|
| 缺 `--input` | `stream` | usage error；无 traceback | 2 |
| stdin 缺 pcm | `--input stdin` | usage；hint `--pcm s16le` | 2 |
| 非 s16le | `--pcm s16be` | usage | 2 |
| 非 16k | `--sample-rate 8000` | usage | 2 |
| channels 0/3 | `--channels` | usage | 2 |
| chunk 越界/非 10ms | `--chunk-ms 99/10001/105` | usage | 2 |
| unfixed 负数 | `--unfixed-chunks -1` | usage | 2 |
| max 0/负 | `--max-stream-seconds 0` | usage | 2 |
| emit 非法 | `--emit yaml` | usage | 2 |
| realtime+stdin | `--input stdin --realtime` | usage | 2 |
| 目录/URL/glob/`-` | 各一个 | input/capability 或 usage（按裁决） | 4/2 |
| transformers/remote/server | `--backend ...` / `--server` | capability；hint `[vllm,mic]` | 4 |
| 无 vllm auto/vllm | 无 fake vllm | capability；无 traceback | 4 |
| 非法语言 | `--language xx` | usage | 2 |
| context 互斥 | `--context a --context-file f` | usage | 2 |
| `--json` + 任意 emit | 两种 emit / flag 前后 | usage；无条件互斥（Q2） | 2 |
| stdin 专属 flag + file/mic | `--pcm/--sample-rate/--channels` | usage（Q8） | 2 |
| GPU flags | `--gpu-memory-utilization 0/1.1` | usage | 2 |
| GPU flags 透传 | `--gpu-memory-utilization 0.3` / `--device cuda:3` | fake 后端记录 kwargs / `CUDA_VISIBLE_DEVICES` | 0 |
| 能力先于模型 | 显式 transformers/remote + 缺失模型 | capability（Q14） | 4 |
| `--help` | `stream --help` | 无 torch/vllm/sounddevice import | 0 |

### 4.2 S2 音频源

| 用例 | 输入 | 期望 |
|---|---|---|
| 1s mono stdin | 16000 帧 s16le | 最终 1 行；audio_s≈1.0；seq=0 |
| 5s / chunk 2s | 80000 帧 | 2 个非最终 + 1 最终；seq 0,1,2 |
| 4s 整块 | 64000 帧 | 2 个 chunk + 空尾最终；最终 text 不变 |
| 4.5s | 72000 帧 | 2 chunk + 尾包最终；audio_s≈4.5 |
| 奇数字节 | 1s + 1 byte | stderr warning；rc 0；audio_s≈1.0 |
| 立体声残帧 | 1s stereo + 2 bytes | warning；按丢帧处理；下混正确 |
| 立体声下混 | L=1000/R=3000 | 送入模型首样本≈2000/32768 |
| 空 stdin | 0 字节 | 1 行最终快照；text=""；language=null；rc 0 |
| 分片写 stdin | 多次 write | 不丢样本；最终 seq 正确 |
| file 不存在/目录/URL | 各一个 | 4 |
| realtime | 2s/chunk 1s | 墙钟≥约 1.5s；无补静音 |
| mic 缺失 | 无 fake sounddevice | 4 + `[mic]` hint |
| mic 设备错误 | fake 抛 PortAudioError | 3 + 设备摘要，无 token |
| mic 成功 | fake 1s PCM | 最终快照 audio_s≈1；块 100ms |
| realtime | 2s/chunk 1s | 墙钟≥约 1.5s；无补静音 |
| realtime 尾块 | 1.5s/chunk 1s | <约 1.75s（EOF 尾块不睡，Q9） |

### 4.3 S3 快照

| 用例 | 期望 |
|---|---|
| JSONL 行合法性 | 每行独立 `json.loads`；无日志/半行 |
| seq | 0..N 连续；最终行 `is_final=true` 且仅一次 |
| 累计语义 | 后续 `text` 以前一快照为前缀/累计（允许修订，但不是 delta） |
| `language` | 原始标签或 null；不伪造 |
| `elapsed_s` | ≥0 且非递减 |
| 人类非 TTY | 只打印最终文本一次；无 JSON |
| 人类 TTY | 出现 `\r`；结束换行 |
| 中途失败 | 已输出过行 → 最后一行 `ok:false,is_final:true,error.code`；无半行 |
| 首行前失败 | 普通 error envelope；无 seq/is_final |
| max-stream-seconds | rc 5；单条 `ok:false,is_final:true`；`error.kind=quality`（Q1） |
| Ctrl-C | rc 130；最终快照；无半行 |
| 无 state/jobs | `QASR_HOME/jobs` 为空；无 `.qasr/state.jsonl` |
| 无 token | stdout/stderr/state 不含 `QASR_SERVE_TOKEN` |
| vLLM banner | fake vllm 打印 stdout → 仍只有 JSONL |

### 4.4 C completion

| 用例 | 期望 |
|---|---|
| bash/zsh/fish | rc 0；末尾单换行；包含全部顶层命令 |
| 语法 | `bash -n`；zsh/fish 存在则 `-n` |
| flag 覆盖 | 根 + 每个顶层子命令 `--help` 的全部 long option（Q3） |
| 无 import | torch/vllm/qwen_asr/sounddevice blocker 下 rc 0 |
| 未知/缺 shell | 2 |
| `--json` 同用 | 2 |
| 多余参数 | 2（按 P2-Q13） |
| token/userinfo | 输出不含 token/userinfo |

### 4.5 F finetune lint-jsonl

| 用例 | 期望 |
|---|---|
| 合法行 | rc 0；total/valid 正确；`ok:true` |
| 空行 | 跳过，不计 total |
| 非法 JSON | 该行 error；继续后续行；rc 4 |
| 非 object | error |
| 缺/空 audio/text | error |
| prompt 非字符串 | error |
| 音频不存在/目录/NUL | error |
| `--no-check-files` | 不存在不报错；类型/NUL 仍校验 |
| 前缀合法 | `language Chinese<asr_text>正文` → 0 |
| 方言/None | 取自 `qasr languages --json` 的 dialect/lid_labels → 0 |
| 非法前缀/空正文 | error |
| 未知字段 | warning；strict → 4 |
| 重复 audio | warning；strict → 4 |
| strict 无 warning | 0；有 warning → 4 |
| 文件不存在/不可读 | 4 |
| JSON schema | 通过 `schemas/finetune-lint.json` |
| 人读模式 | `path:line: field: message` + summary |
| 无 import | torch/datasets/vllm/qwen_asr blocker 下正常 |

### 4.6 全局/schema/打包

- `schemas/stream.json`、`schemas/finetune-lint.json` 存在、合法 JSON、
  样例输出通过 jsonschema；`additionalProperties:false` 的字段集与实现一致。
- `pyproject.toml` 有 `mic`/`all` extras；`all` 含 `local,mic,hf,modelscope`。
- README 提及 `stream`/`completion`/`finetune` 与安装矩阵。
- `--help`/`--version`/`completion`/`finetune` 不 import 重运行时。

### 4.7 gpu06 验收复核点（只复核证据，不在 Mac 冒充）

| 证据 | 复核点 |
|---|---|
| 60s `stream --input file --emit jsonl` | 每行可 `json.loads`；seq 连续；最终 `is_final=true` 且唯一；文本非空 |
| WAV 抽 PCM 走 stdin | 与 file 的 seq/结束语义一致；允许文本差异；无半行 |
| `--language zh --context ...` | init 收到 context；输出 language 为模型原始标签，不伪造 |
| `--max-stream-seconds` < 音频 | rc 5；单条 `ok:false,is_final:true,error.kind=quality`（Q1） |
| tmux 发 SIGINT | 最终快照 + rc 130；无半行 JSON（Q15） |
| 显式 `--backend transformers` / `--server` | rc 4（Q14） |
| 无 sounddevice 的服务器 `--input mic` | rc 4 + `[mic]` 安装 hint |
| `--gpu-memory-utilization` / `--device cuda:N` | vLLM 启动参数与 `CUDA_VISIBLE_DEVICES` 生效（d38697a） |

---

## 5. Fixtures 与假服务

| fixture | 内容 |
|---|---|
| `fake_stream` | 临时目录放 `vllm/__init__.py`、`qwen_asr/__init__.py`；fake `Qwen3ASRModel.LLM` 实现 `init_streaming_state/streaming_transcribe/finish_streaming_transcribe`，按真实 qwen-asr 语义缓冲整 chunk，记录 init 参数/每块采样/尾包到 JSONL log |
| `fake_stream` 故障注入 | `QASR_FAKE_STREAM_FAIL=init|chunk:N|finish`、`QASR_FAKE_STREAM_SLEEP=秒`、`QASR_FAKE_STREAM_BANNER=1` |
| `fake_sounddevice` | fake `InputStream`/`read`，从 `QASR_FAKE_MIC_PCM` 喂 16k 单声道块；可注入打开失败 |
| `fake_model_dir` | 复用 P1 的完整合成模型目录（config/tokenizer/safetensors header） |
| `pcm` | stdlib 生成 s16le mono/stereo、指定采样值、残字节 |
| `import_blocker` | 生成 `torch/vllm/qwen_asr/sounddevice/datasets` 模块，import 即写 marker 并抛错 |
| `pty_runner` | stdlib `pty` 启动 stream，捕获 TTY 人类模式 |
| `jsonl_reader` | 逐行 `json.loads`，失败时报告行号与原始行 |

**约束**：所有测试只通过 `subprocess` 调 `python -m qasr_cli`；fake 模块是独立
文件，不 import `qasr_cli`；不改 `qasr_cli/`、`schemas/`、`README`、`P2-REPORT`。

---

## 6. 验收 checklist（对应 P2-TASKS）

| 任务 | 黑盒验收点 |
|---|---|
| S1 | 参数矩阵/默认值/互斥/能力错误/退出码/hint |
| S2 | file/stdin/mic；s16le、16k、单/双声道、残帧、EOF、realtime |
| S3 | 累计快照、seq、is_final、最终快照、错误行、cap、人类模式 |
| S4 | stdout 纯净、无 state/jobs、无 token、无重运行时 import |
| C | 三 shell 语法/覆盖/静态性/冲突 |
| F | 全部字段规则/前缀/方言/None/未知字段/重复/strict/no-check-files |
| §5 | schema、extras、README |
| §6.1 | 默认无 GPU 套件全绿；仓库 tests/review_tests/review_tests_p1/ruff |
| §6.2 | gpu06 证据由协调者提供；本套件不伪造 |

---

## 7. 判定标准

- **PASS**：P2 契约项全部有黑盒证据；0 BLOCKER / 0 MAJOR；MINOR 不影响
  契约或已有明确裁决。
- **PASS WITH CHANGES**：仅存在已记录、可最小修复的 MAJOR/MINOR，且不改变
  主链路；给出逐条复现与修复建议。
- **FAIL**：存在 BLOCKER，或 MAJOR 导致契约不可验收/数据/安全风险。

GPU/vLLM/1h 证据不作为本轮 PASS 的独立依据；若协调者未提供，报告会明确标注
“未独立验证”，但不会因此单独判 FAIL。

---

## 8. 裁决状态与执行口径

Q1–Q16 已由 `P2-TASKS.md` §7（`522a130`）定稿，`review_tests_p2/` 已按裁决
改为正向断言，不再保留 `ambiguous` 标记。审查报告会把“实现偏离裁决”与
“裁决本身仍有歧义”分开列出。`simulated` 用例只在 Mac 上补充状态机/PCM
证据；gpu06 的真实 vLLM 证据仍由协调者提供。

### 8.1 测试文件与职责

| 文件 | 覆盖 |
|---|---|
| `test_p2_stream_usage.py` | S1 参数/能力/互斥/Q7/Q8/Q14/GPU 透传 |
| `test_p2_stream_pcm.py` | S2 stdin 成帧、下混、残帧、EOF、Q10 |
| `test_p2_stream_snapshots.py` | S3/S4 快照、seq、错误行、cap、TTY、SIGINT、纯净性 |
| `test_p2_mic_realtime.py` | mic 成功/失败/缺失、`--realtime` 节奏 |
| `test_p2_completion.py` | C 三 shell 语法、覆盖（Q3）、冲突、静态性 |
| `test_p2_finetune.py` | F 全部字段/前缀/方言/strict/文件检查/schema |
| `test_p2_schemas_packaging.py` | 新 schema、extras、README、无重运行时 |
