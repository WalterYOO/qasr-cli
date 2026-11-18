# P2 任务书：流式与生态

> 基线：`v0.2.0`（P1 最终验收 `15b281a`，审查基线 `6a0f524`）。
> 设计依据：`DESIGN.md` v0.3 §2.1/§3.5/§5/§6/§8。
> 流程：Claude partner 实现 + 自测；Codex partner 独立审查/测试（`P2-REVIEW-*`、
> `review_tests_p2/`）；协调者做契约裁决与 gpu06 验收。实现方不得修改
> `review_tests*/` 与 `P2-REVIEW-*`。

---

## 0. 范围

P2 只做三组能力：

| 块 | 能力 |
|---|---|
| S | `qasr stream`：file / stdin(PCM) / mic，vLLM 单流，累计快照 |
| C | `qasr completion bash\|zsh\|fish` |
| F | `qasr finetune lint-jsonl` |

**明确不做**（出现即按 capability error 处理，不静默降级）：

- 远程流式（服务端不暴露 chunk 回滚语义）；
- 流式时间戳、流式对齐、流式翻译、说话人分离；
- 多卡/分布式 serve、视频输入、目录 watch、Windows 专用路径、常驻 daemon；
- 训练执行、数据集构建（只做 lint）。

---

## 1. 全局契约

- 退出码沿用 P0/P1：`0` 成功、`2` 用法、`3` 模型/运行时、`4` 输入/能力、
  `5` 质量/运行上限、`6` 批量部分失败、`130` 中断。
- `--json` 模式下 stdout 必须**恰好**是约定格式；日志、warning、进度一律 stderr。
- `--help`、`--version`、`completion`、`finetune lint-jsonl` 不得 import
  torch / vllm / qwen_asr / sounddevice。
- 所有新命令必须在无 GPU、无 vLLM、无麦克风的机器上给出可执行的
  capability/安装提示，而不是 ImportError/Traceback。
- `QASR_SERVE_TOKEN` 仍然只从环境读取；stream/completion/finetune 的任何输出、
  日志、state、schema 不得出现 token 或 server userinfo。
- 新增 schema：`schemas/stream.json`（单行快照）、`schemas/finetune-lint.json`。
  `schemas/` 清单测试必须覆盖。

---

## 2. S 块：`qasr stream`

### S1 CLI 表面

```
qasr stream --input mic
qasr stream --input stdin --pcm s16le --sample-rate 16000 --channels 1 --emit jsonl
qasr stream --input file.wav --realtime
qasr [--backend vllm] stream --input a.wav --language zh --context "..." --emit jsonl
```

| 参数 | 默认 | 约束 |
|---|---|---|
| `--input` | 必填 | `mic` / `stdin` / 单个本地音频路径；目录、glob、URL、`-` → 2 或 4（见下） |
| `--pcm` | 无 | 仅 stdin 需要；只接受 `s16le` |
| `--sample-rate` | `16000` | P2 只支持 16000；其他值 → 2 |
| `--channels` | `1` | 1 或 2；2 时求均值下混 |
| `--chunk-ms` | `2000` | 100–10000，且须为 10ms 整数倍 |
| `--unfixed-chunks` | `2` | ≥0 |
| `--unfixed-tokens` | `5` | ≥0 |
| `--max-stream-seconds` | `3600` | >0；超过 → 5 |
| `--emit` | `text` | `text` / `jsonl` |
| `--realtime` | off | 仅 file；按墙钟节奏喂音频 |
| `--language` | auto | 同 P0 强制语言表（30 种）；输出仍以状态里的原始标签为准 |
| `--context` / `--context-file` | 空 | 互斥；语义同 `transcribe` |

- `--backend`：只有 `vllm` 可用；`auto` 在 vLLM 可用时解析为 vllm，否则 → 4；
  显式 `transformers`/`remote`/配置了 `--server` → 4，hint 指向
  `uv tool install "qasr-cli[vllm,mic]"`。
- 模型解析、离线、指纹、错误语义与 `transcribe` 一致；缺失模型 → 3。
- 全局 `--device` / `--dtype` / `--gpu-memory-utilization` 与 `transcribe`/`bench`
  同语义：显式设备映射到 `CUDA_VISIBLE_DEVICES`，`--gpu-memory-utilization`
  必须透传给 `VLLMBackend`（共享 GPU 上不传会按 0.9 OOM）。
- `--json` 只影响错误 envelope 与人类模式；与 `--emit jsonl` 同用是允许的
  （不额外输出 envelope），与 `--emit text` 同用 → 2。
- `--realtime` 只允许 file；与 stdin/mic 同用 → 2。
- `--emit jsonl` 与人类模式都由同一状态机产出，不允许两条代码路径。

### S2 音频源

- **file**：本地普通文件；用现有 `audio.py` 解码为 16k 单声道 float32。
  文件不存在/不可解码 → 4。默认尽快处理；`--realtime` 时按 `chunk_ms` 睡眠，
  不追帧、不补静音。
- **stdin**：必须显式 `--pcm s16le`，否则 → 2。按字节流读取，尾部不足
  2 字节的残字节丢弃并向 stderr 发 warning；EOF 视为正常结束。
  2 声道按帧 `(L+R)/2` 下混。每次最多读 `chunk_ms` 对应采样。
- **mic**：延迟 import `sounddevice`；缺失 → 4，hint 给出
  `uv tool install "qasr-cli[mic]"`。16k 单声道，块大小 100ms。
  设备打不开/采样率不支持 → 3，附原始设备错误摘要（截断、无 token）。
- 三种源都必须：Ctrl-C 后先 flush 尾包并输出最终快照，再以 130 退出；
  任何源的内部读取错误（非中断）→ 3，且不得留下半行 JSON。

### S3 流式状态与快照

- 直接使用 `Qwen3ASRModel.init_streaming_state/streaming_transcribe/
  finish_streaming_transcribe`；`--chunk-ms` → `chunk_size_sec`。
- 每完成一个 chunk 发一条快照；输入结束（EOF/mic 停止/文件读完）后调用
  finish，发一条 `is_final=true` 的快照；即使尾包为空也发最终快照。
- **JSONL 是累计快照，不是 delta**；`seq` 从 0 递增，最终快照也占一个 seq。
- 行格式（`schemas/stream.json`，`additionalProperties: false`）：

  ```json
  {"schema_version":1,"ok":true,"command":"stream","seq":0,
   "audio_s":2.0,"elapsed_s":0.41,"language":"Chinese",
   "text":"今天天气","is_final":false}
  ```

  - `language`：模型原始标签；未知/静音时为 `null`；`text` 允许空串。
  - `audio_s`：截至本快照已送入模型的音频秒数；`elapsed_s`：进程启动至今。
- 中途失败且已输出过快照时，最后补一行
  `{"schema_version":1,"ok":false,"command":"stream","seq":N,"is_final":true,
  "error":{kind,code,message,retryable,details}}` 后按该错误码退出；
  尚未输出过任何行时走普通错误 envelope。
- 人类 `text` 模式：TTY 用 `\r` 覆盖当前行、结束换行；非 TTY 只在最终快照
  打印一次文本。`--emit text` 时不允许同时 `--json`（`--json` 只影响 envelope，
  与 jsonl 流冲突 → 2）。
- 超过 `--max-stream-seconds`：停止读源、flush 尾包、发最终快照，然后以 5
  退出，error.kind=`quality`，message 明确是运行上限而非识别质量。

### S4 资源与安全

- 内存只保留 `audio_accum` 与当前 buffer；不落盘、不写 state、不登记 jobs。
- `--context-file` 读文件失败 → 4；内容不打印到 stderr。
- 模型/设备初始化失败 → 3；stdout 不得被 vLLM 日志污染（复用 P1 的
  `configure_vllm_logging` / `redirect_vllm_logging` / atexit fd 守卫）。

---

## 3. C 块：`qasr completion`

```
qasr completion bash
qasr completion zsh
qasr completion fish
```

- 只接受这三个 shell；其他值 / 缺参 → 2。
- 输出为可直接 `source`/`eval` 的脚本到 stdout，末尾单个换行；不写文件。
- 脚本必须覆盖当前**全部**顶层子命令与每个子命令的常用 flag（从
  parser/registry 生成或由单一常量表生成，禁止手抄两份清单）。
- 纯静态文本；不 import torch/vllm/qwen_asr；不探测模型或网络。
- `--json` 与 `completion` 同用 → 2（completion 不是 JSON 命令）。
- 校验：`bash -n` 通过；zsh/fish 若本机存在则 `zsh -n` / `fish -n` 通过；
  三种脚本都必须包含 `transcribe`、`align`、`serve`、`stream`、`finetune`、
  `completion`。
- README 给出三种 shell 的一行安装方式。

---

## 4. F 块：`qasr finetune lint-jsonl`

```
qasr finetune lint-jsonl train.jsonl [--json] [--strict] [--no-check-files]
```

- 输入：官方 SFT JSONL（`finetuning/README.md`）。逐行校验，不 import
  torch/datasets，不解码音频（只查文件存在与可读）。
- 每行必须是 JSON object，允许字段：`audio`（必填，非空字符串）、
  `text`（必填，非空字符串）、`prompt`（可选字符串）。
  - `audio`：绝对路径或相对 JSONL 所在目录；NUL → 错误；不存在/不是普通文件
    → 错误；`--no-check-files` 时跳过存在性检查（仍校验类型）。
  - `text`：若以 `language <Name><asr_text>` 开头，`<Name>` 必须是
    `languages` 的 LID 标签（含 22 方言）或 `None`，且 `<asr_text>` 后正文
    非空；无前缀视为合法（模型按无语言前缀训练）。
  - `prompt`：存在时必须为字符串。
  - 未知字段：warning（与 SFT 脚本丢弃行为一致），`--strict` 下升级为错误。
  - 空行：跳过，不计入 total。
  - 同一 `audio` 重复出现：warning。
- 错误逐行累计，不 fail-fast；有任一错误 → 4；只有 warning → 0；
  `--strict` 且存在 warning → 4。
- 文件不存在/不可读 → 4；非法 JSON → 该行为错误，继续后续行。
- `--json` 输出（`schemas/finetune-lint.json`）：

  ```json
  {"schema_version":1,"ok":false,"command":"finetune","action":"lint-jsonl",
   "path":"train.jsonl","total":3,"valid":2,
   "errors":[{"line":2,"field":"audio","message":"..."}],
   "warnings":[{"line":3,"field":"text","message":"..."}]}
  ```

  `ok` 表示无错误（`--strict` 时无警告才为 true）；人读模式逐条打印
  `path:line: field: message`，最后一行 summary。
- `qasr finetune --help` 列出子命令；未知子命令 → 2。

---

## 5. 打包与文档

- `pyproject.toml` 增加 `mic = ["sounddevice>=0.4"]` 与
  `all = ["qasr-cli[local,mic,hf,modelscope]"]`（与 DESIGN §6 一致）；
  `vllm` extra 不变。
- README：命令总览加 `stream/completion/finetune`；安装矩阵补
  `[vllm,mic]`；明确远程不支持流式、macOS 无 vLLM。
- `P2-REPORT.md`：交付说明、契约偏差、验收证据（本机 + gpu06）、限制。

---

## 6. 测试与验收

### 6.1 默认（无 GPU / 无 vLLM / 无麦克风）

- 单元：`stream` 状态机用 fake backend（`init_streaming_state` /
  `streaming_transcribe` / `finish_*` 返回可预测文本）覆盖
  chunk 切分、seq、最终快照、空尾包、cap、错误行、文本模式。
- PCM：stdin 字节流、s16le 解码、奇数字节残尾、立体声下混、EOF、短块。
- 源：file 解码/不存在/目录/URL、`--realtime` 时间节奏（monkeypatch sleep）。
- 能力：无 vllm → 4、remote/server → 4、transformers 显式 → 4、
  无 sounddevice → 4、非 16k → 2、mic+`--realtime` → 2。
- stdout：`--emit jsonl` 每行可独立 `json.loads`，整个 stdout 无日志；
  `--json` 冲突 → 2。
- completion：三 shell `-n` 校验、覆盖子命令、无 torch import。
- finetune：合法/非法/前缀语言/方言/None/未知字段/重复 audio/空行/
  `--strict`/`--no-check-files`/schema 校验。
- 仓库既有 `tests`、`review_tests`、`review_tests_p1` 全绿；ruff 通过。

### 6.2 gpu06（vLLM，协调者执行）

- `stream --input file.wav --emit jsonl`：60s 音频 → 快照 seq 连续、
  最终行 `is_final=true`、文本非空、stdout 每行均为合法 JSON。
- `stream --input stdin --pcm s16le`：从 WAV 抽 PCM 管道输入，结果与 file
  一致（允许文本差异，seq/结束语义一致）。
- `--language zh --context ...`：语言标签与 context 生效。
- `--max-stream-seconds` 小于音频时长：最终快照后退出 5。
- Ctrl-C（tmux 发送 SIGINT）：最终快照 + 130，无半行 JSON。
- 显式 `--backend transformers` / `--server`：4；无 `sounddevice` 的服务器上
  `--input mic`：4。

### 6.3 独立审查

Codex 在干净 `git archive` 快照上只通过 `python -m qasr_cli` 黑盒验证；
`review_tests_p2/` 不得 import 实现内部；假 vLLM 模块与假 `sounddevice` 均为
stdlib fixture。最终给出 PASS / PASS WITH CHANGES + BLOCKER/MAJOR/MINOR 计数。

---

## 7. 契约裁决（Codex P2-Q1..Q16，2026-11-18）

以下裁决优先于前文含糊处，实现与审查均以此为准。

| ID | 裁决 |
|---|---|
| Q1 | cap 触发时只发**一条** `ok:false,is_final:true,error.kind="quality"` 快照，`seq` 取下一序号；即使此前没有任何行也发这条错误快照，不走普通 envelope。不再额外发 `ok:true` 的最终快照（`is_final` 全流只出现一次）。 |
| Q2 | `stream` 与全局 `--json` **无条件互斥**（任意位置、任意 `--emit`）→ 2，错误本身按 envelope 输出（因为 `--json` 已给出）。机器输出只由 `--emit jsonl` 控制。 |
| Q3 | completion 的可测契约：对每个顶层子命令，`qasr <cmd> --help` 中出现的**所有 long option** 都必须出现在对应 shell 脚本里；根命令的 long option 也要覆盖；脚本必须包含全部顶层子命令名。测试用 `--help` 动态提取，不维护第二份清单。 |
| Q4 | 不引入任何运行时 fake 注入开关。stream **成功路径**只在 gpu06 以黑盒方式验收（`-m gpu` 脚本，由协调者/审查方在服务器执行）；Mac 本机黑盒只覆盖 usage、能力错误、PCM 成帧与非成功路径。实现内部单元测试可用 fake backend。 |
| Q5 | `--strict`：每条 warning 同时追加一条同 `field` 的 error，`warnings` 原样保留；`valid` 只统计“无 error”的行；重复 `audio` 的 warning 同样升级。 |
| Q6 | 残字节按**帧字节数**（`channels × 2`）判断：不足一帧全部丢弃，stderr warning 带丢弃字节数；不产生半个样本。 |
| Q7 | 目录/glob/URL → 4；`-` → 2（hint 指向 `--input stdin`）；`--input` 重复出现由 argparse 处理为后者覆盖，不额外报错。 |
| Q8 | `--pcm/--sample-rate/--channels` 只对 stdin 有意义，与 file/mic 同用 → 2；mic 采集块固定 100ms，`--chunk-ms` 只控制累计解码块（仍须 100–10000ms 且 10ms 整数倍）。 |
| Q9 | `--realtime`：每个完整 chunk 处理完后睡 `chunk_ms`；首块不预睡；处理超预算不补睡；EOF 尾块不睡。 |
| Q10 | `audio_s = 已送入 streaming 的累计采样 / 16000`，保留 3 位小数（含最终 partial tail）；partial tail 只在最终快照体现，不单独发非最终快照。 |
| Q11 | 错误行是独立 `oneOf` 变体，只要求 `schema_version/ok/command/seq/is_final/error`；`seq` 为下一序号；`error` 对象与 `schemas/error.json` 的 `error` 完全一致。 |
| Q12 | finetune `path` 原样输出；重复 audio 用 resolve 后绝对路径判断；symlink 指向普通文件即可；JSON 语法错误 `field="json"`；summary 固定含 total/valid/errors/warnings 数量。 |
| Q13 | completion stdout 末尾恰好一个 `\n`；zsh/fish 不存在时跳过语法检查但保留内容/覆盖断言；多余参数 → 2。 |
| Q14 | 先判 stream 能力（非 vllm、remote、`--server` → 4），再解析模型（缺模型 → 3）；显式 `--backend vllm` 但运行时不可用统一 → 4。 |
| Q15 | 只要进入 stream 状态机，Ctrl-C 都先发最终快照再 130（已完成的快照为 `ok:true`，读取中出错为 `ok:false`）；未进入状态机则普通 envelope。TTY 用 `\r` 覆盖并在结束补 `\n`，非 TTY 只打印最终文本一次。 |
| Q16 | `schemas/stream.json` 用 `oneOf` 区分 success/error 变体；`finetune-lint.json` 的 errors/warnings item 要求 `line/field/message` 三者必备，`field` 允许 `"json"` 等哨兵值。 |

---

## 8. 交付顺序与门禁

1. S1/S2/S3（stream + 源 + 快照）→ 本机 fake backend 测试。
2. C（completion）与 F（finetune lint）→ 独立 commit。
3. 打包/README/schema → commit。
4. 协调者 gpu06 验收 → Codex 终审 → tag `v0.3.0`（由协调者决定）。
