# qasr-cli P2 实现报告

> 任务书：`P2-TASKS.md`（含 d38697a 两条澄清与 522a130 Q1–Q16 裁决）。
> 范围：`stream`、`completion`、`finetune lint-jsonl` 与打包/文档。

## 1. 里程碑对照

| 里程碑 | 状态 | 内容 |
|---|---|---|
| S | ✅ 完成 | `qasr stream`：file / stdin s16le PCM / mic，单流 vLLM，累计快照 |
| C | ✅ 完成 | `qasr completion bash\|zsh\|fish`，从 argparse 生成 |
| F | ✅ 完成 | `qasr finetune lint-jsonl`，SFT JSONL 逐行校验 |
| 打包/文档 | ✅ 完成 | `[mic]` / `[all]` extras、README、本报告 |

## 2. 提交

| commit | 内容 |
|---|---|
| `12a5d77` | `feat(stream): add vLLM streaming with cumulative snapshots` |
| `f21de70` | `feat(completion): generate bash, zsh and fish scripts from argparse` |
| `67be093` | `fix(stream): account processing time in realtime pacing` |
| `077a223` | `feat(finetune): add lint-jsonl SFT data validator` |
| `26dcc66` | `docs(p2): packaging, README and implementation report` |
| `ac526bb` | `fix(stream): precheck microphone support before model loading` |

每个功能 commit 自带测试；`67be093` 同时把显式 `--backend vllm` 不可用时的
hint 统一为 `uv tool install "qasr-cli[vllm,mic]"`。

## 3. 测试命令与结果

```bash
uv run --no-sync ruff check qasr_cli tests        # All checks passed
UV_HTTP_TIMEOUT=300 uv run --no-sync pytest -q tests
# 415 passed, 6 skipped

UV_HTTP_TIMEOUT=300 uv run --no-sync pytest -q review_tests
# 164 passed, 17 skipped
UV_HTTP_TIMEOUT=300 uv run --no-sync pytest -q review_tests_p1
# 204 passed, 8 skipped
```

默认测试不需要 GPU、vLLM 或麦克风。`review_tests_p2/` 由独立审查方维护，
本实现不修改、不纳入实现 commit。

### 3.1 本机覆盖（无 GPU / 无 vLLM / 无麦克风）

- `stream`：fake backend 覆盖 chunk 切分、累计文本、`seq` 连续、最终快照、
  短尾包、cap 单条 quality 错误、流中错误行、流前错误普通 envelope、
  text 模式 TTY/非 TTY、SIGINT 最终快照 + 130。
- PCM：stdin s16le、奇数字节残尾（按 frame 丢弃并 warning）、立体声下混、
  EOF、短块。
- 源与能力：file 不存在/目录/URL → 4，`-` → 2，stdin 缺 `--pcm` → 2，
  `--pcm/--sample-rate/--channels` 用于 file/mic → 2，非 16 kHz → 2，
  `--realtime` + stdin/mic → 2，remote/transformers/`--server` → 4，
  无 vLLM → 4，无 sounddevice → 4。
- `--json` 与 `stream` 无条件互斥 → 2；`--emit jsonl` 每行可独立
  `json.loads`，stdout 无第三方日志。
- `completion`：三种 shell 的脚本均覆盖 `transcribe/align/serve/stream/
  finetune/completion` 及 `qasr <cmd> --help` 中的全部长选项；`bash -n`
  通过，本机存在 `zsh`/`fish` 时同样执行 `-n`；末尾恰好一个换行。
- `finetune`：合法/非法 JSON、`audio`/`text`/`prompt` 类型、NUL、缺失/
  不可读/目录、`--no-check-files`、LID/方言/`None` 前缀、空正文、未知字段
  warning 与 `--strict` 升级、重复 audio、空行不计入 total、JSON schema、
  人读 summary、未知子命令 → 2；lint 路径不 import torch/datasets。

## 4. 契约裁决落地

| 裁决 | 实现 |
|---|---|
| Q1 cap | 只发一条 `ok:false,is_final:true,error.kind="quality"`，`seq` 为下一序号；不发普通成功 final |
| Q2 `--json` | 与 `stream` 无条件互斥 → 2；错误本身按 envelope 输出 |
| Q3 completion | 从 parser/registry 反射生成，无第二份手写清单 |
| Q4 fake | 无运行时注入开关；默认测试仅 monkeypatch 内部单元 |
| Q5 strict | warning 保留，同时追加同 field 的 error；duplicate audio 同样升级 |
| Q6 残尾 | 按 `channels × 2` frame 判断，丢弃不完整帧并在 stderr 报字节数 |
| Q7 输入 | 目录/glob/URL → 4；`-` → 2 并提示 `--input stdin`；重复 `--input` 后者生效 |
| Q8 stdin 参数 | 仅 stdin 有意义，file/mic 同用 → 2；mic 采集块固定 100 ms |
| Q9 realtime | 首块不预睡、EOF 尾块不睡、不补睡；每块 sleep 剩余预算 |
| Q10 `audio_s` | 已送入样本 / 16000，3 位小数；部分尾块只出现在最终快照 |
| Q11 错误行 | 独立 `oneOf`，含 `schema_version/ok/command/seq/is_final/error` |
| Q12 finetune | `path` 原样输出；重复 audio 按解析后绝对路径；语法错误 `field="json"` |
| Q13 completion 输出 | stdout 恰好一个尾换行；额外参数 → 2 |
| Q14 能力优先 | 先判 stream 能力（→4），再解析模型（缺失 →3）；显式 vllm 不可用 → 4 |
| Q15 SIGINT | 状态机已启动则 flush 尾包、发 final、退出 130；此前走普通 envelope |
| Q16 schema | `stream.json`/`finetune-lint.json` 均用 `oneOf`/严格必填字段 |

## 5. 与任务书的偏差 / 实现决定

1. **realtime 节奏**：按 Q9“处理超预算不补睡”，每个完整 chunk 的 sleep
   预算为 `max(0, chunk_ms/1000 - 该 chunk 处理耗时)`；首块、EOF 尾块不睡。
2. **显式 vLLM 错误**：`require_vllm()` 的原始 message 保留为
   `CapabilityError.message`，hint 统一为
   `uv tool install "qasr-cli[vllm,mic]"`；原 hint 放入
   `details.original_hint`。
3. **`--strict` 计数**：warning 同时追加 error，`valid` 只统计无任何 error
   的行；`warnings` 数组仍保留原 warning（Q5）。
4. **重复 audio 检测**：即使 `--no-check-files` 也按解析后的绝对路径检测；
   `--no-check-files` 只跳过存在性/可读性。
5. **fish 覆盖断言**：除 `complete -c qasr -l <opt>` 外，脚本行尾注释中列出
   长选项字面量，确保覆盖测试对三类 shell 使用同一断言方式。
6. **`--dry-run` + stream**：明确拒绝 → 2，避免“干跑流式”的歧义。
7. **mic 错误分类与顺序**：`sounddevice` 缺失在 `--input mic` 确定后立即
   预检 → 4（capability），早于模型解析和 vLLM 加载；设备打开/读取失败仍在
   采集时按 3（model/runtime）处理。
8. **text 模式**：TTY 用 `\r` 覆盖并最终换行；非 TTY 只在结束时打印最终文本。

## 6. gpu06 验收（协调者执行）

本机不连接、不传输文件到 gpu06。以下证据由协调者在 gpu06（实现 HEAD
`1f7788f`；其后 `3df825e` 仅改模型缺失文案，stream 代码相同；模型 1.7B，
CUDA0，`--gpu-memory-utilization 0.3`）执行并提供：

- **file 200s**：rc0，101 行，`seq` 0..100，`is_final` 仅一次；最终
  `audio_s=200.0`，elapsed 75.17s（首条快照 46.61s，按契约含模型加载），
  English，text 2703 字符。
- **stdin s16le 同一音频**：rc0，101 行，`seq`/final 语义相同，
  `audio_s=200.0`，English，text 与 file 输入完全一致（2703 字符）。
- **cap=5s**：rc5；2 条 `ok:true` 非最终快照 + 1 条
  `ok:false`/`quality`/`is_final` 快照（`seq=2`）。
- **SIGINT**：tmux C-c → rc130；9 行，最后 `seq=8` 为 `ok:true`/`is_final`，
  `audio_s=16.0`，无半行 JSON。
- **能力边界**：显式 `--backend transformers`、`--server` → rc4，消息与
  hint 正确。
- **中文 4.2s**：`--language zh` + context + `--chunk-ms 1000 --realtime`
  → rc0，5 行，`seq`/`audio_s` 单调，`language=Chinese`，最终文本与 P1 一致，
  elapsed 6.4s。
- **mic（无 sounddevice）**：`ac526bb` 把 sounddevice 预检前移到模型解析前；
  gpu06 复验（HEAD `89c1222`）rc4、WALL=0.17s，stderr 无 vLLM
  EngineCore/architecture 日志，仅 mic 安装 hint。
- **缺失模型**：`--backend vllm --model missing/model` → rc3（能力检查之后）；
  **文件不存在** → rc4。

## 7. 限制

- 不含远程流式、流式时间戳/对齐/翻译/说话人分离、多 GPU、视频、watch、
  Windows、daemon。
- 不含训练执行或数据集构建；`finetune` 只做 JSONL 静态校验，不解码音频。
- vLLM 与 `stream` 仅 Linux + CUDA；macOS 无 vLLM。
- stdin 仅支持 16 kHz s16le PCM；其他采样率/编码在 CLI 层拒绝。
- `completion` 是静态脚本生成，不探测模型、网络或动态值。

## 8. 结论

P2 三项功能与打包/文档均已实现，默认测试与既有 review 套件通过。gpu06
vLLM 黑盒验收已由协调者完成 file/stdin/cap/SIGINT/能力边界/中文场景；
mic 预检顺序问题已由 `ac526bb` 修复并在 gpu06 复验（rc4、0.17s、无引擎日志）。

独立审查（Codex，干净 `git archive 21f111a`）：**PASS，0 BLOCKER /
0 MAJOR / 0 MINOR**；`review_tests_p2` 187 passed、P0/P1 回归与仓库测试全绿、
ruff 通过。审查结论见 `P2-REVIEW-REPORT.md`。
