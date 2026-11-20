何时读我：命令失败、退出码非 0、结果看似成功但不对，或要判断下一步该重试、换参数还是换后端时。

# 故障分流

版本基线 `qasr 0.3.2`。先看退出码，再看 `--json` 的 `error.kind` / `retryable` / `details` / `hint`。

## 退出码 → error.kind

| code | kind | 含义 | 下一步 |
|---|---|---|---|
| 0 | — | 成功 | 仍读 `.ok`；doctor 可能 `status:"fail"` |
| 2 | `usage` / `config` | 参数互斥、缺必填、配置/manifest schema | 按 `hint` 修参数；不重试 |
| 3 | `model` / `runtime` | 模型缺失/半包、torch/vLLM、下载、运行时 | `doctor` + `models verify`；修环境或下载 |
| 4 | `input` / `capability` | 输入/音频/路径/语言/能力不支持 | 换输入或后端；不重试 |
| 5 | `quality` | `align` coverage 低，或 `stream` 达时长上限 | 看阈值/音频；`align` 没写任何文件 |
| 6 | `partial` | 批量部分失败 | 读 `.errors[]`；仅 `retryable` 才 `--retry` |
| 130 | — | SIGINT/SIGTERM | 看 state/lock，再 `--resume` |

`--fail-fast` 或“全部失败”时返回首个错误的码，而不是 6。

## 按码分流

### 2（用法/配置/manifest）

常见触发：

- transcribe 的 `--json` 与 `-f json/jsonl/tsv/md` 同用；`--json` 与 `--jsonl` 同用。
  （`align -f ... --json` 合法，不要误判。）
- `--out-dir` 与 `--output` 同用；多格式却只用 `--output`；`--output` 配多输入。
- 多输入没有 `-o`，也没有 `--json`/`--jsonl`。
- `--resume`/`--skip-existing` 缺 `--out-dir`（后者可用 `--output`）。
- `--retry` 与 `--fail-fast` 同用。
- `--gpu-memory-utilization` 给非 vLLM 后端。
- `--server` 与显式 `--backend transformers|vllm` 同用。
- `align` 缺 `--language`；参考文本源不是恰好一个；窗口/overlap 越界。
- manifest 未知字段、重复 id、`audio` 与 `audio_base64` 同时给、`output` 绝对/含 `..`。
- `stream` 用了 `--json`/`--dry-run`；`--pcm` 没配 stdin；stdin 没给 `--pcm`。
- `completion badshell`；`jobs list --limit` 越界。
- `config set` 未知键或 token 类键。

处理：读 `error.message` 与 `hint`，修正后重跑；不要盲目重试。

### 3（模型/运行时/后端）

```bash
qasr --json doctor | jq '{status, bad: [.checks[] | select(.status != "ok") | {name, status, hint}]}'
qasr --json models list | jq '.models[] | {alias, status, present}'
qasr --json models verify 1.7b
```

- `torch`/`qwen-asr` fail：装 `qasr-cli[local]`（Python 3.12）。
- `model:*` 不是 `complete`：半包/缺失；`models pull <alias> --resume`（先取得用户同意）。
- 显式 `--backend vllm` 在非 Linux/CUDA：`transcribe` 3、`stream` 4；`align` 忽略 vllm、
  始终 transformers（`remote` 仍 4）。不要在本机硬切 vllm。
- `--gpu-memory-utilization` 只属于 vLLM；其他后端是 2。
- 远程连接/超时：增大 `--timeout`、降低 `--jobs`；再考虑 `--retries`。
- `serve` 启动 3：`qwen-asr-serve` 未安装或 CUDA/vLLM 不可用。模型不完整不会直接 3，
  而是回退 repo 并可能下载；先看 `serve --dry-run` 的 `.source`。

### 4（输入/能力）

- 输入不存在、目录没有音频、glob 无匹配、音频不可解码/超限。
- 未知语言或方言标签用于 `-l`（22 个方言只能 `auto`）。
- `--timestamps` 配 `-l auto`，或缺 alignable 语言。
- `-f srt/vtt` 没给 `--timestamps`。
- `--timestamps` 配 `--max-chunk-seconds >180`。
- 目标输出已存在（transcribe 默认拒绝）。
- 远程要时间戳/align/stream；`--granularity` 非 sentence。
- manifest 记录 `timestamps:true` 但没有 `language`。
- `stream` 收到 URL、目录/不存在的文件。
- `jobs show` 未知 job/未登记 out-dir。
- `models path`/`rm` 指向不存在的模型（部分为 3，见命令自身）。

处理：换输入/后端/语言；远程能力不支持的不要重试。

### 5（质量/上限）

- `align` 的 `result.coverage` 低于 `--coverage-min`：**不会写任何输出**。
  校对文本/音频，或降低阈值（`0` 关闭门禁），必要时换对齐语言。
- `stream` 达到 `--max-stream-seconds`（默认 3600）：exit 5，无文件产物；
  `--emit jsonl` 会留一条 `ok:false` 终止快照。按需增大上限或分段。
- `transcribe` 没有质量门禁，**不会**返回 5。README 里“subtitle quality gate”不是 transcribe 行为。

### 6（批量部分失败）

```bash
qasr --json transcribe ./audio -o out/ --resume \
  | jq '{summary, errors: [.errors[] | {id, kind: .error.kind, code: .error.code, retryable: .error.retryable}]}'
```

- `.summary.ok > 0` 且 `.ok:false` 是部分成功；不要当整体成功。
- 仅 `retryable:true` 且用户同意时加 `--retry`（只重试一次）。
- 参数/输入/能力错误不重试；修好输入后用同一命令 `--resume`。
- 失败项明细也在 `<out-dir>/.qasr/state.jsonl`；`jobs show` 给 `.first_error` 与 counts。

### 130（中断）

- 先确认 `.qasr/lock` 的 PID 已退出；active 锁会阻止下一次运行（2）。
- 看 `state.jsonl` 中 `running`/`pending`，再用 `--resume`；不要重复提交。
- 流式 Ctrl-C 会先尝试发最终快照，再 130。

## 常见“假成功”

| 现象 | 真相 |
|---|---|
| `doctor` exit 0 | 默认即使 `status:"fail"` 也 0；必须读 `.status`/`.checks[]`，CI 用 `--strict` |
| `--dry-run` 返回 `ok:true` | 只证明解析/计划，不证明模型存在或目标可写 |
| `models list` 的 `present:true` | 不代表完整；看 `status=="complete"` / `models verify` |
| `models show <不存在>` | rc0/`ok:true`、`status:"missing"`、`path:null`；必须读 `.status` |
| sidecar `out/*.json` 的 `.outputs` | 可能为空/缺失；以 stdout envelope 为准 |
| `--jsonl` 最后一行 | 不是批次 summary；jsonl 只有 item 行，`done:` 只在写文件模式的 stderr |
| `jobs` counts 之和 < `job.total` | `total` 含 spec 校验失败项，它们不写 state |
| 零样本音频 `ok:true` | 文本为空是合法结果；看 `duration_s` 与 `.result.text` |
| `--skip-existing` 跳过 | 只说明文件存在且非空，不保证内容正确 |
| `bench` 数字 | 只测吞吐/RTF，不是准确率 |
| `--resume` 跳过 | 依赖 job key + 输出存在 + 输出 SHA，不是“目录存在” |
| `--offline` | 不约束 `serve` 子进程；serve 仍可能下载 repo |
| `--server` 带 token | 服务端没有内建校验；token 只保护客户端请求头 |

## 远程与流式的额外检查

- 远程 408/429/5xx/连接失败可重试；401/403/400 是输入/凭据问题，不重试。
- 远程 context 后语言可能不符：读 `.result.warnings[]` 的
  `language_not_enforced_on_chat`、`server_ignored_language`。
- 远程 `.model.id` 是实际模型；服务端多个模型时客户端只用第一个（除非 `--model` 匹配）。
- `stream` 正常结束：`seq` 从 0 连续、最后一条 `is_final:true`、无 `ok:false`。
  快照文本可能被更短文本替换，这是累计语义，不是丢字。
- `stream` 文件输入读到 EOF 即结束，不是 `tail -f`；跟随增长数据请用 `--input stdin`。

## 最小自检

```bash
qasr --json transcribe a.wav | jq -e '.ok == true and .command == "transcribe"'
qasr --json transcribe ./audio -o out/ --resume \
  | jq -e '.summary.total == (.summary.ok + .summary.failed + .summary.skipped)'
qasr --json jobs show JOB_ID | jq -e '.state == "present" and .counts.pending == 0 and .counts.running == 0'
```

错误对象必有 `kind/code/message/retryable/details`，`hint` 可选；`schema_version` 恒为 1。
