何时读我：一次处理多个文件、目录、stdin 清单、URL 或 manifest，或要断点续跑、查看 jobs/state 时。

# 批量、manifest 与断点续跑

版本基线 `qasr 0.3.4`。批量仍走 `transcribe`，不是独立命令；`align` 只接受单个音频。

## 输入展开

```bash
qasr --json transcribe a.wav b.wav -o out/ --resume
qasr --json transcribe ./audio --recursive -f txt,srt -l zh --timestamps -o out/ --resume
qasr --json transcribe 'audio/**/*.wav' --recursive -o out/   # glob 由 qasr 展开；** 递归需 --recursive
printf '%s\n' a.wav b.wav | qasr --json transcribe --stdin-list -o out/
```

- 支持扩展名：`.wav .mp3 .flac .m4a .ogg .opus .webm .mp4`。
- 目录扫描按文件名排序；`--recursive` 才进入子目录，且跳过符号链接目录。
- glob 结果按路径字符串排序，只保留受支持的音频文件。
- 去重按解析后的真实路径；同一文件写两次只处理一次。
- 目录输入的 id 保留相对子路径（输出也镜像到对应子目录）；不同 id 若落到同一输出路径，会以 2 拒绝。
- 顺序：按命令行给出顺序；每个目录/glob 内部稳定排序。不要依赖并发完成顺序。
- 多输入且没有 `--json`/`--jsonl` 时必须 `-o DIR`（2）；`--output` 只接受一个输入（2）。
- `--manifest` 与位置参数互斥（2），`--stdin-list` 与 `--manifest` 也互斥（2）。stdin 可与位置参数并用。
- 单独写 `-` 不是输入：用 `--stdin-list` 或显式 `/dev/stdin`。

### URL

- 只接受 `http(s)://`；下载到 `QASR_HOME/cache/url/`，同名缓存命中即复用。
- 单文件上限 2 GiB，连接/读取超时 60 s；失败是 4（input），不会自动重试下载。
- `--offline` 遇到未缓存 URL 会 4；dry-run 只给出缓存路径，不联网。
- URL 会联网并可能把内容写入本地缓存；先确认来源可信、目标端可访问。
- 远程 `--server` 则是把本地音频**上传**到服务端（单文件 25 MB 上限），与 URL 下载方向相反。

## manifest（异构参数）

每条记录一行 JSON；`--manifest` 强制要求 `--out-dir`。

```jsonl
{"id":"ep01","audio":"audio/ep01.wav","language":"zh","formats":["txt"]}
{"id":"ep02","audio":"https://example.com/ep02.wav","language":"zh","formats":["srt"],"timestamps":true,"output":"sub/ep02"}
{"id":"ep03","audio_base64":"AAAA","language":"en","formats":["json"]}
```

允许字段：`id audio audio_base64 language context formats output timestamps granularity`。

- `audio` 与 `audio_base64` 恰好一个；否则 2。
- `id` 可省略（按源生成）；重复 id → 2。
- `formats` 必须是字符串数组；`output` 必须是相对 `--out-dir` 的路径，不能绝对、不能含 `..`。
- 未知字段 → 2；空文件 → 2；JSON 语法错 → 2。
- 记录带 `timestamps:true` 时**必须自带可对齐 `language`**；缺失 → 4（capability，不是 schema 2）。
- per-item `timestamps` 只影响该项的模式，**不会**把 CLI 默认 270 s 改成 180 s；
  要时间戳就显式 `--max-chunk-seconds 180`。
- 不要用 CLI 级 `--timestamps` 兜底：它会把所有未写 `timestamps` 的记录切到时间戳模式
  （显式 `false` 仍为 false），缺可对齐语言的记录直接 4。

```bash
qasr --json transcribe --manifest jobs.jsonl -o out/ --max-chunk-seconds 180 --resume
qasr --json transcribe --manifest jobs.jsonl -o out/ --dry-run | jq '.plan.inputs, .plan.outputs'
```

dry-run 会解析每条记录并列出计划，但不会下载 URL、不会检查模型是否存在、不会检查目标是否已存在。

## state / 锁 / job key

输出目录下：

```
<out-dir>/.qasr/state.jsonl   # 每个 item 的最新状态（追加写）
<out-dir>/.qasr/manifest.json # 本次运行的记录
<out-dir>/.qasr/lock          # {pid, host, created_at}
```

- 只允许一个活跃进程写同一 `<out-dir>`；锁被活跃 PID 持有时以 2 拒绝。
  stale 锁（进程不在/非同机）会被忽略；确认进程已死后才可手工删除 `lock`。
- state 记录字段包括 `id`、`job_key`、`status`（`pending/running/done/failed/skipped`）、
  `outputs`（路径+sha256）、`error`、`started_at`、`finished_at`。
- 启动时会把本机遗留的 `running` 记录改写为 `pending`（recover），再按 resume 规则处理。
- 不要手工编辑 `.qasr/` 下的文件；agent 也不应改。

`--resume` 的 job key 由规范化请求参数、音频 SHA-256、模型 fingerprint、aligner fingerprint、
backend、`qasr_version`、相对输出路径共同决定。**不是**“目录存在就跳过”。

一个 item 被跳过需同时满足：

1. state 最新记录为 `done` 或 `skipped`；
2. 记录的 `job_key` 与本次一致；
3. 记录的每个输出仍存在且 SHA-256 匹配。

模型、语言、context、格式、backend 或 qasr 版本变化都会改变 key，从而重跑。
`--overwrite` 会忽略 resume 记录并替换输出；两者同时给出时以 overwrite 为准。

`--skip-existing` 是更弱的检查：只跳过“目标存在且非空”的输出，不校验内容或参数；
全部输入都命中时甚至不会解析/加载模型。

## 失败与重试

- `--continue-on-error` 是默认行为；`--fail-fast` 在首个失败处停止。
- `--retry` 只对 `error.retryable==true` 的失败重试**一次**；不能与 `--fail-fast` 同用（2）。
- 退出码：全部成功 0；有失败且至少一个完成（ok/skipped）→ 6；全部失败 → 首个错误的码；
  `--fail-fast` → 首个错误的码。
- 部分失败时 `.summary.ok > 0` 且 `.ok:false`；不要因为 `.summary.ok` 非零就认为整体成功。
- `error.retryable` 为 false 的（参数、输入、能力）不要重试，改参数或换输入。

## 批量 JSON 与 jobs

batch envelope：

```json
{"schema_version":1,"ok":false,"command":"transcribe",
 "summary":{"total":3,"ok":2,"failed":1,"skipped":0},
 "items":[],"errors":[{"id":"b","source":"b.wav","error":{"kind":"input","code":4,"message":"...","retryable":false,"details":{}}}]}
```

- 单输入没有 `.summary`；看顶层 `.ok` / `.skipped`。
- `--jsonl` 只有逐条 item 行，没有 summary；`done: ...` 只在写文件模式的人类 stderr 出现。
- `summary.total` 包含 preflight 就失败的 spec，因此一定等于 ok+failed+skipped；
  但 `jobs` 的 `total` 也包含 spec_errors，而它们不写 state，故五类 counts 之和可能小于 `job.total`。
- 进度、警告、`done: ...` 都在 stderr；`2>/dev/null` 不应改变 stdout 的 JSON。

```bash
qasr --json jobs list --limit 20 | jq '.jobs[] | {job_id, out_dir, state, total}'
qasr --json jobs show JOB_ID | jq '{state, counts, first_error, state_path}'
qasr --json jobs show out/ | jq '.counts'          # out-dir 必须已登记
```

- `jobs show` 接受 job id 或已登记的 out-dir；未知目标 → 4。
- `.state` 只有 `present|missing`（missing = state 文件不在了，不代表任务没跑过）。
- `.counts` 为 `pending/running/done/failed/skipped`；`pending==0 && running==0` 且 `state=="present"` 才算落定。
- `.first_error` 是首个 failed 项的 `{id,error}`；完整错误仍在 state 与当时的 stdout envelope。

## 并发

- 本地后端强制 `--jobs 1`；并行只来自 `--batch-size`（vLLM 默认 8，transformers 默认 1）。
- 远程 `--jobs 1..32`（默认 4）；`--batch-size` 在远程恒为 1。
- 多进程写同一 out-dir 会被 lock 拒绝；需要并行时分目录，或只用远程 jobs。
