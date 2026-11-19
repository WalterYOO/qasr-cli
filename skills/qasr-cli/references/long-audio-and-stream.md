何时读我：音频很长（几十分钟/小时级）、时间戳跨块异常，或要用 `qasr stream` 做实时/管道转写时。

# 长音频切块与流式转写

版本基线 `qasr 0.3.0`。0.3.0 **没有 VAD**：不做静音检测、不按说话人切块；切块只看时间窗口。
实时流式也只在 Linux + CUDA + `[vllm]` 下可用。

## transcribe 的切块阈值

| 场景 | 默认 chunk | 硬限制 |
|---|---|---|
| 纯文本（无 `--timestamps`） | 270 s | `--max-chunk-seconds` 10–1200 |
| 带 `--timestamps` | 180 s | 不得超过 180（>180 → exit 4） |
| 远程 `--server` | 270 s | 最多 600 s；超出会降为 600 并 warning |
| `align` 窗口 | 180 s | `--window-seconds` 10–180 |

```bash
qasr --json transcribe long.wav -l zh --max-chunk-seconds 270 -o out/
qasr --json transcribe long.wav -l zh --timestamps --max-chunk-seconds 180 -f srt -o out/
qasr --json transcribe long.wav -l zh --dry-run | jq '.plan.inputs[0] | {duration_s, windows}'
qasr --json align long.wav --text-file long.txt -l zh -f srt -o out/ --dry-run \
  | jq '.plan.windows'
```

- 纯文本多窗口时，结果可能带 `warnings:["chunk_boundary_text_uncertain"]`：跨块文本拼接不确定。
- 带时间戳的长音频由 ASR 产生窗口锚点，再在每个窗口内做强制对齐；可能出现
  `chunk_reference_match_low:<index>:<ratio>`、`chunk_align_failed:<index>:<message>`。
  读 `.result.warnings[]`，必要时缩短 chunk 或提高音频质量后重跑。
- `align` 自动窗口化，通常不需要手工切块；窗口重叠是左侧上下文，不是重复输出。
- 单文件 OOM：降 `--max-chunk-seconds` 或换 `0.6b`。**不要**降 `--batch-size`——
  单文件不经过 batching，改它无效。
- macOS/MPS 长音频更容易显存/算子受限：优先 CPU 慢跑或 0.6b，先 dry-run 看窗口数。
- 远程上传按 25 MB 限制；600 s 是服务端体积与超时的折中，超过只会 warning 并降为 600。

## align 窗口

```bash
qasr --json align long.wav --text-file long.txt -l zh -f srt -o out/ \
  --window-seconds 120 --window-overlap 5
```

- `--window-seconds` 10–180，默认 180；`--window-overlap` 0–10，默认 3，且必须小于窗口。
- 窗口 >1 时需要 ASR 模型提供锚点；`-l auto` 时先按文稿脚本检测，失败才用 ASR 做音频 LID。
  只有单窗口且语言显式时才可只用 aligner。
- `--coverage-min` 默认 0.5；低于阈值 exit 5，且**不会写任何输出文件**。
- `align` 没有 `--overwrite`；重复跑会覆盖同名输出，先 dry-run 看 `.plan.outputs[]`。

## stream

```bash
qasr stream --input a.wav --emit jsonl                  # Linux+CUDA，文件读到 EOF
qasr stream --input mic --emit text                     # 需 [mic] + sounddevice
arecord -q -f S16_LE -r 16000 -c 1 \
  | qasr stream --input stdin --pcm s16le --emit jsonl  # 原始 PCM 管道
```

### 前置与输入

- 仅本地 vLLM：Linux + CUDA + `qasr-cli[vllm]`；`--backend transformers|remote` → 4。
- 模型必须本地完整存在（`status=="complete"`），缺模型 → 3。
- `--input` 必填，取值 `mic | stdin | 本地文件`；`-` 不是音频源（2）。
- URL → 4；目录/不存在的文件 → 4（在能力检查之后）。
- `--input stdin` 必须 `--pcm s16le`；`--pcm`/`--sample-rate`/`--channels` 只对 stdin 合法（否则 2）。
- 采样率固定 16000（`--sample-rate` 写别的值 → 2）；`--channels 1|2`。
- `--chunk-ms` 默认 2000，范围 100–10000 且必须是 10 的倍数。
- `--unfixed-chunks` 默认 2、`--unfixed-tokens` 默认 5（尾部未固定窗口）。
- `--realtime` 只对文件输入有效（按墙钟节奏喂块），mic/stdin 用会 2。
- 语言用 `-l` 时只接受 30 种可强制语言；方言标签会 4。默认 `auto` 交给流式模型。
- `--context` 可重复；`--context-file` 与 `--context` **互斥**（与 transcribe 不同）。
  文件每行一个、`#` 注释、按空格合并。

### 输出语义

- `--json` 与 `--dry-run` 对 stream 无条件 exit 2；机器输出只用 `--emit jsonl`。
- `--emit jsonl`：每行一个累计快照，`seq` 从 0 连续递增；关键字段
  `schema_version ok command seq audio_s elapsed_s language text is_final`；正常结束只有最后一条 `is_final:true`。
- 快照是**累计文本**，后续可能用更短/修正后的文本替换前面的快照；不要逐行拼接。
- `--emit text`：TTY 下用 `\r` 实时刷新，结束时换行；非 TTY（管道/文件）只输出最终文本。
- 文件输入读到 EOF 就结束，**不是 `tail -f`**；要跟随增长的数据流请用 `--input stdin`。
- `--max-stream-seconds` 默认 3600；到点 exit 5，这是运行上限，不是识别质量结论。
- Ctrl-C：先尽量发出最终快照，再以 130 结束。
- 启动前错误（参数/能力/模型）走普通错误路径：`--emit jsonl` 时若尚无输出行，错误在 stderr；
  已开始输出后的错误会给一条 `ok:false`+`error` 的终止快照（`--emit text` 只写 stderr）。
- `stream` 不写任何文件产物；需要文件请改用 `transcribe`。

### 检查

```bash
qasr stream --input a.wav --emit jsonl \
  | jq -e 'select(.ok == false) // empty'          # 有错误快照就打印
qasr stream --input a.wav --emit jsonl \
  | jq -s '{last: .[-1], monotonic: ([.[].seq] == ([.[].seq] | sort))}'
```

- 正常流：`seq` 连续、最后一条 `is_final:true`、没有 `ok:false`。
- 文本可能回滚（累计快照被更短文本替换）是设计行为，不是丢字。
