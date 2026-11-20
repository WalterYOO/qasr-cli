何时读我：要做单文件转写/强制对齐，需要完整参数、格式矩阵、JSON 字段或字幕断句规则时。

# transcribe / align：命令与输出契约

版本基线 `qasr 0.3.4`。全局 flag（`--json`、`--backend`、`--model`、`--server`、`--dry-run` 等）
放在 `qasr` 之后、子命令之前，或叶子命令之后；不要放在 `models`/`config` 这类中间层。

## transcribe

```
qasr [全局 flag] transcribe [选项] INPUT...
```

`INPUT` 接受文件、目录、glob、`http(s)://` URL；多输入另见 `batch-and-resume.md`。
模型必须先存在（`models list` 的 `status=="complete"`），缺模型直接 3，不会自动下载。

### 选项分组

| 组 | 选项 | 要点 |
|---|---|---|
| 模型/后端 | `--model`、`--aligner auto\|none\|PATH`、`--backend auto\|transformers\|vllm\|remote`、`--device`、`--dtype`、`--attn-implementation`、`--model-dir`、`--offline`、`--server` | 全局 flag；`--server` 隐含 remote |
| 输出 | `-f/--format LIST`、`-o/--out-dir DIR`、`--output PATH\|-` | `-o` 与 `--output` 互斥；多格式写文件用 `-o`，`--output` 单格式 |
| 语言 | `-l/--language` | 默认 `auto`（LID）；时间戳时必须显式且可对齐 |
| 上下文 | `--context TEXT`（可重复）、`--context-file FILE` | 多个片段按空格合并、去重；文件每行一个、`#` 注释 |
| 时间戳 | `--timestamps`、`--granularity sentence\|word\|char` | 默认 `sentence`；需 aligner 模型 |
| 批量 | `--jobs`、`--batch-size`、`--max-inference-batch-size`、`--timeout`、`--retries` | 本地 jobs 恒 1；远程 jobs 1–32 |
| 长音频 | `--max-chunk-seconds SECONDS` | 默认 270；带时间戳默认/上限 180；范围 10–1200 |
| 输入模式 | `--recursive`、`--stdin-list`、`--manifest FILE` | manifest 与位置参数/stdin-list 互斥 |
| 续跑 | `--resume`、`--overwrite`、`--skip-existing`、`--retry`、`--fail-fast`、`--continue-on-error` | `--retry` 与 `--fail-fast` 互斥（2） |
| 其他 | `--max-new-tokens auto\|N`、`--no-progress`、`--quiet`、`--verbose` | 进度只进 stderr |

### 语言

- `auto`：模型做 LID，`.result.language_source=="auto"`。
- 显式 `-l`：30 种可强制语言（iso 或别名，如 `zh`/`中文`），`.result.language_source=="forced"`。
- `transcribe --timestamps`：只接受 11 种可对齐语言 `zh en yue fr de it ja ko pt ru es`；`auto` → 4。
  `align` 不同：`-l auto` **合法**（先按文稿脚本检测，失败再做音频 LID）。
- 22 个中文方言标签（如 `Sichuan`、`Cantonese (Hong Kong)`）只是 LID 输出，不能 `-l` 强制；
  强制它们会 4。粤语 `yue` 既可强制也可对齐。
- 未知语言（`-l xx`）是 4（input），不是 2。

### 格式与 stdout

`FORMATS = txt, json, jsonl, srt, vtt, tsv, md`（help 只列前五种，`tsv/md` 仍可用）。

| 场景 | 行为 |
|---|---|
| `--json` | stdout 一个对象；单输入 item envelope，多输入 batch envelope |
| `--jsonl` | 仅 transcribe；每条 item 一行，无 summary；`done:` 只在写文件模式的人类 stderr |
| `-o DIR` / `--output PATH` | 写文件，stdout 通常无内容（`--json` 仍打印 envelope） |
| `--output -` | 单输入、单格式，内容写 stdout |
| 无输出选项 + 单输入 | 默认 `txt` 文本写 stdout；指定 `srt/vtt/tsv/md` 时写该格式 |
| 多输入无 `--json`/`--jsonl` | 必须 `-o`，否则 2 |

互斥规则（transcribe 专属）：

- `--json` 与 `-f json/jsonl/tsv/md` → 2；与 `-f txt/srt/vtt` 合法。
- `-f srt`/`-f vtt` 必须同时 `--timestamps` 和可对齐语言，否则 4。
- 无 `-o` 且无 `--json/--jsonl` 时多格式（`-f srt,vtt`）是 2；`--json` 下多格式合法，
  **无 `-o`** 时只输出 envelope、不落盘，给 `-o` 仍写多个文件（已存在则 4）；`--output` 只接受一个格式（2）。
- `--output` 只接受一个输入（2）。
- 既有目标文件默认拒绝（4）；`--overwrite` 或 `--resume` 才允许替换。
- `--skip-existing` 只跳过“已存在且非空”的输出，不代表结果正确。

### 单文件 JSON envelope

```json
{
  "schema_version": 1, "ok": true, "command": "transcribe",
  "input": {"id": "a", "source": "a.wav", "duration_s": 12.3, "sha256": "..."},
  "model": {"id": "Qwen/Qwen3-ASR-1.7B", "backend": "transformers",
            "device": "mps", "dtype": "fp16", "fingerprint": "sha256:..."},
  "result": {"text": "...", "language": "Chinese", "language_source": "forced",
             "segments": [{"text": "...", "start": 0.0, "end": 1.2, "kind": "sentence"}],
             "timestamp_source": "forced_aligner", "text_transform": "...", "warnings": []},
  "metrics": {"load_ms": 0, "infer_ms": 0, "wall_ms": 0, "rtf": 0.0},
  "outputs": [{"format": "srt", "path": "out/a.srt", "sha256": "..."}]
}
```

常用 jq：

```bash
qasr --json transcribe a.wav | jq -r '.result.text'
qasr --json transcribe a.wav | jq '{lang: .result.language, src: .result.language_source, warn: .result.warnings}'
qasr --json transcribe a.wav -l zh --timestamps | jq '.result.segments[:3]'
qasr --json transcribe ./audio -o out/ --resume | jq '.errors[] | {id, kind: .error.kind, retryable: .error.retryable}'
```

注意：

- `.result.segments` 无时间戳时通常为空；`timestamp_source=="forced_aligner"` 才可信。
- `.result.warnings` 可能含 `chunk_boundary_text_uncertain` 等；不要只看 `.ok`。
- `input.duration_s` 为 `null`（如 URL dry-run）或 0 时，先确认音频不是空样本。
- `outputs` 是**写文件后**组装的；sidecar `out/*.json` 里的 `outputs` 可能为空，以 stdout 为准。
- `--jsonl` 没有 summary；批次统计只在**写文件模式**（stdout 无内容）的人类 stderr `done: ...` 行。

## align

```
qasr [全局 flag] align AUDIO (--text TEXT | --text-file FILE | --srt FILE) -l LANG [选项]
```

- 三者恰好一个；缺/重复 → 2。参考文本为空 → 4。
- `--language` 必须可解析（CLI 或 config）；缺失 → 2。`auto` 会用文字脚本或音频 LID 探测，
  `.language_source` 为 `script` 或 `lid_probe`；显式语言为 `explicit`。
- 默认 `--granularity word`；默认输出格式 `json`；`--coverage-min` 默认 0.5（0 关闭门禁）。
- 默认窗口 `--window-seconds 180`、`--window-overlap 3`；范围 10–180 / 0–10 且 overlap < window。
  长音频自动窗口化；窗口 >1 或语言为 `auto` 时需要 ASR 模型做锚点/LID。
- **没有 `--overwrite`**：目标存在即原子覆盖。先 `--dry-run` 看 `.plan.outputs[].path`，
  取得用户确认或换新目录。
- `--srt` 会解析草稿 SRT 的文本作为参考，重新对齐/重定时；时间码来自新对齐结果。
- 覆盖率低于 `--coverage-min` → 5，且在 `_write_outputs` 之前抛错：磁盘不会留下本次结果。

### align JSON

```json
{
  "schema_version": 1, "ok": true, "command": "align",
  "input": {"source": "a.wav", "duration_s": 12.3},
  "language": "Chinese", "language_source": "explicit",
  "window_plan": {"window_seconds": 180, "window_overlap": 3, "count": 1, "windows": []},
  "result": {"items": [{"text": "...", "start": 0.0, "end": 0.4}],
             "coverage": 0.97, "warnings": []},
  "metrics": {"load_ms": 0, "infer_ms": 0},
  "outputs": [{"format": "srt", "path": "out/a.srt", "sha256": "..."}]
}
```

顶层 `warnings` 仅在非空时出现；`result.warnings` 总是数组。
`align` 的 `-f json` 与 `--json` **不冲突**：`-f` 描述写文件格式，`--json` 决定 stdout；
`align` 的 `-f` 同样支持 `txt,json,jsonl,srt,vtt,tsv,md`（help 只列 `json,srt,vtt,txt`）；全局 `--jsonl` 仍为 2。`-f` 多格式需要 `-o`；`--output` 只接受单格式。

```bash
qasr --json align a.wav --text-file script.txt -l zh -f srt -o out/ --dry-run
qasr --json align a.wav --text-file script.txt -l zh -f srt -o out/
qasr --json align a.wav --srt draft.srt -l en -f srt -o out/ --coverage-min 0.8
qasr --json align a.wav --text-file t.txt -l zh --coverage-min 0.8 \
  | jq -e '.ok == true and .result.coverage >= 0.8'
```

## 字幕断句规则（SRT/VTT）

- 单条 cue 上限 **7 s**、显示宽度 **42 列**；CJK 字符按 2 列计，拉丁词不拆。
- 优先在句末标点（`。！？!?；;`）断开；否则在 ≥ **700 ms** 的停顿处断开；再否则按长度切。
- 秒→毫秒只在入口转换一次，**half-up** 取整（`floor(ms + 0.5)`）。
- cue 强制单调、最小 1 ms；超出音频时长会截断；没有空间时丢弃末尾 cue。
- SRT 时间码用逗号 `00:00:01,234`，VTT 用点并带 `WEBVTT` 头。
- 没有时间戳 segment 时写 srt/vtt 会 4；空音频（零样本）可成功但文本为空，需看 `duration_s`。

## 相关退出码

| code | transcribe 场景 | align 场景 |
|---|---|---|
| 2 | 参数互斥、缺 out-dir、manifest schema | 缺 `--language`、参考文本源不唯一、窗口越界 |
| 3 | 模型/aligner 缺失、torch/vLLM 不可用 | aligner/ASR 缺失、运行时失败 |
| 4 | 输入不存在、语言非法、srt 无时间戳、目标已存在 | 音频/文本不存在、语言不可对齐、空文本 |
| 5 | 不会发生（transcribe 没有质量门禁） | coverage 低于阈值 |
