# qasr-cli P0 独立审查报告

> 审查者：Codex Partner（独立于实现）
> 日期：2026-11-18
> 审查对象：`6348abe` M1、`6174a96` M2、`2f36b37` M3、`e5593f7` M4、`57044b6` skipped 修复、`e7364e1` 复审修复、`1b80363` 断管修复
> 依据：`DESIGN.md` v0.3、`P0-TASKS.md`
> 结论：**PASS** — BLOCKER 0 / MAJOR 0 / MINOR 0
> 放行结论：全部初审问题已修复并复核，无阻塞项。

---

## 0. 摘要

| ID | 严重度 | 初审 | 复审（`e7364e1` / `1b80363`） | 标题 | 主要位置 |
|---|---|---|---|---|
| MAJOR-1 | MAJOR | 必须修 | fixed | `--fail-fast` 提前退出后，未处理项永久停在 `running`，汇总把未处理项算成 ok | `qasr_cli/transcribe.py` |
| MAJOR-2 | MAJOR | 必须修 | fixed | 配置里的 `context` / `format` 被静默忽略 | `qasr_cli/commands/transcribe.py` |
| MAJOR-3 | MAJOR | 必须修 | fixed | manifest 中不同原始 id 经 sanitize 后撞名，`--overwrite` 下静默覆盖，两个结果变一个 | `qasr_cli/jobs.py`、`qasr_cli/inputs.py`、`qasr_cli/transcribe.py` |
| MAJOR-4 | MAJOR | 必须修 | fixed | 单输入多格式输出到 stdout 时 rc=0、stdout 为空，第二个格式被静默丢弃 | `qasr_cli/commands/transcribe.py` |
| MINOR-5 | MINOR | 建议修 | fixed | cue 可越过音频时长 1 ms | `qasr_cli/subtitles.py` |
| MINOR-6 | MINOR | 建议修 | fixed | 空音频 + `--timestamps` 仍要求 aligner | `qasr_cli/commands/transcribe.py` |
| MINOR-7 | MINOR | 建议修 | fixed | 下游关闭管道时 exit 120 + `BrokenPipeError` traceback | `qasr_cli/cli.py` |
| MINOR-8 | MINOR | 建议修 | fixed | `align --dry-run` 仍要求本地 aligner | `qasr_cli/commands/align.py` |
| MINOR-9 | MINOR | 建议修 | fixed | `qasr models` 缺 action 返回模型错误 3，而非用法错误 2 | `qasr_cli/commands/models.py` |
| MINOR-10 | MINOR | 待确认 | fixed | `--granularity sentence` 不聚合 JSON segments，真实中文仍输出 13 个 `kind=char` | `qasr_cli/backends.py` |
| MINOR-11 | MINOR | 设计欠账 | fixed | `models pull --resume` 解析后未生效；未写 `source.json` | `qasr_cli/catalog.py` |

**BLOCKER：无。**

主链路（无 torch 启动、模型校验、转写、时间戳、SRT/VTT、真实对齐、resume、kill -9、锁）整体可用；初审的 4 个 MAJOR 均为边界契约问题且已全部修复，不是“跑不起来”的问题。

---

## 1. 审查范围与方法

### 1.1 范围

- 阅读 `DESIGN.md`、`P0-TASKS.md`、`qasr_cli/`、`schemas/`、`tests/`、`README.md`。
- 以黑盒 `subprocess` 为主：`python -m qasr_cli` 和 `pytest review_tests/`。
- 对纯函数（如 cue 构造）使用最小代码级复现作为诊断证据，但不替代黑盒验收。
- 未修改 `qasr_cli/`、`tests/`、`schemas/`、`pyproject.toml`、`DESIGN.md`、`P0-TASKS.md`。

### 1.2 环境

| 项 | 值 |
|---|---|
| OS | macOS 26.6.2 arm64 |
| Python（主要测试） | 3.12.11，`/tmp/qasr-m1-venv/bin/python`（无 torch） |
| 运行时集成 | `/Users/walter/workspace/QwenLM/qwen-tts-cli/.venv/bin/python` 的 site-packages |
| 模型 | `/Users/walter/workspace/QwenLM/Qwen3-ASR/models/{Qwen3-ASR-0.6B,Qwen3-ASR-1.7B,Qwen3-ForcedAligner-0.6B}` |
| 集成音频 | `/tmp/qasr_asr_zh.wav` |

### 1.3 测试结果

```text
review_tests -m "not integration"
=> 166 passed, 1 skipped, 16 deselected in 9.69s

tests -m "not integration"
=> 197 passed, 6 deselected in 0.90s

review_tests -m integration（真实 0.6B + aligner）
=> 16 passed, 167 deselected in 95.63s

tests -m integration
=> 6 passed, 197 deselected in 11.51s

ruff check qasr_cli schemas tests review_tests
=> All checks passed!
```

复审后 `review_tests/` 中已无任何 `xfail` / strict 标记（`grep -rn xfail review_tests/` 无结果），此前对应 MAJOR-3、MINOR-6 等修复项的 5 个 strict XPASS 已全部改为正常断言并通过。

> 注：§6 的补测现已全部落地（config format/context、多格式 stdout、无 `--overwrite` 的 id 撞名、`--fail-fast` 状态），见 §6 与 §7。

无 torch 启动冒烟全部通过（stdout 为结果，stderr 0 字节）：

| 命令 | rc |
|---|---|
| `qasr --help` | 0 |
| `qasr --version` | 0 |
| `qasr languages --json` | 0 |
| `qasr config show --json` | 0 |
| `qasr models verify 1.7b --json` | 0 |
| `qasr --dry-run transcribe ...` | 0 |
| `qasr doctor --json` | 0（缺 torch 时 status=fail，但默认退出码仍为 0） |
| `qasr doctor --strict --json` | 3（符合预期） |

另外：`python -c "import qasr_cli.cli; ..."` 确认 `torch` / `qwen_asr` 未进入 `sys.modules`。

### 1.4 复现用 fixture

0 样本合法 WAV（44 字节，用于空音频相关用例）：

```bash
/tmp/qasr-m1-venv/bin/python - <<'PY'
import wave
with wave.open("/tmp/qasr-empty.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
PY
```

非空中文音频使用集成 fixture `/tmp/qasr_asr_zh.wav`（约 5 秒）。

---

## 2. MAJOR

### MAJOR-1：`--fail-fast` 提前退出后，未处理项停在 `running`，汇总虚报 ok

**类型**：实现 bug。

**位置**

- `qasr_cli/transcribe.py:463`、`:477`：检测到任意 `spec.error` 后直接 `break`，没有把剩余项落为终态。
- `qasr_cli/transcribe.py:563`：`ok_count` 仅看 `spec.error is None and not spec.skipped`，把“根本没处理”的项也算 ok。
- `qasr_cli/transcribe.py:583`：`RunResult.summary` 使用上述 `ok_count`。
- `qasr_cli/transcribe.py:702-716`：`_fail_item` 只设置 `spec.error`，不写 state；失败终态依赖后面的 collect 循环，而 fail-fast 跳过的项没有 collect 机会。
- `qasr_cli/transcribe.py:418-435`：即使 fail-fast 在循环入口就退出，也会先加载模型，造成无意义等待。

**复现**

```bash
rm -rf /tmp/qasr-ff && mkdir -p /tmp/qasr-ff && cd /tmp/qasr-ff
: > bad0.wav
printf '%s\n' \
  '{"id":"bad","audio":"/tmp/qasr-ff/bad0.wav","formats":["txt"]}' \
  '{"id":"good","audio":"/tmp/qasr_asr_zh.wav","formats":["txt"]}' > ff3.jsonl

NUMBA_CACHE_DIR=/tmp/numba-cache \
PYTHONPATH=/Users/walter/workspace/QwenLM/qwen-tts-cli/.venv/lib/python3.12/site-packages:. \
/tmp/qasr-m1-venv/bin/python -m qasr_cli transcribe \
  --manifest ff3.jsonl \
  --model /Users/walter/workspace/QwenLM/Qwen3-ASR/models/Qwen3-ASR-0.6B \
  -o out-ff4 --fail-fast
```

**实际**

```text
rc=6
stderr: done: 1 ok, 1 failed, 0 skipped
state: [('good', 'running'), ('bad', 'failed')]
out-ff4/ 下只有 .qasr/，没有 good.txt
```

`--json` 模式下 payload summary 是 `ok=0`，但 stderr 汇总和内部 `RunResult.summary` 是 `ok=1`；用户看到“1 ok”，实际没有任何 good 输出。更严重的是 `good` 永久停在 `running`，后续 resume 虽会把它恢复为 pending，但本轮状态机已经不自洽。

**期望**

- fail-fast 退出时，未处理项必须是明确的 `skipped` 或 `pending` 终态，并写入 state，不能留 `running`。
- `RunResult.summary`、payload `summary`、stderr 汇总必须来自同一组“已处理/已跳过/失败”计数。
- fail-fast 若在入口即确定不处理任何项，不应先加载模型。

**最小修复建议**

1. 在 `run()` 中把 `pending` 拆成 `processed` / `not_started`；break 后把 `not_started` 显式标为 `STATUS_SKIPPED` 或写一条 `pending` 终态记录。
2. `ok_count` 只统计 `STATUS_DONE` 的 spec；`total` 仍为请求总数。
3. 把 `break` 判断提前到加载模型之前；若 `fail_fast` 且已有 `spec.error`，直接跳过 backend 构造。

---

### MAJOR-2：配置中的 `context` / `format` 被静默忽略

**类型**：实现 bug（配置消费缺失）。

**位置**

- `qasr_cli/commands/transcribe.py:88-91`：`context` 只从 CLI `ns.context` / `--context-file` 读取，完全没有读 `cfg["context"]`。
- `qasr_cli/commands/transcribe.py:130-131`：格式只从 CLI `--format` 或 `_default_formats_hint(ns)` 读取。
- `qasr_cli/commands/transcribe.py:360-368`：`_default_formats_hint` 只看 CLI 参数和 `out_dir/output`，不读 `cfg["format"]`。
- `qasr_cli/config.py:30-31`：`context`、`format` 是合法配置键，类型校验也存在，因此不是“拒绝配置”，而是“接受后不用”。
- `jobs` 配置键同样只在 CLI 路径使用，`server` 是 P1 空实现；本条的复现以 `context/format` 为主。

**复现**

```bash
mkdir -p /tmp/qasr-cfg
cat >/tmp/qasr-cfg/config.json <<'EOF'
{"model":"/nonexistent/model","format":["json"],"context":["hotword"]}
EOF

QASR_CONFIG=/tmp/qasr-cfg/config.json \
/tmp/qasr-m1-venv/bin/python -m qasr_cli --json transcribe \
  /tmp/qasr-empty.wav -o /tmp/qasr-cfg/out
find /tmp/qasr-cfg/out -maxdepth 1 -type f
```

**实际**

- `rc=0`
- 生成的是 `out/empty.txt`，不是配置要求的 `out/empty.json`
- state 中 `job_key=24d3d96c...`
- 显式 `--context=hotword` 时 `job_key=498e2248...`

也就是说，配置里的 context 不会进入 prompt，也不会进入 job key；配置里的 format 不会影响输出文件。用户修改配置后，输出与 resume 缓存都按旧请求执行，属于静默错误。

**期望**

- 配置优先级：内置默认 < `QASR_HOME/config.json` < 项目 `.qasr/config.json` < 环境变量 < `--config` < CLI。
- `context`、`format` 应与其他配置键一样参与合并；CLI 显式值覆盖配置。

**最小修复建议**

1. `context_default` 先取 `cfg.get("context")`，再拼接 CLI `--context` / `--context-file`。
2. `_default_formats_hint` 在无 CLI `--format` 时使用 `cfg.get("format")`，再回退 `default_output_formats()`。
3. 对尚未实现的 `jobs` / `server` 配置键，要么在 P0 明确拒绝并给提示，要么在文档标注“仅 P1 生效”，不要静默接受。

---

### MAJOR-3：manifest 原始 id sanitize 后撞名，`--overwrite` 下静默覆盖

**类型**：实现 bug（输出冲突检测发生在 sanitize 之后，且比较的是 sanitize 后的 id）。

**位置**

- `qasr_cli/jobs.py:339-341`：manifest 只对**原始** `record["id"]` 去重。
- `qasr_cli/inputs.py:344`、`:350`：构造 `InputItem` 时把 id 经过 `sanitize_id()`，`a/b` 和 `a_b` 都变成 `a_b`。
- `qasr_cli/commands/transcribe.py:347`：`output_base` 由 sanitize 后的 `item.id` 生成。
- `qasr_cli/transcribe.py:170-181`：冲突检测比较 `spec.id`；两项的 `spec.id` 都是 `a_b`，因此 `seen[rel] != spec.id` 永远为假，检测不到冲突。

**复现**

```bash
mkdir -p /tmp/qasr-ids
cat >/tmp/qasr-ids/ids.jsonl <<'EOF'
{"id":"a/b","audio":"/tmp/qasr-empty.wav","formats":["txt"]}
{"id":"a_b","audio":"/tmp/qasr-empty.wav","formats":["txt"]}
EOF

/tmp/qasr-m1-venv/bin/python -m qasr_cli --json transcribe \
  --manifest /tmp/qasr-ids/ids.jsonl -o /tmp/qasr-ids/out --overwrite
find /tmp/qasr-ids/out -maxdepth 1 -type f
```

**实际**

```text
rc=0
out/a_b.txt 只有一个文件
payload: summary {"total":2,"ok":2,...}
两个 item 的 input.id 都是 "a_b"
```

不加 `--overwrite` 时：第一个成功、第二个因文件已存在失败，`rc=6`，错误 id 也还是 `a_b`，用户无法区分两个输入。`--overwrite` 下则是 rc=0 的静默数据丢失。

**期望**

- `T4.3`：冲突默认报错，`--overwrite` 只覆盖“上一次运行的旧输出”，不应让同一批中的两个不同输入争用同一个输出文件。
- 冲突检测应基于原始 id 或最终输出路径；sanitize 只用于文件名，不应用于唯一性判断。

**最小修复建议**

1. 在 manifest 解析阶段同时保留 `raw_id`，对 `sanitize_id(raw_id) + output_base + formats` 做冲突检测。
2. 或在 `check_output_conflicts()` 中比较 `(id, 原始来源)`，而不是比较 sanitize 后的 `spec.id`。
3. 对目录镜像输入也做同样检查：不同相对路径 sanitize 后可能落到同一目标。

---

### MAJOR-4：单输入多格式输出到 stdout 时 rc=0、stdout 为空

**类型**：实现 bug（缺少“多格式 + stdout”用法错误）。

**位置**

- `qasr_cli/commands/transcribe.py:211-220`：`total == 1` 时只根据 `selected[0]` 选择 stdout_mode，没有检查 `len(formats) > 1`。
- `qasr_cli/transcribe.py:724-735`：没有 `out_dir` / `output` 时 `_write_outputs()` 直接返回 `[]`，不会写文件；stdout 只渲染 `spec.formats[0]`。
- `DESIGN.md:106` / `P0-TASKS.md T4.3`：多格式 + stdout 必须是用法错误。

**复现**

```bash
/tmp/qasr-m1-venv/bin/python -m qasr_cli \
  transcribe /tmp/qasr-empty.wav -f txt,json
```

**实际**

```text
rc=0
stdout_bytes=0
stderr: ok empty
```

`txt` 被选中为 stdout 模式，`json` 被静默丢弃；因为空文本的 txt 渲染就是空串，所以用户看到“成功但什么都没有”。

**期望**

```text
rc=2
stderr/JSON: 用法错误，提示多格式必须使用 --out-dir/--output，或一次只输出一种格式
```

**最小修复建议**

在 stdout mode 分支中加入：

```python
elif total == 1:
    selected = specs[0].formats if specs else ["txt"]
    if out_dir is None and not output and len(selected) > 1:
        raise UsageError("multiple formats require --out-dir or --output")
```

同时补一个黑盒回归：`transcribe empty.wav -f txt,json` → rc=2。

---

## 3. MINOR

### MINOR-5：cue 可越过音频时长 1 ms

**类型**：实现 bug（边界舍入）。

**位置**：`qasr_cli/subtitles.py:67-70`、`:183-186`。

**复现**

```bash
/tmp/qasr-m1-venv/bin/python -c \
"from qasr_cli.subtitles import build_cues, Segment; print(build_cues([Segment('x',0.9996,1.0)], audio_duration_s=1.0))"
```

**实际**：`[Cue(start_ms=1000, end_ms=1001, text='x')]`，`end_ms` 超过 1000 ms。

**期望**：`end_ms <= duration_ms`，同时保持 `end >= start + 1`。优先在末 cue 把 start 压到 `duration_ms - 1`；只有在契约明确允许时才考虑零时长末 cue。

**最小修复**：在 clamp 时先处理 `start >= duration_ms`，不要把 start 固定到 duration 后再强行加 1 ms。

---

### MINOR-6：空音频 + `--timestamps` 仍要求 aligner

**类型**：实现 bug（空音频短路不完整）。

**位置**：`qasr_cli/commands/transcribe.py:266-267`。

**复现**

```bash
/tmp/qasr-m1-venv/bin/python -m qasr_cli --json \
  --model /nonexistent/model --aligner /nonexistent/aligner \
  transcribe /tmp/qasr-empty.wav --timestamps -l zh
```

**实际**：`rc=3`，`aligner not found: /nonexistent/aligner`。

**期望**：0 样本音频没有可对齐内容，应像普通空音频一样 rc=0、空文本、`language=null`、warning 含 `empty audio`，不需要 aligner。

**最小修复**：`need_align = any(spec.timestamps and not spec.empty for spec in specs)`。

---

### MINOR-7：下游关闭管道时 exit 120 + traceback

**类型**：实现 bug（SIGPIPE/EPIPE 处理）。

**位置**：`qasr_cli/cli.py:301-302`。

**复现**

```bash
rm -rf /tmp/qasr-many && mkdir -p /tmp/qasr-many/many && cd /tmp/qasr-many
for i in $(seq -w 0 89); do cp /tmp/qasr-empty.wav many/$i.wav; done
set -o pipefail
/tmp/qasr-m1-venv/bin/python -m qasr_cli --jsonl transcribe many 2>pipe.err | head -c 100 >/dev/null
echo $?
tail -3 pipe.err
```

**实际**：

```text
120
Exception ignored in: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>
BrokenPipeError: [Errno 32] Broken pipe
```

**期望**：管道消费者提前退出是正常场景；应无 traceback，退出 0 或 130/141。

**最小修复**：在 `main()` 捕获 `BrokenPipeError` 后把 stdout 重定向到 `/dev/null`，再返回 0/130；不要返回模型错误 3。

---

### MINOR-8：`align --dry-run` 仍要求本地 aligner

**类型**：实现 bug / 行为不一致。

**位置**：`qasr_cli/commands/align.py:122-124`。

**复现**

```bash
cd /tmp
QASR_MODEL_DIR=/tmp/nonexistent-models QASR_HOME=/tmp/qasr-dry-home \
/tmp/qasr-m1-venv/bin/python -m qasr_cli --json --dry-run \
  align /tmp/qasr_asr_zh.wav --text hello -l en
```

**实际**：`rc=4`，`no aligner model available`。

**对照**：`transcribe --dry-run --model /nonexistent/model` rc=0，能输出 plan。`DESIGN.md:177` 要求 dry-run 只解析与规划、不加载权重，因此 align 不应因本地权重缺失而失败。

**最小修复**：dry-run 时 `resolve_aligner(..., required=False)`，plan 中写请求的/自动解析的路径或 alias；仅在真正执行时 `require_local`。

---

### MINOR-9：`qasr models` 缺 action 返回 3

**类型**：实现 bug / UX 一致性。

**位置**：`qasr_cli/commands/models.py:32-34`。

**复现**

```bash
QASR_HOME=/tmp/qasr-smoke/home /tmp/qasr-m1-venv/bin/python -m qasr_cli --json models
```

**实际**：`rc=3`，`model error: a model alias is required`。

**期望**：缺子命令属于参数错误，应 `rc=2`、`kind=usage`。对照 `qasr config` 无 action 会默认 `show` 并 rc=0，两者行为也不一致。

**最小修复**：在 action 为空时抛 `UsageError("models requires list|path|verify|pull")`。

---

### MINOR-10：`--granularity sentence` 不聚合 JSON segments

**类型**：实现 bug / 与 DESIGN 文本存在歧义，需确认。

**位置**：`qasr_cli/backends.py:152-172` 只对 `granularity == "char"` 做拆分；`sentence` 直接透传 aligner items。

**复现**

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache \
PYTHONPATH=/Users/walter/workspace/QwenLM/qwen-tts-cli/.venv/lib/python3.12/site-packages:. \
/tmp/qasr-m1-venv/bin/python -m qasr_cli --json transcribe /tmp/qasr_asr_zh.wav \
  --model /Users/walter/workspace/QwenLM/Qwen3-ASR/models/Qwen3-ASR-0.6B \
  --aligner /Users/walter/workspace/QwenLM/Qwen3-ASR/models/Qwen3-ForcedAligner-0.6B \
  --timestamps -l zh --granularity sentence
```

**实际**：`nseg=13`，`kinds={'char': 13}`，第一条是 `{'text':'甚','kind':'char'}`。

**期望**：`DESIGN.md:511-514` 写的是 `sentence` 按标点/停顿聚合成 cue，并“仅影响字幕与 JSON segments”。当前 SRT cue 仍会被 `build_cues()` 聚合，但 JSON `segments` 仍是字符级。

**建议**：若 P0 的 JSON 契约要求 `sentence` 聚合，则在 `segments_from_timestamps()` 增加 sentence 聚合；若认为“sentence 只影响字幕、JSON 保留 aligner 原始粒度”，请修改 DESIGN 措辞并在 README 说明。当前按 MINOR 记录，不阻塞 P0。

---

### MINOR-11：`models pull --resume` 未生效，未写 `source.json`

**类型**：实现欠账 / DESIGN §9.2 有要求，P0-TASKS 未逐条展开。

**位置**：`qasr_cli/catalog.py:499-560`。

**证据**

- `cli.py:166` 解析了 `models pull --resume`，但 `pull_model()` 没有 `resume` 参数，`snapshot_download()` 也没有收到 resume/continue 相关参数。
- `grep -rn "source.json" qasr_cli/` 无结果；`pull_model()` 只返回内存中的 payload，没有把 repo/revision/文件清单/哈希写入模型目录。

**期望**（DESIGN §9.2）：`models pull` 写 `source.json`，并支持 `--resume`。

**建议**：P0 可先明确 `--resume` 为“透传给 huggingface_hub 的 resume_download（若版本支持）”；`source.json` 至少在 pull 成功后写入 repo、revision、文件列表与大小/哈希。若本阶段不做，README 应标注该 flag 尚未生效，避免用户误以为可续传。

---

## 4. 已确认不是缺陷的语义 / 测试口径

以下项已按协调者定稿语义验证，**不作为实现 bug 上报**：

1. **多输入 + `--json`/`--jsonl` 是合法 batch envelope**。  
   实测：`transcribe empty.wav e2.wav --jsonl` → rc=0、2 行、每行一个 item；多输入无任何输出通道才 rc=2。对应 `test_multiple_inputs_with_json_emit_batch_envelope` / `test_multiple_inputs_without_output_channel_are_usage_error` 均通过。
2. **`--format json|jsonl` 与 `--json|--jsonl` 无条件互斥**。  
   即使提供 `-o DIR`，同时使用也 rc=2；`test_invalid_format_combinations` 通过。
3. **0 采样空音频是成功，且不需要模型**。  
   实测 `--model /nonexistent/model` + 0 样本 WAV → rc=0、`text=""`、`language=null`、warning `empty audio`。零字节文件仍是 rc=4，两者区分正确（MINOR-6 只涉及空音频 + timestamps 的 aligner 解析）。
4. **真实 aligner 的 coverage 通常恒为 1.0**。  
   真实 0.6B + aligner 的 `test_align_coverage_full_when_aligner_force_aligns_extra_text` 通过；覆盖率低于阈值 exit 5 由 repo 的 fake-aligner 契约测试覆盖，不要求真实模型制造低覆盖。
5. **此前 7 个 M1 失败是真实契约缺口，已修复**。  
   非法 device/dtype/backend/offline 现在均 rc=2；全输入无效时先报输入 4 再报模型 3。当前 review suite 无回归。
6. **此前 4 个集成失败是测试口径错误，已修正，不是实现 bug**。  
   包括多输入 + `--json`、`--format` 互斥、空音频成功、dry-run 不 import torch 的期望。
7. **transformers/qwen-asr 的日志输出到 stderr 是允许的**。  
   `test_model_stdout_remains_pure_json` 通过；`--json` stdout 可被 `json.loads()` 直接消费。

---

## 5. 与 `P0-TASKS.md` 验收项对照

图例：`PASS` = 已独立验证；`PASS*` = 主路径通过但本项存在上述 MAJOR/MINOR；`PARTIAL` = 环境限制未完全复核。

| 任务 | 验收点 | 结果 | 证据 / 备注 |
|---|---|---|---|
| T1.1 | 骨架、entry point、退出码 0/2/3/4/5/6/130 | PASS | `--help`/`--version` rc=0；错误映射测试通过；无 torch 可启动 |
| T1.2 | `QASR_HOME`、配置优先级、`config show/path` | PASS | `test_m1_config.py` 12 项通过；注意配置**消费**缺陷见 MAJOR-2 |
| T1.3 | 30 可强制 / 22 方言 / 11 对齐、ISO 别名 | PASS | `test_m1_languages.py` 通过；方言强制 rc=4 |
| T1.4 | 别名解析、verify、半包、指纹 | PASS | 0.6b/1.7b/aligner 均 `complete`；合成截断 shard/.aria2 均 `incomplete` + `pull --resume` 提示 |
| T1.5 | doctor 检查项、schema、`--strict` | PASS | `doctor --json` schema 通过；默认 rc=0，`--strict` 缺 torch 时 rc=3 |
| M1 验收 | 无 torch 五项命令、半包、doctor | PASS | 全部独立复跑通过 |
| T2.1 | 文件/目录/glob/URL、排序去重、路径安全 | PASS | `test_m2_inputs_errors.py`：目录排序、递归、glob、symlink loop、URL dry-run/offline 均通过 |
| T2.2 | soundfile 探测、坏文件 InputError、不 import torch | PASS | 缺失/损坏/零字节/目录伪音频均 rc=4；`audio.py` 未 import torch |
| T2.3 | lazy runtime、offline、device/dtype、max_new_tokens auto | PASS | 无 torch 启动；dry-run plan 字段齐全；`--offline` 测试通过 |
| T2.4 | 微批、组失败逐条重试、错误隔离 | PASS | 真实模型 `test_batch_isolates_bad_audio` rc=6，好文件成功 |
| T2.5 | transcribe 参数、JSON/JSONL、stdout 纯净、空音频 | PASS | MAJOR-2/4 复审已修复；多格式 stdout rc=2、配置 format/context 生效；管道 flush 残留见 MINOR-7 |
| M2 验收 | 真实中文转写、多文件 jsonl、坏+好 rc=6、help 不 import torch | PASS | 四项均实测通过；边界契约缺口已在复审修复 |
| T3.1 | `--timestamps` 仅 11 语言、segments、清理标签 | PASS | auto+timestamps rc=4；真实中文 segments 13 项且单调 |
| T3.2 | cue 规则、SRT/VTT、毫秒、granularity | PASS | SRT/VTT 合法、≤7s/42 宽、单调；cue 不再越界；空音频+timestamps rc=0；sentence 聚合为 1 段 |
| T3.3 | align 三种文本源、11 语言、≤180s、coverage gate | PASS | 真实 align 13 items/coverage 1.0；180.5s rc=4；低覆盖 rc=5（fake aligner）；dry-run 不再要求本地 aligner |
| M3 验收 | SRT 合法、align 13 items、低覆盖 rc=5、golden | PASS | 真实集成与契约测试均通过；MINOR 边界已修复 |
| T4.1 | manifest 严格 schema、XOR、id 唯一、output 路径安全 | PASS | 未知字段/重复原始 id/绝对路径/`..`/NUL/base64 均 rc=2 |
| T4.2 | state 状态机、原子写、lock、resume、overwrite | PASS | kill -9、SIGINT 130、锁竞争、resume skip/rerun/tamper 均通过；fail-fast 未处理项落为 `skipped`，无 `running` |
| T4.3 | 输出命名、冲突报错、多格式 stdout | PASS | sanitize 撞名 rc=2 且保留原始 id；多格式 stdout rc=2；无 `--overwrite` 同样拒绝 |
| T4.4 | README、5 个 quickstart、限制、schemas | PASS | README 含安装/转写/批量/对齐/限制；13 个 schema 存在并被 contract 测试校验 |
| T4.5a | `pytest` 纯 CPU 全绿 | PASS | review 166 + repo 197 非集成全绿；集成 16 + 6 全绿；ruff 全绿 |
| T4.5b | `uv tool install --python 3.12 "qasr-cli[local]"` 后 doctor 全绿 | PARTIAL | 现有 `~/.local/bin/qasr` 是 uv tool（base extra），`--version`/`doctor --json` 可运行；但该环境无 torch/qwen-asr，未独立复核 `[local]` 安装后的全绿状态 |
| M4 验收 | 安装、目录 resume、kill -9、半包 verify | PASS | kill -9/resume/verify 均通过；fail-fast 与 id 冲突已修复；安装项仍为 PARTIAL（环境限制） |

---

## 6. 测试覆盖缺口与复审落地

初审列出的补测项已在复审前全部落地，且均为正常断言（无 xfail）：

1. **`--fail-fast` 状态一致性**：`review_tests/test_integration_kill_lock.py:102` `test_fail_fast_leaves_no_running_state`，坏 WAV + 真实 good 音频 + `--fail-fast`，断言 state 无 `running`、stderr 汇总与 payload 一致。
2. **配置消费**：`review_tests/test_m2_config_consumption.py:31` `test_config_format_controls_output_extension`、`:48` `test_config_context_is_base_and_cli_appends_in_order`，断言输出扩展名与 `job_key` 随配置变化。
3. **多格式 stdout**：`review_tests/test_m2_stdout_contract.py:107` `test_multiple_formats_to_stdout_are_usage_error`，`transcribe empty.wav -f txt,json` → rc=2。
4. **sanitize 撞名**：`review_tests/test_m4_manifest_resume.py:79` 与 `:102`，分别覆盖 `--overwrite` 与不带 `--overwrite` 的 rc=2，且断言原始 id 保留、无文件写出。
5. **真实 aligner coverage gate**：真实 aligner 的 coverage 通常为 1.0，exit 5 由 repo 的 fake-aligner 契约测试覆盖；这是测试策略而非覆盖缺口。
6. **MINOR-6**：`review_tests/test_adversarial_audio.py:69` `test_empty_wav_with_timestamps_needs_no_aligner`，xfail 已移除并通过。

此前唯一未覆盖的边界是 MINOR-7 的**块缓冲管道关闭**场景；`1b80363` 已在 `main()` 返回路径与 `_emit_error()` 后显式 flush，并扩充了 contract 测试，复审已实测通过（见 §7）。

---

## 7. 复审结果（`e7364e1` → `1b80363`）

复审以最终 `HEAD=1b80363 fix: flush stdout so block-buffered broken pipes exit quietly` 为准（在 `e7364e1` 基础上补 MINOR-7）。工作树只有未跟踪的 `REVIEW-PLAN.md`、`REVIEW-REPORT.md`、`review_tests/`，实现文件无本地改动。

复审回归结果：

```text
review_tests -m "not integration"  => 166 passed, 1 skipped, 16 deselected
tests -m "not integration"         => 197 passed, 6 deselected
review_tests -m integration        => 16 passed, 167 deselected (95.63s)
tests -m integration               => 6 passed, 197 deselected (11.51s)
ruff check qasr_cli schemas tests review_tests => All checks passed!
```

### 7.1 逐条状态

| ID | 状态 | 复审证据（`e7364e1` / `1b80363`） |
|---|---|---|
| MAJOR-1 | **fixed** | `--fail-fast` 复现：rc=6，stderr `done: 0 ok, 1 failed, 1 skipped`；state 中 `good=skipped`、`bad=failed`，无 `running`；`out-ff5/` 无 `good.txt`；耗时 0.48s，未加载模型（坏输入在 preflight 短路）。原“虚报 1 ok”已变为一致的 0 ok。 |
| MAJOR-2 | **fixed** | 配置 `format:["json"]` → 生成 `out/qasr-empty.json`（不再是 `.txt`）。配置 `context:["hotword"]` 与 CLI `--context=hotword` 的 `job_key` 均为 `ec8f6caa0f7010b14aea6f89ea837fed9826892938ee39e472a29b7f010aba02`，证明配置为底、CLI 追加并去重。 |
| MAJOR-3 | **fixed** | `a/b` + `a_b` manifest：带与不带 `--overwrite` 均 rc=2，`error.message="two inputs map to the same output file: a_b.txt"`，`details.ids=["a/b","a_b"]`、`details.sources` 保留原始来源；输出目录未创建，无静默覆盖。 |
| MAJOR-4 | **fixed** | `transcribe /tmp/qasr-empty.wav -f txt,json` → rc=2，stderr `multiple formats require --out-dir or --output`，stdout 为空且无成功伪报。 |
| MINOR-5 | **fixed** | `build_cues([Segment('x',0.9996,1.0)], audio_duration_s=1.0)` → `[Cue(start_ms=999, end_ms=1000, text='x')]`，`end_ms` 不越界。 |
| MINOR-6 | **fixed** | `--model /nonexistent/model --aligner /nonexistent/aligner transcribe empty.wav --timestamps -l zh` → rc=0，`text=""`、`language=null`、`warnings=["empty audio"]`，未解析 aligner。 |
| MINOR-7 | **fixed** | 90 个空 WAV + `--jsonl ... \| head -c 100`，未设置 `PYTHONUNBUFFERED`、`set -o pipefail`：rc=0，stderr 540 字节仅 `ok ...` 行，`Exception ignored` / `BrokenPipeError` / `Traceback` 计数均为 0。`qasr_cli/cli.py:233-251`、`:282-283`、`:332-342` 增加显式 flush 与 /dev/null 重定向；复审同时跑 `review_tests -m "not integration"` → 166 passed / 1 skipped，ruff 全绿。 |
| MINOR-8 | **fixed** | `QASR_MODEL_DIR=/tmp/nonexistent-models qasr --json --dry-run align ... --text hello -l en` → rc=0，plan 含 `aligner.id="Qwen/Qwen3-ForcedAligner-0.6B"`、`path`、`language`、`formats` 等字段，不要求本地权重。 |
| MINOR-9 | **fixed** | `qasr --json models` → rc=2，`error.kind="usage"`、`message="models requires an action"`、`hint="list \| path \| verify \| pull"`。 |
| MINOR-10 | **fixed** | 真实 0.6B + aligner、`--timestamps --granularity sentence`：`nseg=1`、`kinds={"sentence":1}`，文本 `甚至出现交易几乎停滞的情况`，不再是 13 个 `kind=char`。repo 集成测试也断言默认 JSON 为 sentence 粒度。 |
| MINOR-11 | **fixed** | `qasr_cli/catalog.py:500-529` 增加 `resume_download`（不支持时回退）与 `source.json` 原子写；`pull_model(..., resume=True)` 已由 `commands/models.py:63-68` 传入。`pytest tests/unit/test_catalog.py -k "resume or source"` → 2 passed。 |

### 7.2 修复项回归说明

- 原 5 个 strict XPASS / xfail 已全部移除，`grep -rn xfail review_tests/` 无结果，相关用例改为正常断言并通过。
- §6 的 4 个黑盒补测均已落地：fail-fast 状态、配置消费、多格式 stdout、sanitize 撞名（含无 `--overwrite`）。
- 实现侧新增 `tests/contract/test_review_fixes.py`（11 例）覆盖这些契约；复审未发现修复引入的回归。
- 对抗用例（kill -9 续跑、锁竞争、路径逃逸、坏音频、180s 边界、非法配置、stdout 污染、无 torch 启动、块缓冲管道关闭）均由 `review_tests` 全绿覆盖。
- MINOR-7 的块缓冲断管场景已在 `1b80363` 修复，无残留项。

---

## 8. 最终判定

**PASS** — BLOCKER 0 / MAJOR 0 / MINOR 0。

P0 的验收项 T1.1–T4.4、T4.5a 均已独立复核通过：无 torch 启动、模型校验（含合成半包 fixture）、中文转写、时间戳/字幕、真实强制对齐、批处理隔离、state/resume/kill -9、锁与原子写、路径安全、配置消费、输出冲突检测均符合 `DESIGN.md` v0.3 与 `P0-TASKS.md`。

残留项与说明：

1. **MINOR-7 已关闭**：`1b80363` 修复后，原 rc=120 + `BrokenPipeError` 场景实测 rc=0 且 stderr 干净。
2. **T4.5b** 仍为 PARTIAL：现有 `~/.local/bin/qasr` 是无 torch 的 uv tool，`qasr --version` / `doctor --json` 可运行；`[local]` extra 的完整安装环境未由审查者独立复验，属环境限制而非实现缺陷。

**放行建议**：P0 可放行，无待修审查项。
