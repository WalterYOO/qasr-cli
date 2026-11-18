# qasr-cli P2 独立审查报告

> 审查者：Codex（独立于实现方）
> 基线：`21f111a`（最终 stream 代码 `1f7788f`；mic 预检 `ac526bb`；
> catalog 文案 `3df825e`；gpu06 证据文档 `21f111a`）
> 契约：`P2-TASKS.md`（`522a130` §7 Q1–Q16 裁决）＋ `DESIGN.md` v0.3
> 方法：干净 `git archive` 快照 + `subprocess` 黑盒；不 import `qasr_cli`
> 状态：**PASS**

## 1. 结论

| 级别 | 数量 | 摘要 |
|---|---:|---|
| BLOCKER | 0 | 无 |
| MAJOR | 0 | 无 |
| MINOR | 0 | 3 项已全部修复并复验：`ac526bb`（mic 预检）、`1f7788f`（非有限 cap + elapsed_s） |

主链路（S1–S4、C、F、schema/打包）已通过黑盒验收；gpu06 的
file/stdin/cap/SIGINT/能力/中文 realtime 证据与契约一致。上述 MINOR 不影响
P0/P1 行为，也不造成数据或安全风险；修复后原复现命令全部转绿。

## 2. 审查范围与方法

### 2.1 范围

- S1：`stream` 参数矩阵、默认值、互斥、能力/模型优先级、退出码与 hint。
- S2：file / stdin s16le / mic；成帧、残字节、双声道下混、EOF、realtime。
- S3：累计快照、`seq`、唯一 `is_final`、cap 单条错误行、人类/TTY 模式。
- S4：stdout 纯净、无 state/jobs、无 token 泄漏、无重运行时 import。
- C：bash/zsh/fish completion 的覆盖、语法、静态性与冲突。
- F：`finetune lint-jsonl` 字段、语言前缀、strict、重复 audio、schema。
- §5：`[mic]`/`[all]` extras、README、P2-REPORT 交付说明。

### 2.2 非范围

- 不修改 `qasr_cli/`、`schemas/`、`tests/`、`README.md`、`P2-TASKS.md`、
  `P2-REPORT.md`；不 commit。
- 不操作 gpu06、不向服务器传文件；GPU 证据只复核协调者记录。
- `review_tests_p2` 中标记 `simulated` 的用例用子进程 `PYTHONPATH` 注入
  fake `torch/vllm/qwen_asr/sounddevice` 与 `sitecustomize` 模拟 Linux，
  **只作本机补充证据，不是 gpu06 证据**。

### 2.3 环境

- 解释器：`/tmp/qasr-m1-venv/bin/python`（torch-free：pytest / jsonschema /
  ruff / numpy / soundfile）。
- 快照：`git archive 21f111a | tar -x -C /tmp/p2-review-21f111a`，再复制
  `review_tests_p2/`；`PYTHONPATH` 强制指向快照根，避免 editable install
  干扰。
- 需要 loopback 的仓库/P1 契约测试用 escalated 运行；其余在沙箱内完成。

### 2.4 凭据检查

- 已扫描 `review_tests_p2/`、`P2-REVIEW-PLAN.md`、`P2-REVIEW-REPORT.md`
  的 URL userinfo 与高熵字面量：唯一真实 userinfo 位于
  `review_tests_p2/test_p2_completion.py:151`，协调者已就地替换为
  `user:fake@`（4 字符占位）；其余无凭据。
- 另扫描 `review_tests_p1/`：命中的 `user:***@` 均为 f-string
  `{secret}` 插值，不是硬编码凭据。

## 3. 执行结果

| 套件 | 命令 | 结果 |
|---|---|---|
| P2 独立套件 | `pytest -q review_tests_p2 -m "not gpu"` | **187 passed / 6 deselected / 0 xfailed** |
| 仓库测试 | `pytest -q tests` | **420 passed / 6 skipped** |
| P0 回归 | `pytest -q review_tests -m "not integration and not gpu"` | **164 passed / 1 skipped / 16 deselected** |
| P1 回归 | `pytest -q review_tests_p1 -m "not integration and not gpu"` | **204 passed / 8 deselected** |
| 静态检查 | `ruff check qasr_cli tests review_tests_p2` | **All checks passed** |

`6 deselected` 是 `-m gpu` 用例（本机无 CUDA/vLLM）。仓库 `6 skipped` 是
`RUN_INTEGRATION=1` 未设置。所有 strict xfail 已移除，无 XPASS 残留。

补充直接黑盒证据：

- `qasr completion bash|zsh|fish` 均 rc0；`bash -n` / `zsh -n` / `fish -n`
  全通过；脚本末尾恰好一个 `\n`。
- `--json` 在 stream 前后两个位置均 rc2，且输出合法 JSON 错误 envelope。
- `stream --input -` → rc2；`stream --input stdin`（缺 `--pcm`）→ rc2；
  `stream --input <目录>` → rc4；`completion bash extra` → rc2；
  `finetune`（缺子命令）→ rc2；`finetune lint-jsonl <不存在>` → rc4。

## 4. 验收 checklist

### 4.1 S 块

| 验收项 | 结果 | 证据 |
|---|---|---|
| 参数范围/默认值/10ms 倍数/16k/声道 | PASS | `test_p2_stream_usage.py`（67 例） |
| stdin 必须显式 `--pcm s16le` | PASS | rc2 + hint |
| stdin 专属参数用于 file/mic → 2 | PASS | `--pcm/--sample-rate/--channels` 三例 |
| `-` → 2；目录/glob/URL → 4 | PASS | 直接命令 + 用例 |
| `--json` × stream 无条件 2（任意位置/emit） | PASS | Q2；前后位置均验证 |
| 能力判定先于模型（transformers/remote/显式 vllm） | PASS | 均为 rc4；无 vLLM 的 Mac 亦为 4 |
| mic 能力预检先于模型加载 | PASS | `ac526bb`；fake 日志无 `model_init`；gpu06 WALL=0.17s、无 EngineCore 日志 |
| 缺失模型 → 3 | PASS | P0/P1 契约回归 |
| `--device cuda:N` → `CUDA_VISIBLE_DEVICES=N` | PASS | fake 模型记录 env |
| `--gpu-memory-utilization` 透传 | PASS | fake kwargs + gpu06 OOM 对比（0.9 OOM / 0.3 正常） |
| file/stdin/mic 三源与 EOF | PASS | PCM/源用例；mic 用 fake sounddevice |
| 残字节按 `channels×2` 帧丢弃 + warning | PASS | mono 奇字节、stereo 2/3 字节残帧 |
| 双声道按帧 `(L+R)/2` 下混 | PASS | 采样级断言 |
| realtime 只睡剩余预算、EOF 尾块不睡 | PASS | 处理耗时注入用例 + gpu06 elapsed 6.4s |
| 快照累计而非 delta、`seq` 连续 | PASS | 多 chunk/尾包/错误行用例 |
| 最终快照唯一且即使空尾包也发 | PASS | 精确 chunk 倍数用例 |
| `elapsed_s` 从命令入口计时（含模型加载） | PASS | `1f7788f`；fake 1s 加载：elapsed 1.011 / wall 1.11 |
| 非有限 `--max-stream-seconds` → 2 | PASS | `1f7788f`；nan/inf/1e309 均为 usage error |
| cap → rc5、单条 `ok:false,is_final:true,quality` | PASS | 3 例 + gpu06 3 行证据 |
| 中途失败补错误行；未输出过则普通 envelope | PASS | chunk 失败 + finish 失败（新增） |
| SIGINT：最终快照后 130，无半行 | PASS | fake SIGINT + gpu06 rc130/9 行 |
| 人类模式 TTY `\r`、非 TTY 只打最终文本 | PASS | PTY 用例 |
| stdout 纯净（vLLM banner/日志） | PASS | banner 注入用例 + gpu06 |
| 不写 state/jobs、不泄漏 token | PASS | 目录断言 + env token 用例 |

### 4.2 C 块

| 验收项 | 结果 | 证据 |
|---|---|---|
| 仅 bash/zsh/fish；缺参/非法 shell → 2 | PASS | 参数化用例 |
| 覆盖全部顶层命令与 `--help` 中 long option | PASS | 动态提取对照（Q3） |
| 末尾单个换行、不写文件 | PASS | 字节级断言 + `-n` |
| `--json`/`--jsonl` → 2 | PASS | 前后位置均验证 |
| 多余位置参数 → 2 | PASS | `completion bash extra` |
| 纯静态、无 torch/vllm/qwen_asr import | PASS | import blocker |

### 4.3 F 块

| 验收项 | 结果 | 证据 |
|---|---|---|
| `audio` 必填/非空/NUL/不存在/非普通文件 | PASS | 37 例 finetune 用例 |
| `text` 必填、语言前缀标签/方言/None、正文非空 | PASS | 前缀与 LID 表用例 |
| `prompt` 存在时必须是字符串 | PASS | 类型用例 |
| 未知字段 warning；strict 升级 error 且保留 warning | PASS | Q5 用例 |
| 重复 audio（resolve 后）warning；strict 升级 | PASS | 相对/绝对/符号链接等价路径 |
| 空行跳过且不计 total；非法 JSON 累计不 fail-fast | PASS | 行号/计数断言 |
| 有 error → 4；仅 warning → 0；strict+warning → 4 | PASS | 退出码矩阵 |
| `--no-check-files` 跳过存在性但仍校验类型 | PASS | 用例 |
| `--json` schema、人读 `path:line: field: message` + summary | PASS | jsonschema + 文本断言 |
| `path` 原样回显（Q12） | PASS | 相对路径用例 |

### 4.4 Schema / 打包 / 文档

| 验收项 | 结果 | 证据 |
|---|---|---|
| `schemas/stream.json` oneOf 成功/错误、错误行六字段 | PASS | 每行快照 jsonschema |
| `schemas/finetune-lint.json` issue 三字段必备 | PASS | jsonschema |
| `pyproject.toml` 有 `mic` 与 `all`（含 local/mic/hf/modelscope） | PASS | 打包用例 |
| README 覆盖 stream/completion/finetune 与安装矩阵 | PASS | 打包用例 |
| `--help`/completion/finetune/stream 参数错误不 import 重运行时 | PASS | import blocker |

## 5. 问题清单

### BLOCKER

无。

### MAJOR

无。

### MINOR（复审结果：全部 fixed）

| 编号 | 问题 | 状态 | 修复 |
|---|---|---|---|
| MINOR-1 | 非有限 `--max-stream-seconds` → rc3 “internal error” | **fixed** | `1f7788f` |
| MINOR-2 | mic 能力预检晚于模型加载 | **fixed** | `ac526bb` |
| MINOR-3 | `elapsed_s` 未包含模型加载 | **fixed** | `1f7788f` |

#### MINOR-1：非有限 cap（已修复）

- 修复：`qasr_cli/commands/stream.py:120` 增加
  `math.isfinite(max_seconds)` 校验。
- 复验（`1f7788f`）：
  ```
  python -m qasr_cli stream --input stdin --pcm s16le \
    --max-stream-seconds nan </dev/null
  # rc=2
  # qasr: usage error: --max-stream-seconds must be a finite positive number
  ```
  `inf` / `1e309` 同样 rc2；原 strict xfail 已改为正向断言并通过。

#### MINOR-2：mic 预检顺序（已修复）

- 修复：`ac526bb` 在模型解析/加载前调用
  `qasr_cli/commands/stream.py:199 core_stream.require_mic()`
  （实现见 `qasr_cli/stream.py:139`）。
- 复验：`test_missing_sounddevice_fails_before_model_load` 通过，fake 日志无
  `model_init`；gpu06 `stream --input mic` → rc4，WALL=0.17s，stderr 无
  EngineCore/architecture 日志。

#### MINOR-3：`elapsed_s` 起点（已修复）

- 修复：`1f7788f` 在 `handle()` 入口取 `started_at`
  （`qasr_cli/commands/stream.py:130`），传入
  `qasr_cli/stream.py:294` 的 `run_stream`。
- 复验：fake 注入 1.0s 模型加载后，`elapsed_s=1.011`、进程 wall=1.11s；
  gpu06 file 200s 最终 `elapsed_s=75.17s`、首条快照 46.61s，已包含加载。

### 文档残留（非缺陷计数）

- `P2-TASKS.md:79` / `:120` 仍写“`--json` 可与 `--emit jsonl` 同用”，
  与 `P2-TASKS.md:249`（Q2 无条件互斥）矛盾。
- 实现与测试均以 §7 Q2 为准；属任务书残留文案，记作 **wontfix（本轮）**，
  不影响验收。

## 6. gpu06 证据复核

以下为协调者在 gpu06（Qwen3-ASR-1.7B，CUDA0，util 0.3）的记录，审查侧只
做一致性核对，未独立连接服务器：

| 证据 | 复核结果 |
|---|---|
| file 200s：101 行，seq 0..100 连续，唯一 final，audio_s=200.0，elapsed 75.17s、首条 46.61s，text 2703 字符 | 符合 S3；elapsed 已含模型加载；与 stdin text 完全一致 |
| stdin 200s：101 行，seq/final/audio_s 同 file | 符合 S2/S3 |
| cap=5s：恰好 3 行（2.0/4.0 ok:true + ok:false quality final） | 符合 Q1：无 `ok:true` 最终行 |
| SIGINT：9 行 seq0..8，唯一 final 且为最后一行，rc130 | 符合 Q15 |
| 显式 transformers / server → rc4 | 符合 Q14 |
| mic 无 sounddevice → rc4，WALL=0.17s，无 EngineCore 日志 | 符合 Q14；MINOR-2 已修复 |
| 缺失模型 → rc3；文件不存在 → rc4 | 符合 S1 |
| 中文 4.2s realtime：5 行，audio_s 单调，language=Chinese，elapsed 6.4s | 符合 Q9/Q10 |
| `--gpu-memory-utilization` 0.3 正常、缺省 0.9 OOM | 透传生效 |
| stdout 无 vLLM 日志（日志在 stderr） | 符合 S4 |

`simulated` 用例不作为上述 gpu06 结论的依据。

早期 gpu06 的 `elapsed_s=28.99s` 为修复前语义（不含加载）；已被上述
`75.17s` / 首条 `46.61s` 取代。

## 7. 测试口径与设计歧义

- 已按 §7 执行，不把下列定稿语义当 bug：多输入/`--json` 批处理、格式互斥、
  空音频、remote 切块、`language_source=server`、fail-fast 退出码等。
- `--json` × stream：以 §7 Q2 为准（无条件 2）；§2 残留文案见 §5 文档残留。
- `elapsed_s`：已按任务书字面实现（含模型加载），不再存在歧义。
- 嵌套子命令（`jobs list --limit`、`finetune lint-jsonl --strict`）不在
  completion 契约内：P2-TASKS §3 只要求“全部顶层子命令与每个子命令的常用
  flag”，当前脚本满足；未按缺陷计。

## 8. 最终判定

**PASS。**

- **0 BLOCKER / 0 MAJOR / 0 MINOR**（3 项 MINOR 已修复并复验）。
- 基线 `21f111a`（stream 最终代码 `1f7788f`）：P2 套件 187 passed /
  6 deselected / 0 xfailed；仓库 420 passed / 6 skipped；P0 164 passed /
  1 skipped；P1 204 passed；ruff 通过。
- gpu06 证据覆盖 file/stdin/cap/SIGINT/mic 预检/能力/中文 realtime，均符合
  §7 裁决；`elapsed_s` 新语义已由 75.17s / 46.61s 证据确认。
- 唯一残留为 `P2-TASKS.md` §2 的陈旧文案（wontfix，本轮不改），不影响实现、
  测试或验收。
