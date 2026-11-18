# P1 独立审查报告（N1–N5，终审）

审查者：独立 Codex reviewer（只审不改实现）
最终基线：`6a0f524`（干净 `git archive` 快照 `/tmp/qasr-6a0f524`）
历史基线：`4f30b14` / `50ee423` / `cf1c9a4`
日期：2026-11-18
依据：`P1-TASKS.md`（含 a53859c / fcca068 / 50ee423 定稿）、`DESIGN.md` v0.3、
协调者 gpu06 实测口径

---

## 1. 结论

**PASS**

- **BLOCKER：0**
- **MAJOR：0**
- **MINOR：0**

`cf1c9a4` 的 5 项修复（1 MAJOR + 4 MINOR）全部复验通过；其引入的 1 个 P0
回归（活跃锁检查晚于模型解析）已由 `6a0f524` 修复并复验。`6a0f524` 干净快照上
`review_tests_p1`、P0 回归、仓库 tests、ruff 与本地 0.6B 集成全部通过，
4 个原 strict xfail 已全部转为正向断言并 XPASS，当前 0 xfail / 0 failed。

GPU/vLLM/1h 证据仍以协调者 gpu06 实测为准；本轮独立复跑限于 Mac/CPU、假服务与
本地 0.6B。该限制不构成本次 P1 代码验收的阻塞项。

---

## 2. 审查方法与最终结果

### 2.1 方法

- 黑盒：`review_tests_p1/` 只通过 `subprocess` 调 `python -m qasr_cli`，不 import
  实现内部；假 OpenAI 服务、假模型目录、合成 WAV 均为 stdlib fixture。
- P0 回归：`review_tests/` 从 tag `v0.1.0` 独立维护；P1 契约变化只改测试断言，
  不改实现。
- 干净快照：所有终审证据来自 `git archive 6a0f524` 的只读导出，避免并行工作树
  竞态。
- 无 torch：`/tmp/qasr-m1-venv` 不安装 torch；远程、dry-run、stdout 纯净与
  import-blocker 用例均在该解释器运行。
- 集成：本地 0.6B + aligner，真实模型由 `RUN_INTEGRATION=1` 门控。

### 2.2 最终结果（HEAD `6a0f524`）

| 套件 | 命令 | 结果 |
|---|---|---|
| P1 黑盒（非集成/非 GPU） | `pytest -q review_tests_p1 -m 'not integration and not gpu'` | **204 passed / 0 failed / 0 xfailed** |
| N5 子集 | `pytest -q review_tests_p1 -m 'n5 and not integration and not gpu'` | **69 passed / 0 failed** |
| P0 回归 | `pytest -q review_tests -m 'not integration and not gpu'` | **164 passed / 1 skipped** |
| 仓库 tests | `pytest -q tests` | **357 passed / 6 skipped** |
| ruff | `ruff check qasr_cli tests` | **pass** |
| P1 integration（本地 0.6B/aligner） | `RUN_INTEGRATION=1 ... -m integration review_tests_p1` | **4 passed / 2 skipped** |

4 个原 strict xfail 已全部移除 marker 并转为正向断言；当前 **0 strict xfail、
0 XPASS、0 failed**。

---

## 3. 终审关闭的 5 项 + 1 个回归

| 项 | 严重度 | 修复提交 | 终审证据 |
|---|---|---|---|
| exactly-180s align 被拆成两窗 | MAJOR-1 | `2e1c023` | 180.0s → 单窗 0–180；180.5s → 两窗 0–177 / 177–180.5 |
| align `--dry-run` 尝试 import torch | MINOR-1 | `7ebc369` | torch/qwen_asr/vllm import blocker 下 rc=0 |
| `dry-run.json` 缺 `plan.windows` | MINOR-2 | `6315c5c` | schema 显式声明 5 个窗口字段；jsonschema 校验通过 |
| `models rm` 接受校内 `..` 路径 | MINOR-3 | `94fbfb2` | `../models/<managed>` → rc=2，目录保留；正常 alias 仍 rc=0 |
| usage error 后残留 jobs 索引 | MINOR-4 | `cf1c9a4` | `--json ... -f json` → rc=2，`QASR_HOME/jobs` 为空 |
| cf1c9a4 引入的活跃锁回归 | 回归 | `6a0f524` | 活跃锁存在时模型解析前返回 rc=2 |

### 3.1 MAJOR-1：exactly-180s 单窗（fixed `2e1c023`）

- 修复：`plan_windows()` 在 `total_samples <= window_samples` 时直接返回单窗，
  不再先减 overlap。
- 复验：

  ```bash
  # 180.0s 音频 + 参考文本
  python -m qasr_cli --json --dry-run align long180.wav \
    --text-file ref.txt --language en
  # windows = 1，core 0–180
  ```

- 用例：`review_tests_p1/test_n4_long_audio.py::test_align_exactly_180s_is_single_window`
  终审 PASS（原 strict xfail 已移除）。
- 180.5s 边界由 P0 `test_align_180_second_boundary[180.5]` 覆盖并通过，窗口 core
  无缝衔接。

### 3.2 MINOR-1：align `--dry-run` 不 import torch（fixed `7ebc369`）

- 修复：`resolve_device_dtype(optional=True)` 且 device/dtype 均为 `auto` 时直接
  返回 `("auto", "fp32")`。
- 复验：用 sitecustomize `meta_path` blocker 阻断 `torch`/`vllm`/`qwen_asr` 后，
  `align --dry-run --json` rc=0，blocker 标记未生成。
- 用例：`review_tests_p1/test_n4_long_audio.py::test_align_dry_run_does_not_import_runtime`
  终审 PASS。

### 3.3 MINOR-2：`dry-run.json` 声明窗口计划（fixed `6315c5c`）

- 修复：`plan.properties.windows` 显式声明
  `index/audio_start_s/audio_end_s/core_start_s/core_end_s`。
- 复验：align dry-run 输出通过 `schemas/dry-run.json` jsonschema 校验。
- 用例：`review_tests_p1/test_schemas_p1.py::test_dry_run_schema_allows_window_plan`
  终审 PASS。

### 3.4 MINOR-3：`models rm` 拒绝 `..`（fixed `94fbfb2`）

- 修复：`remove_model()` 入口检查 `Path(value).parts` 含 `..` → `UsageError(2)`，
  即便 resolve 后仍在 `models_root` 内。
- 复验：

  ```bash
  (cd "$QASR_HOME/models" && python -m qasr_cli models rm \
     ../models/Qwen3-ASR-0.6B --yes --dry-run --json)   # rc=2，目录仍在
  ```

- 用例：`test_models_rm_rejects_dotdot_even_when_inside_models_root` 终审 PASS；
  正常 alias/repo/相对/绝对 dry-run 仍 rc=0。
- 同步把 P1 套件中 `../outside` 的期望从 3 更新为 2（新契约下 `..` 统一先判
  usage，而不是先报 not found）。

### 3.5 MINOR-4：usage error 不登记 job（fixed `cf1c9a4`）

- 修复：`register_job` 移到全部参数/格式/互斥/out-dir 校验之后、`core.run` 之前；
  `ensure_output_unlocked` 保持在模型解析前。
- 复验：`qasr --json transcribe a.wav -o out -f json` → rc=2，
  `QASR_HOME/jobs` 为空。
- 用例：`review_tests_p1/test_n5_batch_ops.py::test_usage_error_does_not_register_job`
  终审 PASS（本轮新增）。

### 3.6 回归：活跃锁必须早于模型解析（fixed `6a0f524`）

- `cf1c9a4` 把 `ensure_output_unlocked` 一并后移，导致活跃锁场景先报模型 rc=3。
- `6a0f524` 将锁检查放回 `--retry/--fail-fast` 校验后、`_normalize_formats` 前，
  保留 `not dry_run` 守卫；`register_job` 仍在末尾。
- 复验：`review_tests/test_m4_manifest_resume.py::test_active_lock_blocks_second_run`
  终审 PASS（active lock → rc=2）。

---

## 4. 已修复项复核（N1–N5）

### 4.1 N1（serve / doctor --server / token / redirect）

复核结论：**fixed**。修复提交 `3eb6798`。

| 项 | 复验 |
|---|---|
| userinfo 剥离与 URL 掩码 | `test_doctor_server_userinfo_is_redacted`、`test_config_show_redacts_server_userinfo` 通过 |
| 非法端口不崩溃 | `test_doctor_invalid_port_with_userinfo_is_classified_not_crashed` 通过 |
| `http_status` / `auth_rejected` | 401/500/超时分类用例通过 |
| serve gpu 范围、aligner 拒绝、未知 flag 后 `MODEL` | `test_n1_serve_doctor.py` 通过 |
| `schemas/serve.json`、`config show` 脱敏 | schema 校验与 token 扫描通过 |
| R10 跨 origin（scheme/host/port 任一变化）不转发 Authorization | 3 跳/跨端口/跨 scheme 用例通过 |

### 4.2 N2（remote 传输、错误映射、并发）

复核结论：**fixed**。修复提交 `fb1b679`、`064cf0a`。

| 项 | 复验 |
|---|---|
| **BLOCKER-1 5xx body token 脱敏** | strict xfail 已移除；stdout/stderr/state 均无 token |
| remote 不 import torch | torch-free + import blocker 用例通过 |
| warning 写 stderr；cached chat-only 重复 warning | `test_remote_warnings_are_echoed_to_stderr` 通过 |
| 未显式 `--model` 时取 `/v1/models` 首个 | `test_default_model_is_first_models_entry` 通过 |
| 显式 `--model` 才匹配；404 不误报 | `test_served_model_resolution` 系列通过 |
| 408/429/5xx/超时→3，普通 4xx→4，retries=总 3 次 | 错误矩阵用例通过 |
| 语言 400 只额外重试一次 | `test_language_rejection_retries_once` 通过 |
| fail-fast 退出码=首个错误码（3/4/5），非 fail-fast 部分失败=6 | 通过 |
| `--jobs` 1–32、默认 4、结果/首错按输入顺序、state 串行、无 running | 并发/顺序/state 用例通过 |
| `language_source="server"` schema 兼容 | `transcribe-item.json` 通过 |
| >25MB 改为客户端 270s 切块，每块 <25MB | `test_upload_over_25mb_is_chunked_client_side` 通过 |
| chunk resume 只重跑未完成窗口 | `test_remote_chunk_resume_only_retries_unfinished` 通过 |

### 4.3 N3（vLLM / auto / stdout 纯净）

复核结论：**fixed（代码与 Mac 黑盒）；GPU 证据依赖 gpu06**。

- Mac 无 vllm：`--backend auto` 仍 transformers；显式 `--backend vllm` rc=3，
  未出现 import 崩溃。
- `--gpu-memory-utilization` 与非 vllm 同用 rc=2；范围 (0,1] 校验通过。
- `CUDA_VISIBLE_DEVICES` 在 `resolve_backend()` 探测前写入；`cuda:N` 重写为内部
  `cuda:0`。
- `VLLM_CONFIGURE_LOGGING=0`、handler 重定向、atexit fd 守卫均在代码中；协调者
  gpu06 复验 `--json` stdout 恰一行、日志在 stderr。
- 本轮未在 gpu06 复跑；不把协调者数据冒充独立证据。

### 4.4 N4（长音频）

复核结论：**fixed**。修复提交 `d3d51d0`、`8c7fee3`、`2e1c023`、`6315c5c`、
`7ebc369`。

| 项 | 状态 |
|---|---|
| `plan_windows` 公式/采样点/末尾短窗/overlap=0/参数校验 | pass |
| `attribute_reference` 单调归属、候选区间匹配 | pass |
| `merge_items` `start=max(start,prev.end)`、`end<=start` clamp、空 item | pass |
| transcribe 270s 默认、timestamps 180s、>180→4、10–1200 校验 | pass |
| remote 270s 默认、600s 上限 warning、逐窗临时 16k mono WAV | pass |
| chunk checkpoint key 含 SHA/窗口/模型/aligner/language/context/版本 | pass |
| `--overwrite` 只清当前 job_key、损坏 chunk 重跑、kill -9 resume | pass |
| 181s align 窗口化、core 无缝、末窗 core_end=duration | pass |
| exactly-180s 单窗 | pass（`2e1c023`） |
| align `--language auto` 脚本判定 / LID probe 代码路径 / `language_source` schema | pass（黑盒验证脚本门禁；LID 结果路径由真实/假 backend 覆盖） |

### 4.5 N5（批量、运维、schema）

复核结论：**fixed**。修复提交 `e47e563`、`fc800c4`、`771b3a7`、`10b8ebc`、
`2c9f23e`、`94fbfb2`、`cf1c9a4`。

| 原缺陷 | 复验证据 |
|---|---|
| `--skip-existing` 先要求模型 | 缺失模型 + 完整输出 → rc=0、`skipped=true`、未出现 `model loaded` |
| `-f tsv/md` + `--json/--jsonl` 被接受 | 现在 rc=2，message 明确互斥 |
| `--stdin-list` 的 `-` 当路径 | 现在 rc=2 `'-' is not an input path` |
| `schemas/config-get.json` 缺失 | 已新增；`config-get.json`、`config-set.json` 均在 inventory |
| 未知配置键导致所有命令 rc=2 | `{"custom_key":"keep"}` + `config set window_seconds 120` → rc=0 且保留 `custom_key` |
| bench 0 值静默用默认 | `--duration 0` / `--iterations 0` → rc=2 |
| remote bench warning 只在 JSON | 现在同时写 stderr |
| `config set --project` 只检查目标文件 symlink | `.qasr` 父目录 symlink → rc=2，未写到外部 |
| `models rm` 校内 `..` | `94fbfb2` 后 rc=2，目录保留 |
| usage error 残留 jobs 索引 | `cf1c9a4` 后 rc=2 且索引为空 |

---

## 5. P0 套件改动判定

Claude 在 P1 期间修改了 3 个 P0 黑盒文件：

- `review_tests/test_m1_cli_surface.py`：`serve` 从“P1 不可用”改为可用。
- `review_tests/test_m1_doctor.py`：`doctor --server` 从 rc=2 改为 rc=0 + server check。
- `review_tests/test_m3_timestamps_align.py`：`align --server` 从“无此 flag”改为
  capability rc=4。

**判定：合理契约更新，不是弱化断言。** 三项均对应 P1-TASKS T1.1/T1.2/T2.1 的
明确交付；没有发现无关断言的删除或放宽。

本轮按 P1 语义更新了 7 条 P0 回归用例（全部通过）：

1. `test_p1_transcribe_flags_are_not_available[--retry]` → 改为“flag 可解析、语义由
   P1 套件覆盖”。
2. 同上 `[--stdin-list]`。
3. `test_align_rejects_audio_over_180s` → 改为 181s `--dry-run` 成功、窗口 ≥2、
   末窗 `core_end≈181`。
4. `test_align_180_second_boundary[180.5]` → 从 rc=4 改为 rc=3（缺模型；证明不再
   被 180s 能力门禁拦截）。
5. `test_align_auto_rejects_latin_script` → 从 rc=4 改为 rc=3（证明 Latin 不再在
   脚本门禁被拒绝；LID probe 结果路径由实现契约测试覆盖）。
6. `test_invalid_format_combinations[-f tsv]` → 移出非法列表；新增 `-f tsv`
   dry-run 接受用例。
7. `test_p1_format_in_manifest_is_rejected` → 改为 manifest `formats:["tsv"]`
   接受，并校验 plan/outputs 扩展名。

---

## 6. xfail 处置

| xfail | 终审处置 |
|---|---|
| N1 2 项、N2 BLOCKER/MINOR、N5 5 MAJOR + 3 MINOR | 已转正向断言并移除 marker |
| `test_align_exactly_180s_is_single_window` | `2e1c023` 后移除 marker，终审 PASS |
| `test_align_dry_run_does_not_import_runtime` | `7ebc369` 后移除 marker，终审 PASS |
| `test_dry_run_schema_allows_window_plan` | `6315c5c` 后移除 marker，终审 PASS |
| `test_models_rm_rejects_dotdot_even_when_inside_models_root` | `94fbfb2` 后移除 marker，终审 PASS |

最终：**0 xfail、0 XPASS**。

---

## 7. P1 验收 checklist

| 任务 | 判定 | 说明 |
|---|---|---|
| T1.1 serve 薄包装 | PASS | argv/dry-run/缺 binary→3/execvp/未知参数透传 |
| T1.2 远程配置与 doctor --server | PASS | 优先级/schema/401/超时/脱敏/重定向安全 |
| T2.1 传输契约 | PASS | multipart/chat/无 prefill/前缀解析/服务端 LID/404 降级/客户端切块 |
| T2.2 超时重试错误映射 | PASS | 3 vs 4、总 3 次、语言 400 一次、Retry-After、部分失败 6 |
| T2.3 远程并发 | PASS | jobs 1–32、默认 4、顺序、state 串行、fail-fast 无 running |
| T3.1 VLLMBackend | PASS* | 代码/Mac 黑盒通过；GPU 证据由 gpu06 提供 |
| T3.2 auto 选择 | PASS* | Mac auto→transformers；显式 vllm 不回退；GPU 证据外置 |
| T4.1 long_audio 纯函数 | PASS | 窗口/归属/合并/覆盖率单测 |
| T4.2 transcribe 切块 | PASS | 180/270、chunk key、resume/overwrite、remote 270/600 |
| T4.3 align 窗口化 | PASS | 181s+、参数、单调合并、exactly-180s 单窗均通过 |
| T4.4 align --language auto | PASS* | 脚本路径通过；LID probe 由真实/假 backend 覆盖，gpu06 有证据 |
| T5.1 输入与重跑 | PASS | stdin-list/skip-existing/retry 语义通过 |
| T5.2 tsv/md | PASS | 表头/转义/language 列/无条件互斥 |
| T5.3 jobs list/show | PASS | 索引/排序/limit/未知 job→4/`^[0-9a-f]{16}$`/OUT_DIR resolve |
| T5.4 bench | PASS | plan/字段/remote warning/token 不泄漏/0 值 rc=2 |
| T5.5 models show/rm | PASS | alias/repo/相对/绝对 dry-run；symlink→2；`..`→2；非托管拒绝 |
| T5.6 config get/set | PASS | --project 根/symlink/未知键保留/枚举/脱敏 |
| T5.7 文档与 schema | PASS | 新 schema 齐全；README 命令可复制；`--help` 不 import torch/vllm |

\* GPU/vLLM 与真实 LID probe 由协调者 gpu06 实测；本轮独立证据限于代码审查、
Mac/CPU 黑盒、假服务与本地 0.6B。

---

## 8. 风险与限制

1. **GPU 未独立复跑**。N3/vLLM、1h 时间戳、1h align、真实 LID probe 的证据来自
   协调者；本报告只核对代码路径与 Mac 行为。最终验收应保留 gpu06 原始输出。
2. **coverage 语义**：真实 aligner 上 coverage 常为 1.0，符合 DESIGN 的文本匹配
   定义；不按“时间覆盖率”另设门禁。
3. **测试口径同步**：P1 契约变化已同步到 P0 回归（7 条）与 P1 套件
   （`../outside` 期望 2、MINOR-4 新用例）；当前无已知未同步断言。
4. **代码/测试边界**：本轮只修改 `P1-REVIEW-*` 与 `review_tests*`，未修改
   `qasr_cli/`、`schemas/`、`tests/`、`pyproject.toml`、`DESIGN.md`、任务书。

---

## 9. 最终判定

**PASS**

`6a0f524` 上 BLOCKER 0 / MAJOR 0 / MINOR 0；P1 各任务验收项全部通过，
4 个原 strict xfail 已转正向并 XPASS。建议以 `6a0f524` 作为 P1 最终验收基线。
