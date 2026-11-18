# qasr-cli P1 独立审查与测试计划

> 审查者：Codex Partner（独立于实现）
> 日期：2026-11-18
> 审查基线：`HEAD=68db738`（P1 任务书提交；P0 已 PASS，tag `v0.1.0`）
> 依据：`P1-TASKS.md`、`DESIGN.md` v0.3 §3.7–3.9/§5/§7/§8/§12.3、`REVIEW-REPORT.md`（P0）
> 当前状态：P1 实现尚未开始；本文件只定义审查与测试，不修改实现。

### 0.1 2026-11-18 事实更新（以 gpu06 实测与 P1-TASKS a53859c 为准）

以下事实已覆盖本文件早期基于 DESIGN/T2.1 初稿的假设，测试按此执行：

1. `/v1/audio/transcriptions` 可用；响应 `text` 带 `language Chinese<asr_text>` 前缀，**两个端点都必须解析前缀**。纯文本无前缀时语言为 `null`。
2. transcription 的 `language` 字段只做服务端校验；合法值不改变 Qwen3-ASR 的 LID 输出。显式语言与输出不一致 → warning（`server_ignored_language`），以服务端语言为准，不得伪造。
3. vLLM 0.14.0 的 chat 模板只渲染 system + 音频占位，忽略 user 文本与 assistant 消息；**assistant prefill / `continue_final_message` 会 400**。实现不得依赖 prefill；chat 无法强制语言，显式语言只 warning。
4. `--retries 2` = 总 3 次尝试；408/429/5xx/超时/连接失败 → 3；普通 4xx → 4；语言拒绝只额外重试一次；批量首错按输入顺序。
5. `align` P1 起默认窗口化：181s 成功，≤180s 走 P0 单窗；`--window-overlap 0` 合法；窗口公式与 merge 规则已定稿。
6. vLLM transcription 默认单文件上限 25MB；**远程长音频必须客户端切块**，默认 270s、上限 600s（超出降级 + warning）。
7. 远程 `--batch-size > 1` 只 warning 并忽略；传输并发只由 `--jobs` 控制。
8. `--stdin-list` **无参数**，从 stdin 读路径（文件用 `< paths.txt`）；与 `--manifest` 互斥，可与位置参数并存，空行跳过，`-` 不是路径。
9. `tsv`/`md` 与 `--json`/`--jsonl` **无条件互斥**（退出 2）；单条无 out-dir 时 tsv/md 可写 stdout。
10. `jobs list --limit` 范围 1–10000，0/负数/超大 → 2；`jobs show` 只认索引，OUT_DIR 必须是索引登记目录。
11. 302 的跨 origin 判定包含 scheme/host/port；同 host 不同 port 也不得转发 Authorization。

---

## 0. 判定口径

| 判定 | 含义 |
|---|---|
| **PASS** | P1 验收项全部有独立黑盒证据；无 BLOCKER/MAJOR；MINOR 仅记录不阻塞 |
| **PASS WITH CHANGES** | 主链路可用，但存在必须在 P1 内修复的 MAJOR，或关键证据只能由 GPU 环境补足 |
| **FAIL** | 存在 BLOCKER，或 N1–N5 任一验收项无法完成，或安全/数据完整性问题 |

严重度：

- **BLOCKER**：token 泄漏、路径逃逸/误删、远程契约不可用、长音频静默丢字/重复、state 永久不一致、显式 backend 静默降级。
- **MAJOR**：验收项不满足、错误码/重试/并发语义错误、resume 复用错误、schema 不兼容、无明显 workaround。
- **MINOR**：帮助文本、warning 文案、非关键字段、边界 UX；有明确 workaround。

审查原则：

1. 以**黑盒 subprocess**为主；不 import `qasr_cli` 内部模块。
2. 假 OpenAI 服务、合成 WAV、PATH 假 binary 均可离线运行；无 GPU 环境必须能跑完非集成套件。
3. 真实模型、vLLM、1h 音频、gpu06 证据只放在 `integration` / `gpu` 标记；由协调者安排后复核证据。
4. 发现实现与任务书冲突时，先区分“实现 bug”与“任务书/设计歧义”；后者记录为风险并在报告中给出建议，不直接判实现 FAIL。
5. 不修改 `qasr_cli/`、`tests/`、`schemas/`、`pyproject.toml`、`DESIGN.md`、`P1-TASKS.md`。

---

## 1. 审查范围

### 1.1 范围内

| 波次 | 范围 |
|---|---|
| N1 | `serve` 薄包装、argv 透传、`--dry-run`、缺 binary、`doctor --server`、远程配置与 token 边界 |
| N2 | `/v1/audio/transcriptions`、`/v1/chat/completions`、context、语言/prefill 降级、404 降级、错误映射、超时/重试、远程 `--jobs`、远程 state/resume |
| N3 | `--backend vllm`、`--backend auto` 选择与回退、`gpu_memory_utilization`、vLLM 批量/失败隔离/timestamps 路径 |
| N4 | 180s/270s 窗口口径、`plan_windows`、`attribute_reference`、`merge_items`、`transcribe`/`align` 窗口化、chunk checkpoint/resume、`align --language auto` |
| N5 | `--stdin-list`、`--skip-existing`、`--retry`、tsv/md、`jobs list/show`、`bench`、`models show/rm`、`config get/set`、README/schema |

### 1.2 非范围

- P2：`stream`、`--mic`、`completion`、`finetune lint-jsonl`、多 GPU、视频抽轨、watch、Windows。
- 训练/微调、说话人分离、翻译、原生异步 server。
- 性能门禁；GPU RTF 只记录，不作为 PASS/FAIL 条件。
- 替代协调者操作 gpu06；审查者只复核证据，不从 Mac 向 gpu06 传文件。

### 1.3 明确不判 FAIL 的既有语义

- 远程不返回时间戳/对齐：这是 P1 能力边界，必须退出 4，不是缺陷。
- 真实 aligner coverage 通常为 1.0：低覆盖门禁用 fake aligner 验证，不要求真实模型制造低覆盖。
- 远程 `language` 为 `null`、`language_source="server"`：任务书已定稿，除非 schema 不兼容。
- 多输入 + `--json`/`--jsonl` 是合法 batch envelope；`--format json|jsonl` 与 `--json|--jsonl` 无条件互斥（P0 定稿）。

---

## 2. 审查环境与方法

### 2.1 无 GPU 环境

| 项 | 值 |
|---|---|
| 主解释器 | `/tmp/qasr-m1-venv/bin/python`（torch-free；已装 pytest/jsonschema/ruff） |
| 运行时集成 | `/Users/walter/workspace/QwenLM/qwen-tts-cli/.venv/lib/python3.12/site-packages:.` |
| 模型 | `/Users/walter/workspace/QwenLM/Qwen3-ASR/models/{Qwen3-ASR-0.6B,Qwen3-ASR-1.7B,Qwen3-ForcedAligner-0.6B}` |
| 音频 | 合成 WAV（1s/25s/181s）；集成用 `/tmp/qasr_asr_zh.wav` |
| 网络 | 默认离线；假 OpenAI 服务只监听 `127.0.0.1` 随机端口 |

### 2.2 GPU 环境

- 由协调者在 gpu06 安排；审查者不自行 ssh、不传文件。
- 需要证据：命令、环境（GPU 编号、CUDA_VISIBLE_DEVICES、版本）、原始输出、失败日志、1h 时间戳/align 的 JSON/SRT 摘要。
- gpu06 经跳板机，非局域网；任何 Mac→gpu06 文件传输必须先经用户确认。

### 2.3 测试套件

`review_tests_p1/` 是独立黑盒套件，默认不进入仓库 `pytest` 的 `testpaths`。运行方式：

```bash
# 非集成（无 GPU、无模型；假服务 + 合成音频）
/tmp/qasr-m1-venv/bin/python -m pytest review_tests_p1 -q -m "not integration"

# 真实 0.6B 集成（不要求 GPU）
RUN_INTEGRATION=1 \
QASR_TEST_MODEL_DIR=/Users/walter/workspace/QwenLM/Qwen3-ASR/models \
QASR_TEST_AUDIO=/tmp/qasr_asr_zh.wav \
QASR_PYTHON=/Users/walter/workspace/QwenLM/qwen-tts-cli/.venv/bin/python \
/Users/walter/workspace/QwenLM/qwen-tts-cli/.venv/lib/python3.12/site-packages:. \
/tmp/qasr-m1-venv/bin/python -m pytest review_tests_p1 -q -m integration

# GPU 证据复核（在 gpu06 侧执行，不由 Mac 传文件）
pytest review_tests_p1 -q -m gpu
```

标记：`p1`（默认全部）、`integration`（需真实模型）、`gpu`（需 CUDA/vLLM）、`slow`（长音频）。

套件文件（截至 Phase 1）：

| 文件 | 覆盖 |
|---|---|
| `helpers.py` | JSON/JSONL、错误断言、合成 WAV、合成模型、schema、secret 扫描 |
| `fake_openai.py` | 线程化假 OpenAI/vLLM：multipart/chat/auth/延迟/错误/并发统计 |
| `conftest.py` | subprocess runner、隔离 `QASR_HOME`、假 binary、import blocker |
| `test_n1_serve_doctor.py` | serve 透传/exec/缺 binary/dry-run；doctor 200/401/500/超时 |
| `test_n2_remote_transport.py` | multipart、language、chat/context、无 prefill、404 降级、远程切块前置 |
| `test_n2_remote_errors.py` | 4xx/408/429/5xx/超时/连接失败、重试次数、首错顺序、部分失败 |
| `test_n2_remote_concurrency.py` | `--jobs` 并发上界、顺序、fail-fast state、state JSONL 完整性 |
| `test_n3_vllm_auto.py` | 显式 vllm 平台门禁、auto 选择、gpu-memory-utilization、doctor vllm |
| `test_n4_long_audio.py` | 窗口公式/边界/overlap=0、180/270 口径、远程切块、dry-run 不 import |
| `test_n5_batch_ops.py` | stdin-list、skip-existing、retry、tsv/md、部分失败 |
| `test_n5_jobs_bench.py` | jobs 索引/list/show、bench dry-run/remote 指标 |
| `test_n5_models_config.py` | models show/rm 路径安全、config get/set/project/symlink |
| `test_security_token.py` | token 输出/磁盘/重定向/userinfo 泄漏面 |
| `test_schemas_p1.py` | 新 schema 清单、`language_source` 兼容 |
| `test_integration_p1.py` | 真实 0.6B 短音频/时间戳/align（需 `RUN_INTEGRATION=1`） |

---

## 3. 设计/任务书风险清单（实现前预判）

以下风险不是对未完成实现的指控，而是审查时必须优先验证的契约点。标 **B** 的是“若成立即 BLOCKER”。

### 3.1 远程契约（transcription vs chat）

| ID | 级别 | 风险 | 验证方式 |
|---|---|---|---|
| R1 | ~~B~~ 已核实 | gpu06 实测 `/v1/audio/transcriptions` 可用；multipart、`model`、`language` 契约按 T2.1（2026-11-18 修订）执行。 | 假服务精确断言字段；已加入 N2 transport 用例。 |
| R2 | MAJOR | chat 端点是否要求 `model` 仍未在任务书写死；真实 vLLM 可能要求。 | 假服务记录 chat body；gpu06 复核 400 原文。 |
| R3 | ~~MAJOR~~ 已关闭 | prefill 已实测必 400；T2.1 已禁止 prefill，改为 warning。 | 用例断言 chat 只发一次、无 assistant 消息。 |
| R4 | MAJOR | `Cantonese`/`yue` 不是 ISO-639-1；`ISO639_1_SUPPORTED_LANGS` 可能不含 `yue`，transcription 端点会 400；任务书的“语言被拒→去掉 language 重试”会丢失强制语种。 | 假服务注入 400；gpu06 用 `--language yue`，确认是否走 chat/prefill 或给出明确错误。 |
| R5 | MAJOR | 语言拒绝、prefill 拒绝、transcription 404→chat 三种降级可能叠加；任务书未定义优先级、重试次数、最终错误和 warning 顺序。 | 假服务组合注入：400 language + 400 prefill + 404；断言请求次数和最终结果。 |
| R6 | MAJOR | 本地 `parse_asr_output` 可能与 qwen-asr 不一致（`language None`、`language Chinese,English`、无 tag、`<|endoftext|>`、前后空白），且**两个端点**都要解析。 | 假服务返回带/不带前缀、多语言标签、空文本；对照 `qwen_asr.inference.utils.parse_asr_output` golden 用例。 |
| R7 | MAJOR | 远程默认模型取 `/v1/models` 第一个；多模型时顺序不稳定，job_key/resume 可能漂移。 | 假服务返回多模型不同顺序；比较两次 job_key。 |
| R8 | MINOR | transcription 响应若带 `language` 字段被强制忽略为 null；用户可能误以为服务端未识别。 | 假服务返回 language；确认文档/warning。 |

### 3.2 `QASR_SERVE_TOKEN` 泄漏面

| ID | 级别 | 风险 | 验证方式 |
|---|---|---|---|
| R9 | **B** | 任务书只覆盖 config/JSON/argv；未覆盖 `--server http://user:<password>@host` 的 userinfo、3xx 重定向、错误 body 回显、job index 的 command 字段、`--verbose` 日志。 | 哨兵 token + 哨兵 URL 密码；递归扫描 stdout/stderr/JSON/state/manifest/config/jobs 文件；302 跨端口重定向检查 Authorization。 |
| R10 | **B** | `urllib` 默认跟随重定向，跨 host 仍可能带 Authorization（或把 token 发到攻击者）。 | 两个假服务：A 返回 302 到 B；B 记录 Authorization；要求 B 收不到 token 或客户端拒绝重定向。 |
| R11 | MAJOR | doctor/错误 envelope 打印完整 URL 或服务端 body，URL 中若含 token/query 会泄漏。 | 让服务端 401 body 回显 `Authorization`；断言输出不含 token，body 截断。 |
| R12 | MAJOR | `jobs list/show` 或 `config show` 保存 `--server` 原文，URL userinfo 泄漏。 | 用含 userinfo 的 server 跑批处理；扫描 jobs/config 文件。 |
| R13 | MAJOR | `config set` 接受 `token`/`serve_token`/`QASR_SERVE_TOKEN` 类键。 | 逐个尝试；要求 2，且不写文件。 |

### 3.3 4xx/5xx/超时/重试映射

| ID | 级别 | 风险 | 验证方式 |
|---|---|---|---|
| R14 | MAJOR | 408 是 4xx 但语义是超时；任务书“4xx→4，超时→3”冲突。409/425/451/499 未定义。 | 假服务逐个状态码；要求有明确映射表，不能“所有 4xx 一律 4”或“所有 408 一律 3”自相矛盾。 |
| R15 | MAJOR | `--retries 2` 的计数口径（总尝试 2 次还是 3 次）未定义；429 的 `Retry-After` 是否遵守未定义。 | 假服务计数请求；断言 attempts=3（initial+2）且 429 有退避；记录任务书缺口。 |
| R16 | MAJOR | 部分失败“取首个错误码”可能按完成顺序而非输入顺序；`--jobs` 下结果不稳定。 | 假服务延迟反转；断言错误码/首个错误来自输入顺序。 |
| R17 | MAJOR | 语言拒绝重试不计入 `--retries`，但可能无限循环（每次都 400 language）。 | 假服务永远 400 language；断言最多一次去 language 重试后失败。 |
| R18 | MAJOR | 3xx 重定向、TLS 失败、连接重置、DNS 失败的分类未定义。 | 假服务 302/自签 TLS/直接断开；断言退出码和 retryable。 |
| R19 | MINOR | 服务端 body 截断长度未定义；截断可能切断 UTF-8。 | 返回 100KB 多字节 body；断言输出有上限且 JSON 可解析。 |

### 3.4 180s/270s 窗口口径

| ID | 级别 | 风险 | 验证方式 |
|---|---|---|---|
| R20 | MAJOR | **任务书自相矛盾**：T4.3 说 `align` 超长时窗口化（默认 `--window-seconds 180`），又说“单窗模式仍拒绝 >180s（P0 行为）”。181s 到底是成功还是 4？ | 181s WAV `align --dry-run`；要求实现/文档给出唯一答案，测试按“默认窗口化成功”设计。 |
| R21 | MAJOR | **overlap=0 矛盾**：T4.1 `plan_windows` 校验 `0 < overlap_s`，T4.3 CLI 说 `--window-overlap` 允许 `0–10`。 | `align --window-overlap 0`；要求接受（按 CLI 契约）或任务书改口径。 |
| R22 | MAJOR | 首窗是否有左重叠、核心区长度是否 `window-overlap`、末窗如何 clamp，任务书未给公式；不同实现会导致窗口数/覆盖率差异。 | 181s/360s/3600s dry-run；断言 core 连续、无 gap/overlap、每窗 audio ≤180s、首 core_start=0、末 core_end=duration。 |
| R23 | MAJOR | 能量切点在 ±5s 搜索可能把 `audio_end` 推过 180s；任务书同时要求“单窗 ≤180s”和“能量最低帧”。 | 合成能量突变音频；检查每个 audio 窗口 ≤180s，且 core 不为空。 |
| R24 | MAJOR | `--max-chunk-seconds` 同时承担“纯文本 chunk”和“timestamps 窗口”，而 `align` 用 `--window-seconds`；命名/默认值不一致，用户易误用。 | `--help` 与 dry-run；断言 timestamps 默认 180、纯文本默认 270、>180 与 timestamps 组合退出 4。 |
| R25 | MINOR | 10–1200 边界非法值的退出码（2 vs 4）未定义。 | 9/1201/0/负数；要求全为 2（用法）或文档明确。 |

### 3.5 长音频归属、单调合并与 chunk resume

| ID | 级别 | 风险 | 验证方式 |
|---|---|---|---|
| R26 | **B** | `attribute_reference` 在重复文本/静音段上可能把同一参考区间分给多个窗口，或跨窗重复归属；最终文本重复/丢字。 | 合成“重复短句”参考 + 假窗口文本；1h 集成检查重复率、字符覆盖。 |
| R27 | MAJOR | `merge_items` 的“重叠用中点或 prev.end”未定死；不同选择导致时间轴不同，测试无法稳定。 | 构造重叠 items；要求实现固定规则并在报告写明；断言 `start>=prev.end`、`end>=start`、`end<=duration`。 |
| R28 | MAJOR | 仅平移 start 不删除重复文本；coverage 可能被重复项抬高。 | 构造完全重叠的相同 items；断言最终文本去重或 coverage 不虚高。 |
| R29 | MAJOR | chunk key 未定义是否包含 `max_chunk_seconds`/`window`/`overlap`/model fingerprint/aligner fingerprint；参数变化后可能错误复用。 | 先跑一次生成 chunks，再改 chunk 参数/模型跑 `--resume`；断言重跑或明确失效。 |
| R30 | MAJOR | chunk 文件损坏/截断/半写时 `--resume` 可能当完整复用。 | kill -9/手工截断 chunk JSON；断言重跑该 chunk，不崩溃。 |
| R31 | MAJOR | `--overwrite` 清理 chunk 的作用域可能删掉其他 job 的 chunk。 | 两个 out-dir/job 并存；`--overwrite` 只清当前 job_key。 |
| R32 | MAJOR | 窗口 ASR 失败/空文本时仍调用 aligner；可能得到空 items，coverage 低，错误码 5/6 优先级不清。 | 假窗口文本为空/失败；断言 warning、继续其他窗口、最终码符合任务书。 |
| R33 | MAJOR | `align --language auto` 的 30s LID probe 在前 30s 静音时会失败，即使后面有语音；任务书要求直接 4。 | 合成“前 30s 静音 + 后段语音”音频；记录行为，评估是否应改为可配置。 |
| R34 | MINOR | `language_source="lid_probe"` 不在 P0 `align.json` 的枚举/字段中；schema 可能不兼容。 | `jsonschema` 校验；要求新增枚举值或新字段。 |

### 3.6 `--jobs` 并发

| ID | 级别 | 风险 | 验证方式 |
|---|---|---|---|
| R35 | **B** | 远程 `--jobs` + `--fail-fast` 时，排队任务被取消但 state 留在 `running`（P0 MAJOR-1 复发）。 | 假服务延迟 + 首项失败；检查 state 无 `running`，summary 一致。 |
| R36 | MAJOR | 结果按输入顺序输出/写盘，但线程完成顺序不同；实现可能按完成顺序。 | 假服务反转延迟；`--jsonl`/`--json` 断言顺序等于输入顺序。 |
| R37 | MAJOR | 多线程同时 append state/写 job index，JSONL 行可能交错或丢记录。 | 20 个并发成功项；逐行 JSON 解析，计数/ID 完整。 |
| R38 | MAJOR | `--jobs` 范围 1–32 的非法值（0/33/非整数）退出码未定义。 | 0/33/-1/abc；要求 2。 |
| R39 | MAJOR | 本地 `--jobs>1` 必须 warning 并强制 1；实现可能静默并行或报错。 | 本地空音频 + `--jobs 4`；stderr warning，行为串行。 |
| R40 | MINOR | 远程 `--batch-size` 与 `--jobs` 双重并发未 warning。 | 同时设置；断言 warning 且 batch-size 不增加传输并发。 |

### 3.7 `jobs` / `config` / `models rm` 路径安全

| ID | 级别 | 风险 | 验证方式 |
|---|---|---|---|
| R41 | **B** | `jobs show JOB_ID` 直接用 job_id 拼路径，`../` 可读/写 `QASR_HOME/jobs` 外文件；`jobs show OUT_DIR` 的 state 路径可能被 symlink 指向敏感文件。 | `jobs show ../../etc/passwd`、symlink state；要求 4 且不读/不写外部。 |
| R42 | **B** | `models rm` 对 `QASR_HOME/models/<alias>` 先 `resolve()` 再删；若 alias 是 symlink，会删到外部目录。 | symlink 指向 `/tmp/outside`；`--yes` 后 outside 必须仍存在，命令非 0。 |
| R43 | **B** | `models rm` 允许绝对路径/`..`/`QASR_HOME/models` 根本身，误删用户目录。 | `models rm /tmp/x`、`..`、`.`、`QASR_HOME/models`；全部拒绝。 |
| R44 | MAJOR | `config set --project` 的“最近项目”未定义；可能写到 `/`、home 或错误目录。 | 在临时 cwd/有 `.git` 的目录跑；断言只写 cwd/.qasr/config.json 或明确的项目根。 |
| R45 | MAJOR | `config set` 对 symlink 配置文件 `rename` 覆盖，可能替换外部文件/丢权限。 | `.qasr/config.json` 为 symlink；要求拒绝或安全写入。 |
| R46 | MAJOR | `models rm` 无 `--yes` 且非 TTY 时行为未定义（应拒绝，不能默认删）。 | subprocess 非 TTY；要求 2/4 且不删。 |
| R47 | MINOR | `jobs list --limit` 负数/0/超大值未定义。 | -1/0/100000；要求 2 或合理 clamp。 |

### 3.8 vLLM 与 auto 回退

| ID | 级别 | 风险 | 验证方式 |
|---|---|---|---|
| R48 | **B** | `--backend vllm` 在 macOS/CPU 上静默回退 transformers；任务书明确“显式 vllm 不静默回退”。 | macOS 无 vllm；要求 3 + hint，plan/JSON 不出现 fallback。 |
| R49 | MAJOR | auto 探测 vllm 通过 `import vllm`，可能初始化 CUDA/占显存或导入崩溃；失败后是否正确回退。 | 假 `vllm` 模块抛异常/成功两种；检查 `backend_fallback` 位置和 warning。 |
| R50 | MAJOR | `backend_fallback` 写在顶层还是结果项未定；`transcribe-batch.json`/`transcribe-item.json` schema 可能不兼容。 | `jsonschema` 校验 batch/single；要求字段可选且 schema 通过。 |
| R51 | MAJOR | `--gpu-memory-utilization` 只对 vllm 有效；对 transformers 静默忽略是坏 UX。 | `--backend transformers --gpu-memory-utilization 0.8`；要求 warning 或 2。 |
| R52 | MAJOR | vLLM + `--timestamps` 仍走 HF aligner；若未安装 torch/HF，错误码/提示是否清晰。 | 无 torch 环境 dry-run/real；要求 3 + 安装 hint，不 import 崩溃。 |
| R53 | MAJOR | vLLM 批量默认 8，但 `--batch-size`/`max_inference_batch_size`/`max_new_tokens auto` 的传递和失败组逐条重试未验证。 | 假 `qwen_asr` 模块（repo 测试）+ gpu06 批量。 |
| R54 | MINOR | auto 在 Linux+CUDA 但 vllm 未安装时是否错误地尝试 vllm；应直接 transformers。 | dry-run plan；无 vllm 环境。 |

### 3.9 serve / doctor / 其它

| ID | 级别 | 风险 | 验证方式 |
|---|---|---|---|
| R55 | MAJOR | argparse 未知参数透传与全局 flag 冲突；`qasr serve ... --json` 可能被透传给 server 或被 qasr 消费。 | 假 binary 记录 argv；多种位置组合。 |
| R56 | MAJOR | `os.execvp` 失败/缺 binary 的退出码 3；实际 exec 前若打印 JSON 会污染 server 输出。 | PATH 无 binary、假 binary 记录 argv。 |
| R57 | MAJOR | `doctor --server` 的 URL 合法性、TLS、redirect、空 models、401/5xx 分类未定义；默认 exit 0 与 `ok=false` 可能矛盾。 | 假服务矩阵；strict/非 strict。 |
| R58 | MINOR | `doctor --json` 新 check 的 `name`/`status`/`hint` 不稳定，破坏脚本。 | 多次运行比较字段。 |
| R59 | MAJOR | P1 配置新键 `timeout/retries/window_*` 未加入 `CONFIG_KEYS`/coercion；`server`/`jobs` 仍被 P0 `_reject_p1_config` 拒绝。 | `QASR_CONFIG` 设置 server/jobs/timeout；要求生效而非 2。 |
| R60 | MAJOR | `language_source` 现有 schema 只允许 `auto|forced`；remote `server` 和 align `lid_probe` 会 schema 失败。 | `jsonschema` 校验 remote/align JSON。 |

### 3.10 a53859c 之后仍需盯住的歧义/缺口

| ID | 级别 | 剩余问题 | 处理 |
|---|---|---|---|
| X1 | ~~MAJOR~~ 已定稿 | `--stdin-list` 无参数、读 stdin（fcca068）。 | 用例保持 stdin 形式。 |
| X2 | ~~MINOR~~ 已定稿 | `tsv/md` 与 `--json/--jsonl` 无条件互斥（fcca068）。 | 已加 rc=2 用例。 |
| X3 | ~~MINOR~~ 已定稿 | `jobs list --limit` 1–10000，0/负数/超大 → 2；`jobs show OUT_DIR` 必须已登记。 | 已加 10001 与越界用例。 |
| X4 | MINOR | `config get KEY` 人类输出是纯值、`key=value` 还是带来源未定义。 | 测试纯值；若不同，记录 MINOR 并改断言。 |
| X5 | MAJOR | chat 请求是否必须带 `model`、音频 part 用 `audio_url` 还是 `input_audio`，真实 vLLM 未在 gpu06 明确。 | 假服务两种都接受；报告记录真实服务实测。 |
| X6 | MAJOR | 429 `Retry-After` 与 `--timeout` 的“总延迟受 timeout 约束”可能被实现为单请求超时，语义仍含糊。 | 只断言 1s Retry-After 被遵守且不会无限重试；总预算语义留待报告。 |
| X7 | MAJOR | 能量切点（±5s）与 180s 硬上限的交互只能靠实现 clamp；黑盒 dry-run 不暴露能量包络。 | GPU/1h 证据检查每窗 ≤180s；纯函数由 repo 单测覆盖。 |
| X8 | MAJOR | 长音频文本去重/不丢字无法仅靠假服务验证；窗口文本拼接质量需要真实 ASR。 | 1h GPU 证据 + 重复短句参考文本；黑盒只锁窗口数和时间轴。 |

---

## 4. 黑盒测试矩阵

说明：`rc` 为退出码；`out` 为 stdout；`err` 为 stderr；`json` 指 `--json` envelope。所有命令通过 `python -m qasr_cli` 或 `qasr` 子进程执行。

### 4.1 N1 — serve / doctor --server

| ID | 命令 / 输入 | 期望 |
|---|---|---|
| N1-01 | `qasr serve --dry-run --json` | rc=0；json `command="serve"`、`argv` 为 list、`model` 解析为 repo/path；无 vllm/torch import |
| N1-02 | `qasr serve 1.7b --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.8 --dry-run --json` | rc=0；argv 含 host/port/利用率；未知参数原样在 argv |
| N1-03 | `qasr serve 1.7b --tensor-parallel-size 2 --foo=bar --dry-run --json` | rc=0；未知参数不报错，顺序稳定 |
| N1-04 | `PATH` 无 `qwen-asr-serve`，`qasr serve 1.7b` | rc=3；err 含安装 hint；无 traceback |
| N1-05 | PATH 放假 `qwen-asr-serve`，`qasr serve 1.7b --port 18000` | 假 binary 收到完整 argv；qasr 不额外污染 stdout |
| N1-06 | `qasr serve --gpu-memory-utilization 2 --dry-run` | rc=2 或明确用法错误（不允许静默透传非法值） |
| N1-07 | 假服务 `/v1/models` 200 + 2 个模型；`qasr doctor --server URL --json` | rc=0；check 有 server/models/latency；模型 id 正确；token 不出现 |
| N1-08 | 假服务 401；`doctor --server URL --json` | rc=0；server check fail/warn + hint；`--strict` rc=3 |
| N1-09 | 假服务 500；`doctor --server URL --json` | rc=0；server check fail/warn；`--strict` rc=3 |
| N1-10 | 假服务 sleep > timeout；`doctor --server URL --timeout 1 --json` | rc=0；server check fail/warn（超时） |
| N1-11 | `doctor --server not-a-url --json` | 不崩溃；失败 check 或 rc=2（按任务书定稿后固定） |
| N1-12 | 哨兵 `QASR_SERVE_TOKEN`；`doctor --json`/`config show --json`/`serve --dry-run --json` | 哨兵不出现在任何 out/err/JSON；doctor 只报 set/unset |

### 4.2 N2 — 远程转写与错误

| ID | 命令 / 输入 | 期望 |
|---|---|---|
| N2-01 | 假服务 transcription 200 `{"text":"language Chinese<asr_text>你好"}`；`qasr --server URL transcribe a.wav --json` | rc=0；`result.text="你好"`、`language="Chinese"`、`language_source="server"`、schema 通过；无前缀文本时 `language=null` |
| N2-02 | 同上，检查 multipart | `file` 文件名/MIME 正确；`model` 来自 `/v1/models` 首个；`language` 缺省不传 |
| N2-03 | `--language zh` / `--language auto` | `zh` 传 `language=zh`；auto 不传 |
| N2-04 | `--model server-model-2` | 覆盖默认 model；请求字段为 `server-model-2`；GET /v1/models 仍只一次 |
| N2-05 | `--context hotword` | 发 `/v1/chat/completions`；system content 含 hotword；user content 含 `audio_url` data URL |
| N2-06 | context + `--language zh` | **不发** assistant prefill/`continue_final_message`；warning 说明 chat 无法强制语言；语言取服务端解析值 |
| N2-07 | 假 chat 返回 `language Chinese<asr_text>你好` | `result.text="你好"`、`language="Chinese"` |
| N2-08 | 假 chat 对任何 assistant/`continue_final_message` 返回 400 | 客户端只发 1 次、不含 prefill；rc=0；若发两次则记 MINOR（多余请求） |
| N2-09 | 假 transcription 第一次 400 “unsupported language”，第二次 200 | 两次请求；第二次无 language；warning；rc=0 |
| N2-10 | 假 transcription 404，chat 200 | 先 transcription 后 chat；warning 降级；rc=0；同进程后续不再试 transcription |
| N2-11 | 假服务 401 | rc=4；error.kind/code、`details.http_status=401`、retryable=false；token 不出现 |
| N2-12 | 假服务 500，`--retries 2` | 总请求 3 次；rc=3；retryable=true；sleep/backoff 有界 |
| N2-13 | 假服务 429 + Retry-After | 重试；rc=3（最终）；记录是否遵守 Retry-After |
| N2-14 | 假服务 sleep > `--timeout 1` | rc=3；retryable=true；无 traceback |
| N2-15 | 两个文件：第一个 401、第二个 200；`--json` | rc=6；summary total=2 ok=1 failed=1；errors 只含第一个 |
| N2-16 | 两个文件都失败（401/500）；`--json` | rc=首个输入错误的码（4），不是完成顺序 |
| N2-17 | `--timestamps` / `--granularity` / `align --server` | rc=4；capability message；假服务零请求 |
| N2-18 | `--format srt` 远程 | rc=4；提示时间戳不可用 |
| N2-19 | URL 输入 + 假音频站 + 假 transcription | 先下载/缓存，再 multipart；rc=0 |
| N2-20 | manifest `audio_base64` + 假 transcription | 请求 body 解码后字节等于原音频；rc=0 |
| N2-21 | `--server` + `--backend transformers` | rc=2；冲突 |
| N2-22 | `--backend remote` 无 `--server` | rc=2 |
| N2-23 | 远程 batch `--jsonl` 两个文件，假服务反转延迟 | rc=0；2 行，顺序等于输入顺序 |
| N2-24 | 远程 `--resume` 第二次 | 第二次全部 `skipped`，假服务请求数不增加（或仅模型探测） |
| N2-25 | 181s WAV + `--server --max-chunk-seconds 60` | 客户端至少 4 个 transcription 请求；每请求 body ≤25MB；文本按输入顺序拼接 |
| N2-26 | 远程 `--max-chunk-seconds 1200` | 降为 600 并 warning；不得发单个超 25MB 请求 |

### 4.3 N2 — 远程 `--jobs`

| ID | 命令 / 输入 | 期望 |
|---|---|---|
| N2-30 | 10 文件 + `--jobs 4`，假服务 delay | 最大并发 `>1` 且 `<=4`；rc=0；顺序稳定 |
| N2-31 | `--jobs 1` | 最大并发=1；严格串行 |
| N2-32 | `--jobs 0/33/-1` | rc=2 |
| N2-33 | `--jobs 4 --fail-fast`，首项 401 | rc=4；state 无 running；未开始项 skipped/pending；summary 一致 |
| N2-34 | 本地 `--jobs 4` 空音频 | stderr warning；实际串行；rc=0 |
| N2-35 | 远程 `--batch-size 8 --jobs 4` | warning：batch-size 不增加传输并发；最大并发仍 ≤4 |

### 4.4 N3 — vLLM / auto

| ID | 命令 / 输入 | 期望 |
|---|---|---|
| N3-01 | macOS 无 vllm：`qasr --backend vllm --dry-run transcribe a.wav --json` | rc=3；hint 安装；无 traceback；不 import 崩溃 |
| N3-02 | macOS：`qasr --backend auto --dry-run transcribe a.wav --json` | plan `backend="transformers"`；device mps/auto；无 vllm |
| N3-03 | `--backend transformers --gpu-memory-utilization 0.8` | warning 或 rc=2；不得静默忽略 |
| N3-04 | 假 `vllm` 导入成功 + CUDA 假模块（如实现支持） | auto 选择 vllm；plan/model.backend=vllm |
| N3-05 | 假 `vllm` 导入抛错 | auto 回退 transformers；`backend_fallback=true`；warning |
| N3-06 | 显式 `--backend vllm` + 假导入抛错 | rc=3；不得回退；无 `backend_fallback` |
| N3-07 | `--backend vllm --timestamps --dry-run` | rc=0 或 3（按平台）；plan 含 aligner；不崩溃 |
| N3-08 | `--server URL --backend vllm` | rc=2 冲突 |

### 4.5 N4 — 长音频窗口

| ID | 命令 / 输入 | 期望 |
|---|---|---|
| N4-01 | `align 181s.wav --dry-run --json --window-seconds 180` | rc=0；plan.windows>=2；每窗 audio ≤180s；core 连续 |
| N4-02 | `align 180s.wav --dry-run --json` | 单窗；与 P0 plan 形状一致 |
| N4-03 | `align 181s.wav --dry-run --json`（默认 window=180） | rc=0；窗口化（按任务书 T4.3）；若实现坚持 P0 拒绝，记录任务书冲突 |
| N4-04 | `align --window-seconds 181` | rc=2 |
| N4-05 | `align --window-overlap 0` | rc=0（按 CLI 0–10）；若 2，记录 T4.1/T4.3 矛盾 |
| N4-06 | `align --window-overlap 10 --window-seconds 10` | rc=2（overlap >= window） |
| N4-07 | `transcribe --timestamps --max-chunk-seconds 181 --dry-run` | rc=4；hint aligner ≤180 |
| N4-08 | `transcribe --timestamps --max-chunk-seconds 180 --dry-run` | rc=0；plan 窗口 |
| N4-09 | `transcribe --max-chunk-seconds 9/1201` | rc=2 |
| N4-10 | `transcribe 25s.wav --max-chunk-seconds 10 --dry-run --json` | plan.windows>=3；窗口边界覆盖全时长 |
| N4-11 | 181s 窗口 plan invariants | 首 core_start=0；末 core_end=duration；core 无 gap/overlap；audio_start<=core_start；core_end<=audio_end；每窗 ≤180s |
| N4-12 | 能量突变 181s（合成） | 切点偏移后仍满足 N4-11；core 非空 |
| N4-13 | `align --language auto` 空 WAV | rc=4；hint 11 语言；不得静默选 en |
| N4-14 | `align --language auto` 前 30s 静音、后段语音（集成） | 记录是 4 还是可用；任务书要求 4，评估 UX |
| N4-15 | 1h 合成音频 `--dry-run`（慢测试） | 窗口数合理；不加载模型；耗时/内存可接受 |
| N4-16 | 远程 181s + `--max-chunk-seconds 60` | 客户端切块 ≥4；每块 ≤25MB；拼接文本不重复整段 |

### 4.6 N5 — 批量、输出、运维

| ID | 命令 / 输入 | 期望 |
|---|---|---|
| N5-01 | `printf 'a.wav\n\nb.wav\n' \| qasr transcribe --stdin-list --dry-run --json` | rc=0；inputs=[a,b]；空行跳过；顺序稳定 |
| N5-02 | `--stdin-list` + `--manifest` | rc=2 |
| N5-03 | `--stdin-list` + 位置参数 | 允许；stdin 在前、位置参数在后；去重 |
| N5-04 | 输入行 `-` | 不当作文件；rc=0 plan 或 4（按定稿），不得读 stdin 二次 |
| N5-05 | `--skip-existing`，输出已存在且非空 | rc=0；item skipped；summary.skipped=1；不加载模型 |
| N5-06 | `--skip-existing`，输出零字节 | 不 skip；重跑（rc=0/6 按输入） |
| N5-07 | `--retry`，假服务 500→200 | 重试一次；attempts=2；rc=0 |
| N5-08 | `--retry --fail-fast` | rc=2 |
| N5-09 | 单文件 `-f tsv` stdout | 表头 `id\tstart\tend\ttext\tlanguage`；字段转义 tab/newline |
| N5-10 | 批量 `-f tsv,md -o out` | 每 item 两文件；tsv 行数/段数正确；md `## id`/列表正确 |
| N5-11 | 任意 `-f` 格式（含 tsv/md）与 `--json/--jsonl` | rc=2（无条件互斥）；不得发 HTTP 请求 |
| N5-12 | 批量空音频 + `-o out --json` | QASR_HOME/jobs/<id>.json 创建；jobs list 可见 |
| N5-13 | `jobs list --json --limit 1` | 倒序；schema `jobs-list.json`；state missing 标记 |
| N5-14 | `jobs show <job_id> --json` / `jobs show <out_dir> --json` | 计数与 state 一致；未知 job rc=4 |
| N5-15 | `jobs show ../../etc/passwd` | rc=4；不读外部 |
| N5-16 | `bench --dry-run --json` | rc=0；plan 含 synthetic/duration/iterations；不加载模型 |
| N5-17 | `bench --json`（真实短音频，集成） | schema `bench.json`；rtf/peak_memory 可为 null；无准确率声明 |
| N5-18 | `models show 1.7b --json`（合成 complete/incomplete fixture） | verify 状态正确；schema；source.json 摘要 |
| N5-19 | `models rm <managed> --dry-run` | 打印路径；不删 |
| N5-20 | `models rm <managed> --yes` | 只删托管目录；输出字节数 |
| N5-21 | `models rm <symlink-to-outside> --yes` | 拒绝；outside 仍存在 |
| N5-22 | `models rm /tmp/x` / `..` / `.` | 拒绝；不删外部 |
| N5-23 | `config get backend`; `config get nope` | 有效值/来源；未知 key rc=2 |
| N5-24 | `config set backend vllm`; `config set device bogus` | 写入成功；非法枚举 rc=2 |
| N5-25 | `config set server URL`; `config show --json` | server 生效；来源正确 |
| N5-26 | `config set QASR_SERVE_TOKEN x` / `token x` / `serve_token x` | rc=2；文件不写 |
| N5-27 | `config set --project ...` | 写 cwd/.qasr/config.json；保留未知键；原子 |
| N5-28 | 新 schema 校验（jobs/bench/models-show/config-get/remote） | `jsonschema` 全通过；`language_source` 枚举兼容 |

---

## 5. 对抗用例

| ID | 场景 | 期望 |
|---|---|---|
| A-01 | 302 重定向到另一个 host，携带 Bearer | 不把 token 发给新 host；拒绝或去 auth 后重试 |
| A-02 | 401 body 回显 Authorization / 100KB 文本 | 输出不含 token；body 截断且 JSON 可解析 |
| A-03 | `--server http://user:<password>@host` | pass 不出现在 doctor/config/jobs/JSON |
| A-04 | 假服务每请求 400 language | 最多一次去 language 重试，不无限循环 |
| A-05 | 假服务 500 无限 | `--retries 2` 后停止；attempts 精确 |
| A-06 | 假服务 sleep 到超时 | rc=3，retryable，无 traceback |
| A-07 | 100 文件 + `--jobs 8` + 首项失败 + fail-fast | 无 running；summary 一致；未开始项 skipped |
| A-08 | 20 并发写 state | JSONL 每行合法；无丢失/交错 |
| A-09 | kill -9 在 chunk 写一半 | 旧 chunk 仍可读；resume 重跑该 chunk |
| A-10 | 截断/损坏 chunk JSON | resume 不崩溃；重跑该 chunk |
| A-11 | 改 `--max-chunk-seconds` 后 `--resume` | 不复用旧参数 chunk |
| A-12 | 1h 参考文本重复短句 | 不跨窗重复归属；coverage 不虚高 |
| A-13 | 重叠 items 完全相同 | 去重或单调平移后不产生重复文本 |
| A-14 | `models rm` symlink/硬链接/根目录 | 不删除外部；根目录拒绝 |
| A-15 | `config set --project` 在 `/` 或 home 无项目 | 不写系统目录；明确错误 |
| A-16 | `jobs show` 路径穿越/symlink state | 不读外部；rc=4 |
| A-17 | PATH 假 `qwen-asr-serve` 记录 argv | token/env 不泄漏；argv 顺序正确 |
| A-18 | 假 `vllm` 导入抛 `OSError`/`ImportError` | auto 回退；显式 vllm 3；无 traceback |
| A-19 | `--backend auto` 在无 torch 环境 | 3 + 安装 hint；不 import 崩溃 |
| A-20 | stdout 被 `head` 提前关闭（P0 回归） | rc=0；无 BrokenPipe traceback |

---

## 6. 假 OpenAI 服务与 fixture 设计

### 6.1 假服务

`review_tests_p1/fake_openai.py` 提供 `FakeOpenAIServer`：

- `ThreadingHTTPServer` 监听 `127.0.0.1:0`；线程安全记录 `requests`、`active`、`max_active`。
- 端点：
  - `GET /v1/models`：默认 `{"object":"list","data":[{"id":"Qwen/Qwen3-ASR-1.7B","object":"model"}]}`。
  - `POST /v1/audio/transcriptions`：multipart 解析（stdlib `email`）；记录 fields/files。
  - `POST /v1/chat/completions`：JSON 解析；记录 messages/audio_url/prefill/model。
- 行为注入：
  - 固定响应、按请求序号响应序列、延迟、401/403/404/405/413/422/429/500/502、`Retry-After`、302 Location。
  - 语言拒绝（400 message 含 language/unsupported）、prefill 拒绝（400 unknown field）、空 models、慢响应。
- 认证：记录 `Authorization`，但测试只断言存在/不存在，不打印。
- 并发：handler 进入时 `active += 1`，`max_active=max(...)`，退出递减；测试用 barrier/sleep 稳定测量。

### 6.2 音频 fixture

- `short.wav`：1s、16k、单声道、静音/正弦；用于远程/批处理。
- `medium.wav`：25s；用于 chunk 规划。
- `long.wav`：181s；用于 180s 边界（约 5.8MB，临时目录生成）。
- `hour.wav`：1h；只在 `slow`/GPU 测试生成（约 115MB，避免提交）。
- `silent_prefix.wav`：前 30s 静音 + 后 10s 正弦；用于 `align --language auto`。
- 所有 WAV 用 stdlib `wave` 生成；不依赖 numpy/soundfile。

### 6.3 模型 fixture

- 合成半包/完整模型：复用 P0 review helpers 的思路，复制 `config.json`/`tokenizer_config.json`/`vocab.json`，配截断 shard 或 `.aria2`；**不**依赖真实 1.7B 必然 incomplete。
- 真实模型：`QASR_TEST_MODEL_DIR` 指向 0.6B/aligner；集成测试用。

### 6.4 PATH 假 binary

- `fake-qwen-asr-serve`：可执行 shell/python 脚本，把 argv 写到 `QASR_TEST_ARGV_FILE` 后 exit 0；用于验证 `serve` 透传/exec。
- `no-serve-bin`：空 PATH 目录；用于缺 binary 3。

---

## 7. 与 `P1-TASKS.md` 验收项对照 checklist

### N1

- [ ] N1-01..06：serve 语法/透传/dry-run/缺 binary/exec 成功。
- [ ] N1-07..11：doctor --server 200/401/500/timeout/非法 URL。
- [ ] N1-12：token 不出现在 config/doctor/argv；只从 env 读。
- [ ] N1 验收：无 vllm 环境 dry-run；假服务 doctor；token 不泄漏。

### N2

- [ ] N2-01..06：transcription multipart、chat context、prefill、语言。
- [ ] N2-07..10：parse、prefill 400、language 400、404 降级。
- [ ] N2-11..16：401/500/429/timeout/部分失败/全失败映射。
- [ ] N2-17..20：capability error、srt 拒绝、URL/base64 输入。
- [ ] N2-21..24：backend 冲突、jsonl 顺序、resume。
- [ ] N2-30..35：`--jobs` 并发/串行/非法/fail-fast/本地 warning/batch-size。
- [ ] N2 验收：假服务覆盖全部错误；并发度；1h 链路留 GPU。

### N3

- [ ] N3-01..08：vllm 显式失败、auto 选择/回退、gpu-memory、timestamps、server 冲突。
- [ ] N3 验收：Mac auto transformers；显式 vllm 3；假 qwen_asr 参数；gpu06 vllm 短音频/批量。

### N4

- [ ] N4-01..12：181s/180s/overlap 0/窗口不变量/能量切点/chunk 参数。
- [ ] N4-13..15：align auto 空音频/静音前缀/1h dry-run。
- [ ] N4 验收：纯函数边界；1h dry-run；1h timestamps/align（集成/GPU）；kill -9 chunk resume。

### N5

- [ ] N5-01..08：stdin-list/skip-existing/retry/互斥。
- [ ] N5-09..11：tsv/md/转义/互斥。
- [ ] N5-12..15：jobs 索引/list/show/路径安全。
- [ ] N5-16..17：bench dry-run/schema。
- [ ] N5-18..22：models show/rm/路径/symlink。
- [ ] N5-23..27：config get/set/枚举/token/project。
- [ ] N5-28：新 schema + `language_source` 兼容。
- [ ] N5 验收：README 命令可复制；kill -9 resume；schema 全通过。

### 横切

- [ ] 无 torch/vllm 环境 `qasr --help`、`serve --dry-run`、`jobs list`、远程命令可启动。
- [ ] 所有新增 JSON 有 `schema_version: 1`；旧字段不删不改名。
- [ ] 退出码表 0/2/3/4/5/6/130 一致。
- [ ] stdout 只有结果；第三方日志在 stderr。
- [ ] token 哨兵扫描：stdout/stderr/JSONL/state/manifest/jobs/config/argv 全部无泄漏。

---

## 8. GPU 验收复核点（gpu06）

审查者不操作服务器；协调者提供以下证据后逐项复核：

| ID | 证据 | 通过标准 |
|---|---|---|
| G-01 | `nvidia-smi` + `CUDA_VISIBLE_DEVICES` + 版本 | 使用空闲卡；不影响他人 |
| G-02 | `qasr serve 1.7b --host 127.0.0.1 --port 18000` 启动日志 | 服务起来；端口正确；无 token 打印 |
| G-03 | `qasr doctor --server http://127.0.0.1:18000 --json` | models/延迟；401/500/超时分类 |
| G-04 | `qasr --server ... transcribe short.wav --json` | 文本与本地 transformers 一致（允许标点差异） |
| G-05 | `qasr --server ... transcribe 1h.flac --jobs 4 --json` | rc=0；耗时/RTF；无超时；结果顺序 |
| G-06 | `qasr --backend vllm transcribe batch/ --batch-size 8 --json` | rc=0；批处理/失败隔离；backend=vllm |
| G-07 | `qasr transcribe 1h.flac --timestamps --language zh -f srt,json -o out` | 时间单调、无跨窗重复、SRT 合法、coverage 合理 |
| G-08 | `qasr align 1h.flac --text-file t.txt --language zh -f srt,json -o out` | 覆盖率 ≥0.5；时间单调；SRT 合法 |
| G-09 | kill -9 1h 任务后 `--resume` | 只跑未完成 chunk；最终输出完整 |
| G-10 | 远程 401/500/超时故障注入 | 4/3/3；retryable 正确 |
| G-11 | `qasr --backend auto` | 选择 vllm；无 fallback；doctor 报告一致 |
| G-12 | `qasr bench sample.wav --iterations 3 --json` | schema；RTF 记录；无准确率声明 |

GPU 证据只作验收复核，不在 Mac 上伪造；任何需要从 Mac 传文件的步骤先停止并询问用户。

---

## 9. 测试套件文件规划

```
review_tests_p1/
  README.md                 # 运行方式、标记、里程碑映射
  conftest.py               # qasr runner、临时 home、WAV fixture、假服务 fixture
  helpers.py                # JSON/JSONL、WAV、schema、断言、token 扫描
  fake_openai.py            # 假 OpenAI/vLLM 服务
  test_n1_serve_doctor.py
  test_n2_remote_transport.py
  test_n2_remote_errors.py
  test_n2_remote_concurrency.py
  test_n3_vllm_auto.py
  test_n4_long_audio.py
  test_n5_batch_ops.py
  test_n5_jobs_bench.py
  test_n5_models_config.py
  test_security_token.py
  test_schemas_p1.py
  test_integration_p1.py    # RUN_INTEGRATION=1 才收集执行
```

原则：

- 测试不 import `qasr_cli`；只通过 subprocess 和 HTTP 观察。
- 假服务不依赖第三方库；multipart 用 stdlib 解析。
- 每个失败断言都打印 rc/stdout/stderr/请求记录，便于出报告。
- 需要真实模型的测试统一 `@pytest.mark.integration`，并在缺 `QASR_TEST_MODEL_DIR` 时 `pytest.skip`。
- GPU 测试统一 `@pytest.mark.gpu`，默认不运行。

---

## 10. 交付物

1. 本计划：`P1-REVIEW-PLAN.md`。
2. 独立套件：`review_tests_p1/`（可在无 GPU 环境运行；当前实现未完成时预期红）。
3. 每个里程碑后：跑对应测试子集，记录原始输出。
4. 最终：`P1-REVIEW-REPORT.md`，含 BLOCKER/MAJOR/MINOR、文件:行号、复现命令、证据、最小修复建议、PASS/FAIL 判定。

---

## 11. 初审问题清单（给协调者的 status）

> 以下 1–8 是 Phase 1 初次 status 的原始记录；其中窗口公式、overlap、merge、
> retries/错误映射、fail-fast/state、路径安全、远程 prefill/语言/25MB 切块已由
> `P1-TASKS.md` a53859c 定稿。仍未闭合的歧义见 §3.10（X1–X8）。

1. **远程契约**：transcription vs chat 的真实字段/`model`/`continue_final_message` 未在 gpu06 验证；`yue` 非 ISO-639-1，语言降级可能丢强制语种。
2. **Token 泄漏**：URL userinfo、302 跨 host、错误 body、job index command 是任务书未覆盖的泄漏面；需把“任何输出/文件不含 token”作为 BLOCKER 门禁。
3. **错误映射**：408/409/425/3xx 未定义；`--retries 2` 计数、429 Retry-After、部分失败“首个错误”按输入还是完成顺序需定稿。
4. **180/270 口径**：align >180 是窗口化还是 P0 拒绝，任务书自相矛盾；`overlap=0` 在 T4.1/T4.3 矛盾；首窗/末窗 core 公式未定。
5. **长音频合并**：重叠规则“中点或 prev.end”未定死；重复文本可能抬高 coverage；chunk key 未要求包含窗口参数/模型指纹，损坏 chunk 复用有数据风险。
6. **`--jobs`**：fail-fast 取消后的 state、并发 append、结果顺序、非法 jobs 值需黑盒锁死；P0 MAJOR-1 有复发风险。
7. **路径安全**：`models rm` 的 symlink/根目录/`..`、`jobs show` 的 job_id 穿越、`config set --project` 的“最近项目”定义都需要明确拒绝面。
8. **vLLM auto**：显式 vllm 不得回退；auto 探测 import 可能初始化 CUDA；`backend_fallback` 放置位置与 schema 兼容需定稿。

---

## 12. N1 实测基线（be65946）

> 审查对象固定为 `git archive be65946` 导出到 `/tmp`；实现者并行写 N2 时工作树
> 有未提交改动，不能直接用于 N1 取证。

- 仓库测试：217 passed / 6 skipped（project venv，loopback 已授权）。
- `ruff check qasr_cli tests`：全绿。
- P0 独立回归：`review_tests/` 165 passed / 17 skipped（3 个 P0“P1 尚不可用”断言已按 P1 语义更新）。
- N1 独立套件：`review_tests_p1/test_n1_serve_doctor.py` 15 passed / 6 xfailed；另有
  `test_security_token.py` 1 个 config-show userinfo xfail。xfail 均为 strict，修复后转 XPASS。
- 真实 tunnel 冒烟：`doctor --server http://127.0.0.1:18000` → server ok，
  1 model（qwen3-asr-1.7b），223–277 ms；torch-free 环境顶层 status=fail 属预期。

已实锤、待实现者处理的 N1 缺口（详见 status msg_9c6413ca1516）：

1. MAJOR：`--server` URL 带 userinfo 时 `urllib` 把 `user:pw@host` 当 host，DNS 失败；
   同地址去掉 userinfo 即可连。`remote.py:59-68,139`。
2. MAJOR：`doctor --json` 的 server check `value` 缺 `http_status`/`auth_rejected`，
   不满足 T1.2 的 HTTP 状态与 auth 拒绝可读性；`doctor.py:221-226`。
3. MINOR：`serve --gpu-memory-utilization 2` 仍 dry-run 成功；`serve.py:102`。
4. MINOR：MODEL 位于未知 flag 之后会被吞并回退 1.7B；`cli.py:303-311`。
5. MINOR：`serve --dry-run --json` 无 schema（dry-run.json 的 command enum 不含 serve）。
6. MINOR：`config show` 原样打印 `--server` userinfo；`commands/config.py:39-49`。
7. MINOR：userinfo + 超范围端口时 doctor 走 runtime error（redact_url 内 `parsed.port` 抛出）。

设计冲突（非实现 bug）：`--config` 文件的 `server` 目前压过 `QASR_SERVE_URL`
（P0/DESIGN §7 的层级），而 T1.2 写的是 CLI > env > config；需协调者定稿。

---

## 13. N2 实测基线（16118cd）

> 审查对象固定为 `git archive 16118cd` 导出到 `/tmp/qasr-n2-src`；N3 并行开发，
> 不使用主工作树取证。N2 只覆盖远程客户端与远程并发，不把 N3–N5 的行为算失败。

### 13.1 结果

- N2 独立套件（排除 `n3/n4/n5/integration/gpu`）：**79 passed / 20 xfailed / 0 failed**。
- 仓库测试（project venv，loopback 已授权）：**242 passed / 6 skipped**。
- `ruff check qasr_cli tests`：全绿。
- 原始 P0 v0.1.0 回归套件：**163 passed / 1 skipped / 3 failed**；3 个失败均为
  P0 时代“P1 尚不可用”旧断言（serve / doctor --server / align --server），属预期。
- 真实隧道（`http://127.0.0.1:18000`，served id `qwen3-asr-1.7b`）：
  - 短音频无 context：rc=0，text 正确，`language=Chinese`、`language_source=server`。
  - context + `--language en`：rc=0，服务端仍返回 Chinese，warnings 为
    `language_not_enforced_on_chat` + `server_ignored_language`。
  - `--model doesnotexist`：rc=4，hint 列出 `qwen3-asr-1.7b`。
  - `doctor --server`：server check ok，1 model，约 371 ms；torch-free 环境顶层
    status=fail 属预期。

### 13.2 N2 发现

1. **N2-BLOCKER-1：5xx 错误 body 未脱敏，token 泄漏到 stdout/stderr/state.jsonl。**
   `qasr_cli/remote.py:289-292`（`_body_snippet` 只截断不 redact）与
   `qasr_cli/remote.py:302`（body 直接拼进 `error.message`）；失败项又在
   `qasr_cli/transcribe.py:578-586` 原样写入 state。
   复现：`pytest --runxfail review_tests_p1/test_security_token.py::test_failed_remote_job_state_has_no_token`。
   证据：rc=3，`stdout_has_token=True stderr_has_token=True state_has_token=True`；
   state 行 message 为 `...echo Bearer {{SECRET_...}}...`。
   最小修复：在任何 body 进入 message 前按当前 `QASR_SERVE_TOKEN` 及
   `Authorization: Bearer ...` 模式替换为 `***`；保留 2 KiB 截断。
2. **N2-MINOR-1：远程模式仍尝试 `import torch` 来解析 device/dtype。**
   `qasr_cli/commands/common.py:34-51` + `qasr_cli/commands/transcribe.py:354`。
   远程不需要本地 runtime，torch-free 环境虽能靠异常回退继续，但会无谓加载
   torch（有 torch 的 Mac 上每个远程请求都付启动成本，也违背“远程只用标准库”的意图）。
   复现：`pytest --runxfail review_tests_p1/test_n2_remote_transport.py::test_remote_does_not_import_torch`。
   证据：import blocker 命中 `torch`，rc 仍为 0。
   最小修复：`is_remote` 时跳过 `resolve_device_dtype`，直接使用配置字符串或
   `auto`，payload 的 device/dtype 保持现状。
3. **N2-MINOR-2：远程 warning 只写 JSON，不写 stderr；且 404 缓存后的后续项不再 warning。**
   `qasr_cli/remote.py:552-616`：warnings 只 append 到结果，从未 `self._warn`。
   复现：`pytest --runxfail review_tests_p1/test_n2_remote_transport.py::test_remote_warnings_are_echoed_to_stderr`。
   证据：`--language en` 的 mismatch 只出现在 `result.warnings`；stderr 只有
   `remote backend ready` / `ok`。404 fallback 第一项有 warning，第二项
   （命中 `_CHAT_ONLY_SERVERS`）warnings 为空。
   最小修复：生成每条 warning 时同时 `_warn`（可用集合去重），缓存命中 chat 时
   也追加降级原因。

### 13.3 任务书歧义（不是实现 bug）

1. **无 `--model` 的默认 served model**：T2.1 写“默认取 `/v1/models` 首个”，
   但 `commands/transcribe.py:330-333` 总是把内置默认 1.7B 传给远程，
   导致 `remote.py:439-445` 的“首个”分支实际不可达；多模型且都不匹配时 rc=4。
   P1-REPORT §4.8 记录的是“单模型 + 已知 alias 才 fallback”。需二选一：
   区分“用户显式 --model”与“内置默认”，或修改 T2.1 文案。
   strict xfail：`test_default_model_is_first_models_entry`。
2. ~~fail-fast 部分成功时的退出码~~（已定稿）：fail-fast 取输入顺序首个
   错误码（3/4/5），非 fail-fast 的部分失败才用 6。实现已在 064cf0a
   对齐，`review_tests_p1` 按 4/3/5 断言。

### 13.4 测试口径修正（本轮已改）

- batch envelope 的 item id 在 `item["input"]["id"]`，不是顶层 `id`；JSONL 同理。
- `state.jsonl` 是 append-only；检查终态必须取每个 id 的最后一条记录，
  历史 `running` 行是正常设计。
- fail-fast 有成功项时按部分失败 rc=6。
- 404→chat 的“缓存”用例改为 `--jobs 1`，避免并发首请求同时 404 的竞态。
- N3/N4/N5 用例加 `n3/n4/n5` marker；N2 用
  `-m 'not n3 and not n4 and not n5 and not integration and not gpu'` 运行。
- N4/N5 才交付的 schema/命令（align windows、jobs/models/config、tsv/md）
  标 strict xfail，交付后 XPASS 再移除。

---

## 14. 最终结论（N1–N5，HEAD 6a0f524）

**判定：PASS。BLOCKER 0 / MAJOR 0 / MINOR 0。**

`cf1c9a4` 的 5 项修复已全部复验：

- `2e1c023`：exactly-180s align 单窗（原 MAJOR-1）。
- `7ebc369`：align `--dry-run` 不再 import torch（原 MINOR-1）。
- `6315c5c`：`dry-run.json` 声明 `plan.windows`（原 MINOR-2）。
- `94fbfb2`：`models rm` 拒绝校内 `..`（原 MINOR-3）。
- `cf1c9a4`：usage error 不再登记 job（原 MINOR-4）。

`cf1c9a4` 引入的活跃锁回归已由 `6a0f524` 修复：锁检查回到
`--retry/--fail-fast` 后、`_normalize_formats` 前，`register_job` 仍在末尾。

最终实测（干净 archive `6a0f524`）：

| 套件 | 结果 |
|---|---|
| `review_tests_p1 -m 'not integration and not gpu'` | 204 passed / 0 xfailed / 0 failed |
| N5 子集 | 69 passed / 0 failed |
| `review_tests`（P0 回归） | 164 passed / 1 skipped |
| 仓库 `tests` | 357 passed / 6 skipped |
| `ruff check qasr_cli tests` | pass |
| P1 integration（本地 0.6B/aligner） | 4 passed / 2 skipped |

4 个原 strict xfail 已全部转为正向断言并 XPASS，当前 0 xfail / 0 XPASS。
GPU/vLLM/1h 证据仍以协调者 gpu06 实测为准，本轮未独立复跑 GPU；该限制不阻塞
P1 代码验收。建议以 `6a0f524` 作为 P1 最终验收基线。

