# qasr-cli P1 实现报告

> 任务书：`P1-TASKS.md`（含 2026-11-18 a53859c 澄清）。
> 状态：N1–N5 已完成（N4 已通过 gpu06 复验）。

## 1. 里程碑对照

| 里程碑 | 状态 | 内容 |
|---|---|---|
| N1 | ✅ 完成 | `serve` 薄包装、全局 `--server`、`doctor --server`、token 边界 |
| N2 | ✅ 完成 | 远程客户端、context/chat、并发与错误映射 |
| N3 | ✅ 完成 | vLLM 后端与 auto 选择 |
| N4 | ✅ 完成 | 长音频窗口化、`align --language auto` |
| N5 | ✅ 完成 | stdin-list/skip-existing/retry、tsv/md、jobs/bench/config/models、schema/README |

## 2. 测试命令与结果

> 本机 lock 中的 numpy 2.2.6 下载受网络阻塞，P1 期间统一使用
> `UV_HTTP_TIMEOUT=300 uv run --no-sync ...`；这不是用户 README 命令。

```bash
uv run ruff check qasr_cli tests        # All checks passed
UV_HTTP_TIMEOUT=300 uv run --no-sync pytest -q   # 323 passed, 6 skipped
```

集成测试（需要本机模型）使用：

```bash
RUN_INTEGRATION=1 QASR_TEST_MODEL_DIR=... pytest -m integration
```

## 3. GPU 实测（gpu06，由协调者执行；本机不操作服务器）

验收基线：`8c7fee3`（N5 无行为变化）。

- **1h timestamps**：`libri_1h.flac` 3599.84s，vLLM 0.6B + ForcedAligner，
  21 个窗口，rc=0，wall=159.19s，649 段；时间单调、无重叠/倒退，
  warnings=[], 末段 `end=3599.843`，stdout 恰好一行 JSON。
- **1h align**：同一音频，参考文本取该次 canonical 输出，transformers 0.6B
  + aligner，rc=0，wall=227.81s，coverage=0.9129，7682 个词级 item；
  首词 1.04s、末词 `end=3599.843`；顶层 `language=English`、
  `language_source=explicit`。
- **align --language auto**（200s LibriVox，拉丁脚本）：LID probe 识别
  English，`language_source=lid_probe`，coverage=1.0，466 items，rc=0。
- **200s timestamps 修复复验**：rc=0，17 段单调，首段覆盖
  "This is a LibriVox recording..."，`chunk_reference_match_low` 消失。
- **chunk resume**：复制 chunk 到新 out-dir 后 `--resume`，chunk mtime 全未变
  （确认复用），wall=53.59s（仅模型加载 + canonical ASR），输出 17 段一致。
- **远程 1h**：`--server` 且未传 `--model`，14 个 270s 客户端切块，rc=0，
  wall=52.17s，`model=qwen3-asr-1.7b`，47,838 字符，stdout 单行 JSON。
- **远程能力边界**：`--timestamps` → rc=4 "does not return timestamps"；
  `align --server` → rc=4；未知 model → rc=4 "server does not serve model"。
- **vLLM stdout 纯净**：修复后 `--json` 输出恰好 1 行且 `json.loads` 成功，
  EngineCore 日志只在 stderr。

## 4. 与 DESIGN / 任务书的偏差

### 4.1 远程 `--batch-size`

DESIGN §3.3 写 remote 默认 batch-size 8，P1 任务书又规定远程 `--batch-size`
不用于传输并发。实现口径：远程有效 batch-size 恒为 1，显式 `>1` 时 warning
并忽略；`--jobs` 是唯一的客户端并发旋钮。默认 8 仅用于 vLLM。

### 4.2 远程 chat 语言 prefill（2026-11-18 gpu06 实测，任务书已更新）

模型自带 chat template 只渲染 system + 音频占位，忽略 assistant 消息；
`continue_final_message` 会 400。实现**不使用 assistant prefill**：context 作为
system；显式 `--language` 时给出 `language_not_enforced_on_chat` warning，
语言以服务端输出解析为准。

### 4.3 transcription 响应语言前缀（gpu06 实测）

`/v1/audio/transcriptions` 返回的 `text` 带 `language X<asr_text>` 前缀，
必须本地解析。`language` 表单字段只做服务端校验（非法值 400），不改变
Qwen3-ASR 输出；显式语言与解析结果不一致时 warning `server_ignored_language`，
不伪造成用户值。

### 4.4 404/405 降级

保留 `/v1/audio/transcriptions` → chat completions 的兼容降级路径；gpu06
实测 transcription 端点存在，正常流程不会触发。

### 4.5 长音频口径

- aligner 单窗硬上限 180s；`align` 自 P1 起 >`--window-seconds` 自动切窗。
- 纯文本 chunk 默认 270s（10–1200）。
- 远程必须客户端切块：默认 270s，`--max-chunk-seconds > 600` 降为 600 +
  warning（vLLM 默认 25MB 单文件上限，1h FLAC 约 47MB 会被 400）。

### 4.6 凭据与重定向安全

`QASR_SERVE_TOKEN` 只读环境变量；URL userinfo 与 `token/key/secret/password`
类 query 参数在 doctor/错误输出中脱敏。所有远程请求使用自定义
`SameOriginRedirectHandler`：同 origin 最多跟随 3 次并保留 Authorization，
跨 origin（scheme/host/port 任一变化）直接拒绝，绝不把凭据发给新 origin。

### 4.7 180s/270s 任务书澄清

`--timestamps` + `--max-chunk-seconds > 180` → 退出 4；纯文本使用 270s 只用于
进度/内存/checkpoint，不作为 aligner 窗口。

### 4.8 远程 served model id（gpu06 实测）

服务端常用 `--served-model-name` 重命名模型，直接把 qasr alias 映射成 HF repo
会 404。远程始终先 `GET /v1/models` 并按以下顺序匹配：精确 served id →
请求值的 repo/alias/basename → 仅一个 served model 且请求是已知 qasr alias
（使用该 id 并 warning）；多模型无法匹配 → 退出 4 并列出可选 id。进程内缓存
解析结果；`prepare()` 在并发 worker 启动前完成解析。payload 的 `model.id`
记录解析后的 served id（未显式配置 model 时由 `/v1/models` 首个模型决定并 warning）。

### 4.9 远程 408/429 错误码

DESIGN 的粗粒度规则是普通 4xx→4；任务书澄清 408/429 属于可重试传输错误。
实现：408/429 与 5xx/超时/连接错误 → `RuntimeFailure`（退出 3，retryable）；
其他 4xx（含 401/403、400、422）→ `InputError`（退出 4）。重试总次数为
`1 + retries`；语言 400 去掉 language 仅额外一次，不占 retries。

### 4.10 `language_source: server` 与 chat 语言

远程结果新增 `language_source="server"`（同步更新 `schemas/transcribe-item.json`
enum）。transcription 与 chat 响应都本地解析 `language X<asr_text>` 前缀。
chat 路径无法强制语言：只要实际使用 chat 且用户显式给了 `--language`（包括
transcription 404/405 降级后）就产生 `language_not_enforced_on_chat`；
解析语言与请求语言不一致（且可映射 ISO）时产生 `server_ignored_language`。

### 4.11 vLLM 后端与 auto 选择

- `VLLMBackend` 通过 `Qwen3ASRModel.LLM(...)` 懒加载，参数含 `dtype`
  （fp32/fp16/bf16 → vLLM 字符串）、`gpu_memory_utilization`、`max_new_tokens`、
  `max_inference_batch_size`。vLLM 0.14 的 `EngineArgs` 不接受 `device`，因此
  不传；显式 `--device cuda:N`（N≥0，含 `cuda:0`）在 import torch/vllm **之前**写入
  `CUDA_VISIBLE_DEVICES=N`（覆盖已有值，显式优先），并把内部请求设备改写为
  `cuda:0`；`auto`/`cuda` 不改环境变量。`--timestamps` 时把 forced aligner 一并传入，时间戳
  仍由 HF aligner 产生；`align` 命令同样复用 HF aligner。
- 显式 `--backend vllm` 在非 Linux/无 CUDA/无 vllm 时可导入性检查失败 → 退出 3，
  hint 指向 `qasr-cli[vllm]`；不会静默回退。
- `auto`：`--server` → remote；Linux+CUDA 且 vllm 可导入 → vllm；否则
  transformers。Linux+CUDA 但 vllm 导入失败时回退 transformers，并在 JSON
  顶层与结果项写 `backend_fallback: true` + warning。
- vLLM 默认 `--batch-size 8`（其余后端 1）；微批失败仍逐条重试定位。
- `--gpu-memory-utilization` 仅 vLLM 可用，其他后端 → 退出 2；取值需在 (0, 1]。
- `doctor` 增加 `backend` check，报告 selected 与 auto resolved（仅读安装元数据，
  不 import torch/vllm）。

### 4.12 时间戳长音频的 canonical ASR 策略

时间戳长音频采用「canonical 全文 ASR + 窗口 ASR 锚点 + 窗口 align」，而不是
逐窗 ASR 文本拼接：全文 ASR 保证 `result.text` 与短音频一致、避免跨窗重复；
`attribute_reference` 把 canonical 文本单调归属到窗口，再逐窗对齐并裁剪核心区。
chunk checkpoint 覆盖窗口锚点与对齐结果，但 canonical 全文 ASR 每次 `--resume`
仍会重跑（见已知限制/后续优化）。窗口归属按候选切片（`len(window)*3+200`）
取首末匹配块链，允许窗口内少量插入/替换，ratio 为窗口匹配字符占比。

### 4.13 P1 最终口径（评审确认项）

- stdout 纯净：vLLM 日志（含 EngineCore 退出日志）只进 stderr；`--json`/`--jsonl`
  在 stdout 只有规定行数。
- CUDA：显式 `--device cuda:N`（含 `cuda:0`）在任何 torch 探测前覆盖
  `CUDA_VISIBLE_DEVICES` 并改写内部设备为 `cuda:0`；auto 仅在探测可导入时选
  vLLM。
- 远程默认模型：未显式配置 model 时不发送本地默认 alias，由 `/v1/models`
  取首个并 warning；解析结果回填 payload/state/chunk key。
- fail-fast：退出码取首个错误（3/4/5），未尝试项标 skipped；只有非 fail-fast
  且已有完成项时部分失败才是 6。
- 凭据：`QASR_SERVE_TOKEN` 仅环境变量；错误 body/URL/reason 统一
  `sanitize_text` 后进入 message/state。

## 5. 已知限制

- 远程不返回时间戳，也不提供对齐端点（capability error，退出 4，且在发请求前拒绝）。
- 远程 `--jobs` 默认 4、范围 1–32；state 的 running/done 追加已串行化，fail-fast
  取消排队任务并由收尾阶段补 skipped，不留下 running 记录。
- 远程默认客户端切块 270s（上限 600s）；服务端 25MB 限制不再依赖用户手动切分。
- 时间戳长音频的 canonical 全文 ASR 不进入 chunk checkpoint，`--resume` 会重跑。
- vLLM 真机验收（gpu06）由协调者安排；本地只做假模块单测，未操作服务器。
- `forced_aligner_kwargs` 中的 `dtype=torch.dtype` 在 qwen-asr 0.0.6 真机
  timestamps 路径仍待 gpu06 复核（走 HF AutoModel 参数，理论上接受）。
- 远程 RTF 含网络时间，仅作记录，不作门禁。
- `QASR_SERVE_TOKEN` 只从环境变量读取；不写入配置、state、jobs 或日志。

### 4.14 `align` JSON 顶层语言字段

`align` 的 `language` 与 `language_source`（`explicit` / `script` / `lid_probe`）
位于 **JSON 顶层**，不在 `result` 内；`result` 只含 `items`/`coverage`/`warnings`。
`transcribe` 的语言字段仍在 `result` 内。两者 schema 均向后兼容。

### 4.15 配置文件未知 key 的容忍策略

配置加载器忽略当前版本不认识的顶层 key（不再让每个命令以 rc2 失败），
`config set` 以原始 JSON 对象回写、保留这些未知 key，只校验本次写入的 key
与值。这样手写/新版本配置不会把旧版本命令全部卡死；未知 key 不参与合并，
也不会出现在 `config show` 的有效配置里。`config set --project` 在写入前
检查目标文件及现有父目录（至少 `.qasr`）是否为符号链接或解析到项目根之外。

## 6. 审查入口

- 假 OpenAI 服务与 serve/doctor 契约测试：`tests/contract/test_serve_doctor.py`
- 远程客户端测试：`tests/contract/test_remote_client.py`（multipart/chat/降级/重试/并发/state/schema/重定向）
- vLLM 假模块测试：`tests/unit/test_vllm_backend.py`；核心批量/fallback：`tests/unit/test_transcribe.py`
- 长音频：`tests/unit/test_long_audio.py`、`tests/unit/test_long_transcribe.py`
- 批量/tsv/md/stdin-list：`tests/contract/test_p1_batch_contract.py`
- jobs/bench/config/models：`tests/contract/test_p1_ops_contract.py`
- 完整 schema：`schemas/bench.json`、`schemas/jobs-list.json`、`schemas/jobs-show.json`、`schemas/models-show.json`、`schemas/config-get.json`、`schemas/config-set.json`
- 全量：`uv run pytest -q`；GPU：`RUN_INTEGRATION=1 ... pytest -m integration`
