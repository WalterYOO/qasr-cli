# qasr-cli P1 实现任务书

> 依据：`DESIGN.md` v0.3 §3.7–3.9、§5、§7、§8、§12.3。P0 已完成并独立审查 PASS，tag `v0.1.0`。
> P1 只做**批量、部署与长音频**；`stream`、`--mic`、`completion`、`finetune`、多 GPU、视频抽轨、watch、Windows 属 P2，本阶段一律不实现。
> 每一波（N1–N5）结束都要能独立验收；N5 完成后交给独立 Codex 审查/测试。

---

## 0. P1 目标

在 P0 单机闭环之上，补上三条能力线：

```bash
# 1) 部署：GPU 机器常驻服务，Mac/其他机器远程调用
qasr serve 1.7b --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.8
qasr --server http://gpu-box:8000 transcribe long.flac --language zh --jobs 4
qasr doctor --server http://gpu-box:8000

# 2) 运行时：Linux/CUDA 上本地 vLLM 离线批量
qasr --backend vllm transcribe ./audio/ --recursive --batch-size 8 --out-dir out

# 3) 长音频：自管切块 + 时间戳/对齐窗口化 + 单调合并
qasr transcribe 1h.flac --timestamps --language zh --max-chunk-seconds 180 -f srt -o out
qasr align 1h.flac --text-file transcript.txt --language zh -f srt -o out

# 4) 运维：高级批量与状态查询
qasr transcribe --stdin-list --skip-existing --retry --out-dir out --resume < paths.txt
qasr jobs list --json ; qasr jobs show <id> --json
qasr bench sample.wav --iterations 3 --json
qasr models show 1.7b --json ; qasr models rm 0.6b --yes
qasr config get backend ; qasr config set backend vllm --project
qasr align 1h.flac --text-file t.txt --language auto
```

**P1 不包含**：`stream`、麦克风、`completion`、`finetune lint-jsonl`、多 GPU 分片、视频抽轨、目录 watch、Windows；远程时间戳/对齐/流式；`QASR_SERVE_TOKEN` 写入配置文件。

---

## 1. 工程约束（在 P0 基础上追加）

- 继续用 **argparse**；继续 **base 不 import torch/qwen_asr/vllm**。
- 新增依赖：
  - `[vllm]` extra：`qwen-asr[vllm]==0.0.6`（DESIGN §9.1）。
  - 远程客户端只用标准库 `urllib`/`http.client`/`ssl`/`json`/`base64`/`mimetypes`；**不引入 requests/httpx**。
  - `long_audio.py` 只用标准库 + numpy；不得 import torch/qwen_asr/soundfile（音频切片由调用方完成，纯函数接收时长/采样点/能量包络）。
- 继续所有 `--json` 输出有 `schema_version: 1`；新增/变更 schema 放 `schemas/` 并用 jsonschema 做契约测试。
- 不修改 `DESIGN.md`。若实现中发现设计矛盾，先记录到 `P1-REPORT.md` 的“偏差”节并立即报告协调者。
- 不实现 P2；不为 P2 预留半成品命令。

### 1.1 长音频窗口口径（P0/P1 边界澄清）

DESIGN §12.3 写“180s/270s 窗口”，§8 又要求对齐器单窗 ≤180s。P1 采用以下可执行口径，并在 `P1-REPORT.md` 记录：

- **带时间戳的转写 / `align`**：送入 aligner 的**单个音频窗口硬上限 180s**。默认 `--window-seconds 180`，窗口含 3s 左重叠上下文，因此核心归属区约 177s。
- **纯文本转写**（不要时间戳）：`--max-chunk-seconds` 默认 270s，仅用于进度、内存上限与分块 checkpoint；ASR 本身仍允许单块更长。显式 `--max-chunk-seconds` 可为 10–1200。
- `--timestamps` 与 `--max-chunk-seconds > 180` 同时出现 → 退出 4，提示 aligner 上限；不要静默截断。
- 270s 不是 aligner 窗口；任何进入 `Qwen3ForcedAligner.align()` 的音频必须 ≤180s（含重叠）。

---

## 2. 交付结构（P1 增量）

```
qasr_cli/
  serve.py            # serve 命令实现：解析/校验/execvp qwen-asr-serve
  remote.py           # 远程 HTTP 客户端（纯标准库）+ RemoteBackend
  vllm_backend.py     # VLLMBackend（懒 import qwen_asr/torch）
  long_audio.py       # 纯函数：窗口规划、文本归属、单调合并、覆盖率
  jobs_index.py       # QASR_HOME/jobs 索引：登记/列出/读取作业
  bench.py            # bench 编排与统计
  commands/serve.py
  commands/jobs.py
  commands/bench.py
  commands/models.py  # 增加 show/rm
  commands/config.py  # 增加 get/set
  commands/transcribe.py  # --stdin-list/--skip-existing/--retry/tsv/md/长音频
  commands/align.py       # --window-seconds/--window-overlap/auto LID 探测
schemas/
  jobs-list.json jobs-show.json bench.json models-show.json config-get.json
tests/
  unit/ contract/ integration/
```

**硬约束**：`serve.py`、`remote.py`、`long_audio.py`、`jobs_index.py`、`bench.py`、`config.py`、`jobs.py`、`cli.py` 不得在模块顶层 import `torch`/`qwen_asr`/`vllm`。

---

## 3. 里程碑与任务

### N1 — serve 薄包装与 doctor --server

**T1.1 `qasr serve`**
- 语法：`qasr serve [MODEL] [--host H] [--port P] [--gpu-memory-utilization F] [PASSTHROUGH...]`。
- `MODEL` 走 P0 别名解析：`1.7b`/`0.6b` 优先本地完整目录；未下载则用 repo id `Qwen/Qwen3-ASR-1.7B` / `0.6B`。
- 已知 flag 归一化后与未知参数一起透传给 `qwen-asr-serve`；**未知参数不报错**，原样放行（`--tensor-parallel-size` 等）。
- `--dry-run`：打印将执行的 argv（JSON 模式含 `command/argv/model/resolved`），不 import vllm、不 exec、不启动服务。
- 找不到 `qwen-asr-serve` → 退出 3，hint：`uv tool install --python 3.12 "qasr-cli[vllm]"` 或 `pip install "qwen-asr[vllm]==0.0.6"`。
- 实际启动用 `os.execvp`，qasr 不自管子进程、不接管日志；若 exec 失败 → 退出 3。
- `pyproject.toml` 增加 `vllm = ["qwen-asr[vllm]==0.0.6"]`；`qasr --help` 与 base 安装仍不 import vllm。

**T1.2 远程配置与 `doctor --server`**
- 全局 `--server URL`，优先级：CLI > `QASR_SERVE_URL` > 配置 `server`；隐含 `backend=remote` 且与显式 `--backend transformers|vllm` 冲突时退出 2。
- `QASR_SERVE_TOKEN` **只从环境变量读**；不写入 config、不打印、不落到 manifest/JSON/jobs/state/日志；doctor 只报告“token 已设置/未设置”。
- `--server` URL 中的 userinfo（形如 `http://user:***@host`）视为敏感：错误/doctor/`jobs` 展示时脱敏；重定向不把 `Authorization` 带到不同 origin（跨 origin 直接报错或拒绝跟随）。
- `doctor --server URL` 新增 check：URL 合法性、TCP/TLS 可达、`GET /v1/models`（带 Bearer）、HTTP 状态、延迟、模型 id、auth 是否被拒；不要求服务端有 GPU。
- `doctor --json` schema 保持兼容：新增 check 项，不改变顶层字段；`--strict` 语义不变。
- 远程不可达/401/5xx：doctor 记 `fail`/`warn` 并给 hint；默认仍退出 0，`--strict` 退出 3。

**N1 验收**
- 无 vllm 环境下 `qasr serve --dry-run --json` 成功、无 vllm import；缺 binary 的模拟环境退出 3。
- 本地启动一个假 OpenAI 服务（测试内起 `http.server`）→ `doctor --server` 能报 models 与延迟；401/超时分类正确。
- `QASR_SERVE_TOKEN` 不出现在 `config show`、`doctor --json`、argv dry-run 输出中。

---

### N2 — 远程客户端与远程并发

**T2.1 传输契约**
- 无 context：`POST {server}/v1/audio/transcriptions`，multipart 字段：
  - `file`（文件名、MIME）、`model`（服务端模型 id；默认取 `/v1/models` 首个，可 `--model` 覆盖）、`language`（ISO 码，如 `zh`；`auto` 时不传）。
  - **2026-11-18 gpu06 实测**：vLLM 0.14.0 的 transcription 响应是 `{"text":"language Chinese<asr_text>...","usage":{...}}`，**必须解析前缀**；`language` 字段会校验非法值（`xx`→400 `Unsupported language...`），但合法值对 Qwen3-ASR 实际输出基本无影响（en/zh/yue 都返回 Chinese）。因此语言以服务端输出为准；显式 language 与输出不一致 → warning `server_ignored_language`，不得伪造成用户值。
- 有 context：`POST {server}/v1/chat/completions`：
  - `messages[0]` 为 `system`，内容为 context 原文（多段 context 以 P0 相同规则合并）；
  - 用户消息 content 含 `{"type":"audio_url","audio_url":{"url":"data:audio/<fmt>;base64,..."}}`；
  - **不要使用 assistant prefill / `continue_final_message`**：模型自带 chat template 只渲染 system + 音频占位，忽略 assistant 消息，prefill 会 400。chat 端点无法可靠强制语言；显式 `--language` 时给出 warning `language_not_enforced_on_chat`，语言取解析结果。
  - 响应 `choices[0].message.content` 用**本地纯函数**解析（不 import qwen_asr）：去掉 `<asr_text>`、`language X` 前缀，返回 `(language, text)`。
- `parse_asr_output` 的本地实现放 `remote.py`，行为对齐 qwen-asr：空输出 → `("", "")`；多语言标签保留原样（如 `Chinese,English`）；两个端点的响应都要解析。
- 如果 `/v1/audio/transcriptions` 返回 404/405（服务端未暴露该端点），回退到 chat completions（无 context 版本）一次并 warning；同一进程缓存该降级决定。当前 gpu06 端点存在，正常不会触发。
- 远程 capability：`--timestamps`、`--granularity`、`align --server`、`stream`（P2）→ 退出 4，message 指明“服务端不返回时间戳/无对齐端点”。
- 音频读取：本地路径直接 multipart/读 bytes；URL 输入先走 P0 缓存下载；`audio_base64` manifest 输入直接解码。
- **远程长音频必须客户端切块**（2026-11-18 gpu06 实测）：vLLM transcription 默认 `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=25`，1h FLAC（47MB）直接 400 `Maximum file size exceeded`。因此 remote 转写默认 `--max-chunk-seconds 270`（16k mono WAV 约 8.6MB，安全），复用 N4 的窗口规划/临时 WAV/文本拼接；`--max-chunk-seconds > 600` 在 remote 下降到 600 并 warning（避免超 25MB）。文档说明可调服务端环境变量，但 qasr 不依赖它。
- token：`Authorization: Bearer $QASR_SERVE_TOKEN`；日志与错误 details 不包含 header。

**T2.2 超时、重试、错误映射**
- `--timeout 300`（秒，请求级）、`--retries 2`：**总尝试次数 = 1 + retries（默认 3）**；仅 retryable：连接错误、超时、5xx、429、408。
- 语言被服务端拒绝（4xx 且 message 含 language/lang/unsupported）→ 去掉 language **最多重试一次** + warning，不计入 `--retries`；若再次失败直接报错，不得循环。
- 映射：400/401/403/404/405/413/422 等普通 4xx → 退出 4（不重试）；408/429/5xx/超时/连接失败 → 退出 3（按 retries 重试）；3xx 不自动跨 origin 重试（见 token 安全）；TLS/DNS/连接重置 → 3。
- 批量部分失败 → 退出 6；全部失败取**输入顺序首个错误**（不是完成顺序）。`--fail-fast` 时取消未开始任务，已开始任务正常收尾或标记失败，不得留下 `running`。
- 429 若带 `Retry-After` 且 ≤ 10s，遵守；否则指数退避（0.5s 起，封顶 5s），总延迟受 `--timeout` 约束。
- 错误 envelope 的 `error.retryable` 与 `details.http_status` 正确；不得把服务端 body 原样无限打印（截断到合理长度，UTF-8 安全，避免 token 泄漏）。

**T2.3 远程并发 `--jobs`**
- `--server` 下 `--jobs` 默认 4，范围 1–32；线程池，结果按输入顺序输出/写盘。
- 本地模式 `--jobs != 1` → warning 并强制 1（P0 行为保持）；`--backend remote` 等同 `--server`。
- `--fail-fast`：取消尚未开始的任务，已完成项正常落盘，退出码取输入顺序首个错误；取消项不得遗留 `running`。
- 远程批量的 state/resume/lock 与 P0 一致；`state.jsonl` 的 append 必须串行化（单写锁/队列），并发完成顺序不得造成行交错或丢记录。
- 远程 `--batch-size` 不用于传输并发，避免双重并发（显式 >1 时 warning 并忽略）；有效传输并发只由 `--jobs` 控制。

**N2 验收**
- 测试内假服务覆盖：multipart 字段、chat+context、语言拒绝重试、404 降级、401→4、500→3（重试次数正确）、超时→3、部分失败→6。
- 假服务记录并发度：`--jobs 4` 时最大同时请求 ≤4 且 >1；`--jobs 1` 严格串行。
- 远程 1 小时音频实际链路在 N5 GPU 验收中完成（见 §6）。

---

### N3 — vLLM 离线后端与 auto 选择

**T3.1 `VLLMBackend`**
- `runtime.py` 懒加载 `qwen_asr.Qwen3ASRModel.LLM(...)`；参数：模型、`device`（CUDA）、`dtype`、`max_inference_batch_size`、`max_new_tokens`、`gpu_memory_utilization`（新增 CLI，仅 vllm）。
- 支持：批量转写、context、LID、`--timestamps`（aligner 仍用 HF 路径，窗口 ≤180s）。
- 不支持：MPS、CPU-only、远程语义；显式 `--backend vllm` 在这些平台 → 退出 3，附安装/平台说明。
- `--gpu-memory-utilization` 仅在 vllm 生效；与 transformers/remote 同用 → 退出 2（不静默忽略）。
- 批处理沿用 P0 `transcribe.py` 微批与错误隔离；vLLM 批量默认 `--batch-size 8`，失败组逐条重试定位。
- `--max-new-tokens auto` 仍按本次最大时长计算。

**T3.2 `--backend auto` 选择**
- 顺序：`--server` → remote；`stream`（P2）→ 必须 vllm；Linux+CUDA+`import vllm` 成功 → vllm；macOS → transformers+MPS；其他 → transformers+CPU。
- auto 下 vllm 可用性探测失败可回退 transformers，但 JSON 顶层/结果项写 `"backend_fallback": true` + warning；显式 `--backend vllm` 不静默回退。
- `doctor` 报告 vllm 可导入性/版本与当前 auto 解析结果。

**N3 验收**
- Mac 上（无 vllm）：`--backend auto` 仍 transformers；`--backend vllm` 退出 3 且不 import 崩溃。
- 假 `qwen_asr` 模块单测：构造函数参数、批量、失败隔离、timestamps 路径。
- gpu06 实机：`--backend vllm transcribe` 短音频/批量成功，结果与 transformers 文本一致（允许标点差异），RTF 仅记录不作门禁。

---

### N4 — 长音频窗口化

**T4.1 `long_audio.py` 纯函数**
- `plan_windows(total_samples, sr, window_s=180, overlap_s=3, search_s=5) -> list[Window]`
  - `Window`：`index, audio_start_sample, audio_end_sample, core_start_sample, core_end_sample`；采样点是权威值。
  - 公式：首个 `core_start=0`；目标核心长 `window_s-overlap_s`；`audio_start=max(0, core_start-overlap_s)`，`audio_end=core_end`；非末尾核心长在 `[window_s-overlap_s-2, window_s-overlap_s]`，末尾可短；`core` 连续无缝、末窗 `core_end=total`；每个 `audio_end-audio_start <= window_s`（默认即 ≤180s）。
  - 可选 `energy` 包络参数：在目标切点 ±`search_s` 内找能量最低帧；切点必须仍满足核心长区间与音频窗上限；找不到/未提供 → 固定切点。切点不得让任一核心区为空。
  - 校验 `window_s <= 180`、`0 <= overlap_s < window_s`（允许 0）、`search_s >= 0`；非法 → UsageError（退出 2）。
- `attribute_reference(ref_text, window_texts) -> list[TextSpan]`
  - 用 NFKC + 去空白 + 去标点的规范化文本做**单调** `SequenceMatcher` 匹配，把全局参考字符区间分配给各窗口；不得跨窗口重复归属。
  - 窗口匹配率低于 0.5 时记录 warning，该窗口仍可交给 aligner；全局覆盖率由既有 `align` 指标计算。
- `merge_items(items, *, duration_s) -> list[Item]`
  - 全局偏移后按 start 排序；强制 `start = max(start, prev.end)`；若 `end <= start` 则 `end = min(start + 1ms, duration_s)`（duration 不足时丢弃）；`end <= duration`；空 item 丢弃。规则固定并测。
- `coverage(ref_text, items) -> float` 复用 P0 规范化/匹配语义。
- 所有函数无 I/O、无模型、无 torch。

**T4.2 `transcribe` 自管切块**
- `--max-chunk-seconds N`（10–1200）：
  - 无时间戳默认 270；有时间戳默认 180，且 >180 → 退出 4；值不在 10–1200 → 退出 2。
  - 用 `long_audio.plan_windows` 规划；按窗口读取音频（soundfile 块读，mono）、转写；时间戳窗口结果按核心区裁剪并加 `audio_start` 偏移。
  - 纯文本窗口结果拼接时处理边界：相邻窗口重叠区只保留核心区对应文本；无法精确切词时保留前窗结果并在 warning 中说明（不得重复整段）。
  - 窗口级 checkpoint：`<out-dir>/.qasr/chunks/<job_key>/<index>.json`（原子写）；`--resume` 复用完整 chunk；`--overwrite` 忽略/清理；最终输出仍走 P0 原子写。
- **remote 同样走客户端切块**：默认 270s、上限 600s（见 T2.1 的 25MB 限制）；每窗写临时 16k mono WAV 上传，按输入顺序拼接文本；chunk checkpoint 与本地一致。
- 单窗口（音频 ≤ `--window-seconds`）时行为与 P0 完全一致（不得回归）。
- `--timestamps` 的窗口结果 `segments` 必须全局绝对时间；`timestamp_source` 仍为 `forced_aligner`。
- chunk checkpoint key 必须包含：输入音频内容 SHA、窗口参数（window/overlap/max_chunk）、模型指纹、aligner 指纹、language、context、qasr 版本；任一变化不得复用旧 chunk。

**T4.3 `align` 窗口化**
- `align` 新增 `--window-seconds 180`（10–180）、`--window-overlap 3.0`（0–10，且 < window）。
- 音频 ≤ window 时走 P0 单窗逻辑（零行为变化）。
- 超长时：
  1. 规划窗口；
  2. 每窗对窗口音频做一次 ASR 得到局部文本（不要求时间戳；模型复用）；
  3. `attribute_reference` 把参考文本归属到窗口；
  4. 对每窗调用 aligner（音频窗口 + 归属文本），得到局部 items；
  5. 只保留 core 内 item，加全局偏移；
  6. `merge_items` 单调化并 clamp 到总时长；
  7. 覆盖率按全局参考文本与最终 items 计算，低于 `--coverage-min` → 退出 5。
- 失败窗口：记录 warning/错误，继续其他窗口；全部失败按 P0 错误码；部分窗口失败导致覆盖率不达标 → 5。
- **P1 起 `align` 默认窗口化**：音频 > `--window-seconds`（默认 180）自动切窗，不再像 P0 一样直接拒绝；≤180s 时走原单窗路径。181s 必须成功而不是退出 4。`--window-seconds` 上限 180。
- 窗口参数非法（如 0/负/`window>180`/`overlap>=window`）→ 退出 2。
- `--dry-run` 输出窗口计划（数量、core/audio 范围），不加载模型。

**T4.4 `align --language auto`**
- 脚本判定保持 P0：Han→zh、Kana→ja、Hangul→ko、Cyrillic→ru。
- 拉丁/混合脚本：不直接报错，先对音频前 30s 跑一次 ASR LID；检测结果属于 11 对齐语言 → 使用并 warning（`language_source: "lid_probe"`）；否则退出 4，要求显式语言。
- 探测失败/空音频 → 退出 4，hint 列 11 语言；不得静默选 en。
- `align` JSON 增加可选 `language_source: "explicit"|"script"|"lid_probe"`（P0 字段不删；schema 向后兼容）。
- `align --server` 仍 capability error（N2）。

**N4 验收**
- 纯函数单测：窗口边界/采样点、末尾短窗、超 180s 拒绝、固定切点、能量切点、单调归属、重叠合并、clamp、空文本。
- 合成 1h 音频（16k 单声道，可含静音段）：`--dry-run` 窗口数正确；`--timestamps --language zh` 全流程时间单调、无跨窗重复、覆盖率/段数合理。
- 1h `align`（真实参考文本或重复短文本拼接）时间单调、覆盖率 ≥0.5、SRT/VTT 合法。
- kill -9 后 `--resume` 跳过已完成 chunk，只跑剩余窗口（用可控慢速 fake backend 验证）。

---

### N5 — 高级批量、运维与完整验收

**T5.1 输入与重跑**
- `--stdin-list`：**无参数**，从 stdin 读路径，每行一个；空行跳过；`-` 不再是路径；与 `--manifest` 互斥；与位置参数可并存（stdin 在前/位置参数在后，稳定排序去重）。需要从文件读时用 shell 重定向 `< paths.txt`。
- `--skip-existing`：目标输出全部存在且非空 → `skipped`，不加载模型；与 `--resume` 可同用（先 skip-existing 再 job_key 判定）；JSON 计数不并入 ok（沿用 P0 修正）。
- `--retry`：对 `error.retryable == true` 的失败项重跑一次；仅一次，记录 `attempts`；与 `--fail-fast` 互斥（用法错误）。
- `--continue-on-error` 默认不变；部分失败退出 6。

**T5.2 输出格式 `tsv` / `md`**
- `tsv`：表头 `id	start	end	text	language`；无时间戳时 start/end 为空；每段一行（有 segments）；字段内 tab/换行转义为空格；UTF-8。
- `md`：每个 item 一个 `## <id>` 节；有 segments 时列表 `- [start - end] text`，否则正文；批量时按输入顺序。
- 两种格式同样支持 stdout 单条与 `--out-dir` 批量；`--format tsv|md` 与其他文件格式一样，与 `--json/--jsonl` **无条件互斥**（退出 2）。
- 更新 `render.py`/输出扩展名映射与 README。

**T5.3 `jobs list` / `jobs show`**
- 批量启动时在 `QASR_HOME/jobs/<job_id>.json` 登记：`job_id, out_dir, created_at, updated_at, command, total`；job_id 用 out_dir 绝对路径 + 启动时间哈希。
- `jobs list [--json] [--limit N]`：按 updated_at 倒序；state 不存在 → `missing`。
- `jobs show JOB_ID|OUT_DIR [--json]`：读 `.qasr/state.jsonl` 汇总 `pending/running/done/failed/skipped`、首错、输出目录；未知 job → 退出 4。JOB_ID 必须是索引中存在的 id（不拼任意路径）；OUT_DIR 必须 resolve 后仍是索引登记的目录，拒绝 `..`/symlink 逃逸。
- `jobs list --limit`：1–10000，非法 → 2。
- 写入原子；并发安全（同一 out-dir 锁仍由 P0 lock 保证）。
- 新 schema：`jobs-list.json`、`jobs-show.json`。

**T5.4 `bench`**
- `qasr bench [INPUT...] [--duration 10] [--iterations 3] [--json]`；无输入时生成 `duration` 秒静音（明确标注 synthetic）。
- 输出：`backend, device, dtype, model, load_s, audio_s, elapsed_s, rtf, iterations, peak_memory_mb`（拿不到则 null）；不做准确率声明。
- 远程 bench 可用（上传合成音频），但 warning 说明网络计入 RTF；vLLM/transformers 本地均可。
- `--dry-run` 只输出计划；新 schema `bench.json`。

**T5.5 `models show` / `models rm`**
- `show`：别名解析结果、本地路径、repo id、来源、大小、文件数、verify 状态、指纹、`source.json` 摘要；`--json` schema `models-show.json`。
- `rm`：只允许删 `QASR_HOME/models/` 下的托管模型；解析 alias 时**先检查目标目录本身不是 symlink**，再校验 realpath 在 models 根内；拒绝 `..`、绝对路径、根目录本身、非托管目录；必须 `--yes` 或 TTY 确认；**非 TTY 且无 `--yes` → 退出 2，不删除**；`--dry-run` 打印将删路径；成功输出删除字节数。
- `pull` 后 `show` 的 verify 状态与 P0 一致。

**T5.6 `config get/set`**
- `config get KEY [--json]`：输出合并后的有效值（含来源可选择）；未知 key → 退出 2。
- `config set KEY VALUE [--project|--home] [--json]`：默认写 `QASR_HOME/config.json`；`--project` 从 cwd 向上找最近的 `.git`/`pyproject.toml` 目录作为项目根（找不到则 cwd），写 `<root>/.qasr/config.json`（不存在则创建）；原子写、保留未知键、枚举校验（device/dtype/backend/offline）。目标文件是 symlink → 退出 2，不写。
- 拒绝设置 `QASR_SERVE_TOKEN`/任何 token 类键（本就不是配置键）；`server` 可设置。
- `config show` 增加新键 `server`、`timeout`、`retries`、`window_seconds`、`window_overlap`、`max_chunk_seconds` 的来源展示。

**T5.7 文档与 schema**
- README 增加：serve/远程、vllm 安装、长音频窗口与 180s 口径、`--jobs` 并发、jobs/bench/config/models 新命令、远程能力边界与退出码、token 只走环境变量。
- 所有新 schema 加契约测试；`--help` 不 import torch/vllm。
- `P1-REPORT.md`：完成对照、测试命令与结果、GPU 实测、偏差与已知限制、审查入口。

**N5 验收**
- `--stdin-list` + `--skip-existing` + `--retry` 组合语义正确；kill -9 后 resume 只跑未完成项。
- tsv/md 输出稳定、字段转义正确；jobs list/show 能跨进程读取。
- `bench --json`、`models show --json`、`config get/set` 通过 schema 与幂等测试。
- README 中的命令逐条可复制运行（无 GPU 的用 `--dry-run`/假服务）。

---

## 4. 关键契约速查（P1 增量）

### 退出码（不变）

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | 参数/配置/manifest schema |
| 3 | 模型/运行时/下载/OOM/远程 5xx、超时 |
| 4 | 输入/音频/路径/能力不支持/远程 4xx |
| 5 | 对齐覆盖率/字幕质量门禁 |
| 6 | 批量部分失败 |
| 130 | SIGINT/SIGTERM |

### 远程能力矩阵

| 能力 | remote |
|---|---|
| 纯文本转写 | ✅ |
| context | ✅（仅 chat 端点） |
| `--jobs` | ✅（客户端并发） |
| 时间戳 / granularity | ❌ 退出 4 |
| align | ❌ 退出 4 |
| stream | ❌（P2） |
| token | 仅 `QASR_SERVE_TOKEN` 环境变量 |

### 长音频口径

- aligner 单次音频 ≤180s（含重叠）。
- 时间戳窗口默认 180s、重叠 3s；纯文本 chunk 默认 270s。
- 全局时间单调、`end >= start`、`end <= duration`。
- 覆盖率低于 `--coverage-min`（默认 0.5）→ 退出 5。

### JSON 兼容

- 既有字段不删不改名；新增字段可选。
- 新命令各自 schema；错误 envelope 不变。
- `backend_fallback` 只在 auto 回退时出现。

---

## 5. 测试要求

| 层 | 必须覆盖 |
|---|---|
| 单元 | serve argv 解析/dry-run；远程 URL/ISO/语言拒绝重试/parse_asr_output；错误映射；long_audio 窗口/归属/合并/覆盖率；tsv/md；jobs 索引；config get/set；models rm 路径安全；stdin-list/skip-existing/retry 语义 |
| 契约 | 新 schema；`--help` 不 import torch/vllm；token 不泄漏；远程 4xx/5xx/超时退出码；stdout 纯净；长音频 dry-run 结构 |
| 模拟 | 测试内 `http.server` 假 OpenAI 服务（multipart/chat/auth/延迟/错误注入）；`FakeBackend` 慢速窗口 + 崩溃注入 resume；假 `qwen_asr` 模块测 VLLMBackend 构造 |
| 集成（无 GPU） | 0.6B 短音频远程假服务端到端；0.6B 长音频窗口（可用 5–10 分钟合成/重复音频）；align 窗口化；`RUN_INTEGRATION=1` 才跑 |
| GPU（gpu06） | vLLM 短音频/批量；远程 1h；时间戳 1h；align 1h；kill -9 resume |

音频 fixture：单元/契约用代码生成的 WAV（静音/正弦/短语音拼接）；GPU 长音频用 16k mono FLAC 减小上传量；不提交模型或音频。

---

## 6. gpu06 测试计划（环境约束）

- **拓扑**：Mac → gpu06 经跳板机（`ssh -G gpu06` 显示 `ProxyJump hk-h20-05-remote`，非同一局域网）。任何从 Mac 向 gpu06 传文件前**必须先向用户确认**；优先在服务器侧用 git/下载/生成，避免大文件直传。
- **GPU 复用**：0/1/2/6/7 有空间。启动前 `nvidia-smi` 选显存最充裕的一张，`CUDA_VISIBLE_DEVICES=<n>` 隔离；不得影响他人任务；ASR 速度只记录，不作性能门禁。
- **长任务**：模型下载、vllm 安装、1h 测试一律 tmux（`tmux new-session -d -s qasr-p1 ...`），禁止 nohup/裸 `&`；日志写 `~/qasr-p1/logs/`。
- **环境**：先检查服务器已有 Python/uv/vllm/CUDA 与模型缓存；优先复用兼容环境；需要新建时按项目要求用 uv（不混用 conda/pip 到同一环境）。
- **端口映射**：服务端 `qasr serve ... --host 127.0.0.1 --port 18000`；Mac 侧 `ssh -N -L 18000:127.0.0.1:18000 gpu06` 放 tmux；客户端用 `http://127.0.0.1:18000`。若跳板机限制，改用 `ssh -J` 或服务器侧客户端验证，并记录。
- **GPU 验收步骤**：
  1. `doctor --server` 通过（token 可选）。
  2. vLLM 本地短音频/批量文本正确。
  3. Mac 经隧道用 `--server` 转写 1h FLAC（纯文本），`--jobs 4`。
  4. gpu06 本地 `--timestamps` 1h，检查单调/覆盖/无重复。
  5. gpu06 本地 `align` 1h，覆盖率 ≥0.5。
  6. 远程 401→4、500→3、超时→3（可在假服务或真实服务故障注入）。
  7. kill -9 后 resume 只跑未完成项。

---

## 7. 提交与报告

- 每个里程碑一次 commit，Conventional Commits；P1 完成打 `v0.2.0`（审查 PASS 后）。
- `P1-REPORT.md` 必须包含：
  1. N1–N5 完成/未完成对照；
  2. 测试命令与结果（含失败输出）；
  3. gpu06 环境、GPU 编号、端口映射、实测数据；
  4. 与 DESIGN 的偏差（尤其 180s/270s、chat 语言 prefill、transcription 404 降级）；
  5. 已知限制（远程无时间戳、网络 RTF、1h 内存等）；
  6. 审查入口（如何跑假服务/单测/GPU 测试、看哪些文件）。
- 独立审查者另写 `P1-REVIEW-PLAN.md` / `P1-REVIEW-REPORT.md`，不得修改实现。
