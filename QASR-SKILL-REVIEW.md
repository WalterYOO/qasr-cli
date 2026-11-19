# QASR-SKILL-REVIEW

**审核对象**：`QASR-SKILL-DESIGN.md`（SHA-256 `6d2a5daa…8166`，576 行）
**事实基线**：`qasr 0.3.0` / tag `b945bb4`，本机 `.venv/bin/qasr`、`schemas/*.json`、`qasr_cli/` 源码
**审核方式**：只读设计稿与 rubric；所有关键命令用 `--dry-run` / 安全探针复核，未跑推理、未下载模型
**修订记录**：v1.1 撤回原 MINOR 4（空元素在未加引号的 `$c` 展开中消失，实测 argv 为 `qasr --help`），并补入 Partner A 提出的 3 条新问题（MAJOR 13–15）。

# 审核结论

**FAIL**。硬门禁 G1/G2/G3 均被击穿：设计稿虚构了“字幕质量门禁”，把 exit 5 说成“产物通常已写出”；token 示例会进入 shell history；`align` 被描述为默认拒绝覆盖，实际会静默覆盖；`serve` 的“不会自动下载、`--offline` 是硬开关”也不成立。此外还有 3 条黄金路径会直接 exit 2，manifest 示例会直接 exit 4。

## 阻塞问题（BLOCKER）

1. **token 进入 argv/history，且与自身规则矛盾** — 设计 172、394-395。172 行路由表写 `QASR_SERVE_TOKEN=... qasr ...`；394 行说“永不写入 argv”，395 行又要求示例统一这么写。`qasr_cli/remote.py:35-38` 只从环境变量读取，前缀赋值会留在 history。`remote.py:71-73` 还会剥掉 URL userinfo，所以 `user:fake@host` 也不是凭据通道。修复：删除所有 `VAR=... command` 形式，只写“用户在自己的交互 shell 中 export/read -s，agent 不读取、不打印”；删掉 `user:fake@host` 占位。

2. **长任务建议违反用户硬规则** — 设计 416 写“建议 `tmux`/`nohup`”。用户规则只允许 tmux，禁止 nohup/裸 `&`。修复：删 nohup，只写 `tmux new-session -d` / `tmux attach`，并明确禁止 `&` 与 nohup。

3. **exit 5 语义错误 + 虚构“字幕质量”门禁** — 设计 435、445。全仓只有两处抛 `QualityError`：`align.py:392-398`（coverage）和 `stream.py:365-371`（max-stream-seconds）；`transcribe` 没有 QualityError，字幕/断句也没有质量门禁。`align.py:415` 才调用 `_write_outputs`，coverage 检查在写文件之前；`stream` 根本不写文件。所以“5 → 字幕质量 / 产物可能已写出 / 先检查 `.outputs[]`”全错。修复：5 只写“align coverage 或 stream 运行上限，通常无产物；先读 error envelope，不要假设 `.outputs[]` 存在”。

4. **`align -o` 默认不拒绝覆盖，会静默毁旧文件** — 设计 106 声称 qasr 默认拒绝覆盖，410 行表格说 `align -o` 同 transcribe、需 `--overwrite`。证据：`qasr_cli/align.py:450-489` 的 `_write_outputs` 没有任何 `target.exists()` 检查，直接 `atomic_write_bytes`（`jobs.py:63-77` 用 `os.replace`）；`align --help` 也没有 `--overwrite`。只有 `transcribe.py:1484` 有存在性检查。修复：把 align 标为“目标存在即覆盖”，要求先 `--dry-run` 看 `plan.outputs`、取得用户确认或换目录；不要写不存在的 `--overwrite`。

5. **`serve` 会未经确认下载模型，`--offline` 也拦不住** — 设计 386、406 声称“qasr 不会自动下载 / 模型缺失 exit 3 / `--offline` 是硬开关”。证据：`qasr_cli/serve.py:57-72`（`resolve_serve_model`）对不完整本地模型主动回退到 repo id；实测 `qasr --json --offline --model-dir /tmp/qasr-models serve 1.7b --dry-run` 输出 `argv:["qwen-asr-serve","Qwen/Qwen3-ASR-1.7B"], source:"repo"`，`qwen-asr-serve` 由 vLLM 启动后会从 HF 拉取。修复：serve 前强制 `models show`/`verify` 为 complete；把“只有 pull 会下载”改成“transcribe/align 不会，serve 的 repo 回退会”；明确 `--offline` 对 serve 不生效。

## 重要问题（MAJOR）

6. **三处 `--json` + `-f json` 直接 exit 2** — 设计 168、210、229。实测 `qasr --json transcribe ... -f txt,json,srt` 和 `-f json` 均返回 `usage error: --format json/jsonl cannot be combined with --json/--jsonl`（`commands/transcribe.py:303-312`）。这与设计 495 行自己的反模式表矛盾。注意 `-f srt/vtt` + `--json` 合法，不能一刀切。修复：三处改 `-f txt,srt`、去掉 `-f json`，或去掉 `--json` 改读文件。

7. **manifest 黄金路径不可执行，且 per-item timestamps 不触发 180s 窗口** — 设计 236 的 `{"id":"ep02",...,"formats":["srt"],"timestamps":true}` 没有 `language`；实测同结构 manifest 直接 `capability error 4: timestamps require an explicit alignment language`（`commands/transcribe.py:243-247, 561-565`）。即使补上 language，`max_chunk_seconds` 仍按 CLI 级 `timestamps_default=False` 取 270（`commands/transcribe.py:377-379`），`_long_timestamps` 会按 270s 规划窗口并调用 aligner（`transcribe.py:1065-1078, 1151+`），与设计宣称的 180s 上限冲突。修复：示例加 `"language":"zh"`，manifest 命令加 `--timestamps` 或 `--max-chunk-seconds 180`。

8. **安装路径缺失，macOS 首用会踩空** — 全文唯一安装命令是 268 行的 `uv tool install --python 3.12 "qasr-cli[vllm,mic]"`。P0 需要的 `qasr-cli[local]` 一次都没出现；`pyproject.toml:10` 是 `>=3.10,<3.13`，本机默认 Python 3.14。`doctor` 在本机正是 `torch not installed` 的 fail 状态，而 §4.1 只体检不给修复命令。修复：首屏加 `uv tool install --python 3.12 "qasr-cli[local]"`，并说明 `[mic]`/`[vllm]`/`hf` 边界。

9. **P0 路由缺 `--stdin-list` 和直接 URL** — §3 的 18 行路由表没有 stdin-list，也没有 `qasr transcribe https://...`；只在 98 行 reference 职责和 manifest 示例里出现。实测两条都能跑（stdin-list rc0；URL dry-run rc0）。修复：各加一行最小命令与前置条件（多输入需 `--out-dir`）。

10. **远程 context 走 chat，`-l` 不被强制，设计未提 warning** — 设计 377 把 remote 的 context/LID 都标 ✓。`remote.py:596-604` 有 context 就走 `/v1/chat/completions`；`remote.py:647-649` 会追加 `language_not_enforced_on_chat`，语言不符还会追加 `server_ignored_language`。设计全文没有这两个 warning，agent 会误以为 `-l zh` 在远程 context 下生效。修复：矩阵注明“context→chat，语言不强制”，路由表要求读 `.result.warnings[]`。

11. **`serve` 示例缺少 Linux+CUDA/`[vllm]` 前置** — 设计 174/279 直接给 `qasr serve 1.7b --host ...`。`serve.py:20-24,127-142` 只是 `qwen-asr-serve` 的 exec 包装，运行需要 vLLM；macOS 上必然失败。设计只把“缺 vLLM”写在失败下一步，没写前置。修复：serve 路由标“Linux+CUDA+`[vllm]`”，macOS 明确不可用。

12. **“方言不能强制”过度概括，粤语被误伤** — 设计 162-163。实测 `-l yue` rc0（`forceable` 含 `yue`，`alignable` 也含 Cantonese）；只有 `Anhui`、`Cantonese (Hong Kong)` 这类 LID 标签才报 `dialect label and cannot be forced`。设计一边写可对齐语言含 `yue`，一边写“方言不能强制、只能 auto”，会让粤语时间戳任务走 `auto` 然后 exit 4。修复：改成“22 个 LID 方言标签不可强制；`yue`/Cantonese 是可强制、可对齐语言”。

13. **`--gpu-memory-utilization` 非 vLLM 说法错误** — 设计 273 写“只在 vLLM 生效”，暗示其他后端会忽略。实际 `qasr_cli/commands/transcribe.py:366-375` 在 `backend_choice != "vllm"` 时直接抛 `UsageError`（exit 2），只有 hint 文案说“ignored”。agent 若在 macOS/transformers 上照抄该参数会稳定失败。修复：改成“仅 `--backend vllm` 可用；其他后端传了直接 exit 2，不要传”。

14. **Claude Code 的发现路径缺失** — 设计 111-112、528-529 只建立 `~/.agents/skills/qasr-cli` 和项目 `.agents/skills`。这是 Codex 的 skill 根，不是 Claude Code 的发现目录；Claude Code 读 `~/.claude/skills`（或项目 `.claude/skills`）。本机 `~/.claude/skills` 已存在且部分条目反向 symlink 到 `../../.agents/skills`，但新用户不会自动拥有。修复：安装段补 `ln -sfn "$PWD/skills/qasr-cli" ~/.claude/skills/qasr-cli`，项目级补 `.claude/skills`，并说明两个客户端各读哪个根。

15. **`serve` 没有内建鉴权，设计未警示** — 设计 394 只讲客户端 token，422 只禁止绑定 `0.0.0.0`，容易被读成“有 token 就安全”。实际 `qasr_cli/serve.py:1-10,76-123` 是 `qwen-asr-serve` 的纯 argv 透传 + `os.execvp`，不注入任何 auth/token 参数；`QASR_SERVE_TOKEN` 只在客户端 `remote.py:35-38` 读取。也就是说服务端裸奔，token 不保护监听端口。修复：`deploy-and-remote.md` 明写“serve 无内建鉴权，只允许 `127.0.0.1`；要对内网/公网暴露必须由用户自备反向代理 + mTLS/Basic Auth”，并在 safety 表加一行。

## 次要问题（MINOR）

1. 设计 176 检查 `.rtf`、`.metrics`；`schemas/bench.json` 只有顶层 `rtf/load_s/audio_s/elapsed_s/iterations/peak_memory_mb`，没有 `.metrics`。
2. 设计 57 描述写 `markdown`；实际格式 token 是 `md`（`render.py:12`），`-f markdown` exit 2。
3. 设计 382 写 vLLM 用 `--max-inference-batch-size`；实际 `--batch-size` 对 vLLM 同样生效且默认 8（`commands/common.py:107-109`、`transcribe.py:579`），后者是模型内部批大小。
4. ~~设计 518 的核对脚本 `for c in "" ...` 会执行 `qasr "" --help`~~ **（已撤回，v1.1）**。未加引号的 `$c` 在 zsh/bash 都会让空元素消失，实测 argv 为 `qasr --help`，脚本行为正确；原判断来自我手工执行了带引号的 `qasr "" --help`，测试方法错误。
5. 设计 534/536 的触发集只有 30 条、阈值 10%/15%，低于 rubric 的 53 条与“正向核心 100%、高风险负向 0 误触发”。
6. 设计 170 说断点续跑看 `.summary.skipped`；单文件 `--resume` 走单 item envelope，没有 `.summary`（`transcribe.py:705-717`）。
7. 设计 287 `models rm 1.7b --yes` 没有先 `--dry-run`/`models show`；设计 291 `config set backend vllm --project` 会持久改项目配置，macOS 上后续命令直接 3。两处示例都与 §6 的确认原则相反。
8. 设计 223 没列 align 的 `language_source` 枚举（`explicit/script/lid_probe`），与 transcribe 的 `auto/forced/server` 不同，容易套错。
9. 设计 465 的 coverage 自检默认 `--coverage-min 0.5`，却断言 `>=0.8`；coverage 在 [0.5,0.8) 时 qasr 先 exit 5，jq 收到 error envelope。建议显式 `--coverage-min 0.8` 或先判 `.ok`。
10. 设计 168 说“所有机器消费都加 `--json`”，但 stream 不能用 `--json`；表头应改成“除 stream 外”。
11. `transcribe -f json` 写出的 sidecar 里 `outputs` 是空数组（`transcribe.py:1377-1383` 先写文件、后组装 payload）；设计 165 让 agent 检查 `.outputs[]` 时应说明读 stdout，不是读 `out/*.json`。
12. 设计 171 对长音频 OOM 建议“降 `--batch-size`”；单文件长音频走 `_process_long_item`，不经过 batching，应改“降 `--max-chunk-seconds`/换更小模型”。

## 维度评分表

| 维度 | 权重 | 分数 | 依据 |
|---|---:|---:|---|
| 事实准确性 | 25 | 2/5 | exit5/字幕质量、serve 下载、方言、bench、markdown、manifest 均错 |
| 触发覆盖与误触发 | 15 | 3.5/5 | 正负向结构好；缺“音频转文字/听写/热词/断点续跑” |
| 渐进披露与 token 成本 | 7 | 4/5 | 7 个 ref 职责清晰，SKILL 提纲 154 行；refs 尚未落地 |
| 任务路由完整性 | 15 | 2.5/5 | 18 行覆盖广；缺 stdin-list/URL/install/serve 前置，manifest 不可执行 |
| 命令可执行性 | 12 | 2/5 | 3 处 `--json`+`-f json` exit2；manifest 缺 language exit4 |
| 安全边界 | 15 | 1.5/5 | token argv、nohup、align 静默覆盖、serve 未确认下载 |
| 失败处理 | 5 | 2/5 | exit5 全错；远程 warning 缺失；其余基本正确 |
| 可维护性与版本化 | 3 | 3.5/5 | 版本锚点/单一来源好；核对脚本有空元素 bug |
| 可评估性 | 3 | 3/5 | 自检可执行性尚可；触发集与阈值弱于 rubric |

加权总分约 **48.6/100**，低于 75，判定 FAIL。

## Rubric errata

1. **“`--format` 与 `--json/--jsonl` 互斥”过宽**（rubric §1.1/§1.5）。实际只有 `-f json/jsonl/tsv/md` 与 `--json/--jsonl` 冲突；`-f srt` / `-f srt,vtt` + `--json` 实测 rc0。审核时不能把 design 的 `-f srt` 示例判错。
2. **align `--language auto` 不只是脚本判定**（rubric §2.2）。`commands/align.py:107-125` 先脚本判定，失败后 `:205-235` 调 `probe_language`，`language_source="lid_probe"`；dry-run 实测也显示 `lid_probe`。Latin 文本不会直接报错。
3. **align 长音频已自动窗口化，不是“P0 ≤180s 上限”**（rubric §2.1/§2.2）。`commands/align.py:79-98` 用 `plan_windows(window_s=180, overlap_s=3)`，`align.py:380-388` 多窗口合并；1 小时音频可跑，单窗 ≤180s。另：Qwen3 README 说 aligner “up to 5 minutes”，与 `utils.py:35` 的 180s 常量冲突，skill 应以 qasr 的 180s 为准。
4. **“方言只能由 LID 输出、不能强制”过宽**（rubric §2.1）。`forceable` 含 `yue`，`-l yue` rc0 且可对齐；只有 22 个 dialects 标签（Anhui、Cantonese (Hong Kong) 等）不可强制。
5. **“Python 必须 3.12 左右”不准确**（rubric §1.1）。`pyproject.toml:10` 是 `>=3.10,<3.13`，3.10/3.11/3.12 都支持；3.12 只是推荐。
6. **`models rm --dry-run` 期望 0 需限定模型位置**（rubric §4.3）。本机模型在 workspace dev path，实测 rc2；只有受管 `QASR_HOME/models` 下才是 0。

## 建议的最小修订集

1. 重写 exit 5：删“字幕质量”和“产物通常已写出”，只保留 align coverage / stream cap，并要求先读 error envelope。
2. 把 align 改成“默认覆盖、无 `--overwrite`”，示例必须先 dry-run + 用户确认。
3. 删所有 token 前缀赋值与 `user:fake@host`；只保留“用户自己 export，agent 不接触”。
4. 删 nohup，只留 tmux。
5. 修 3 处 `--json`+`-f json`；manifest 加 `language` 与 `--max-chunk-seconds 180`。
6. 首屏加 `uv tool install --python 3.12 "qasr-cli[local]"`。
7. 补 stdin-list、直接 URL、serve 前置、远程 chat warning、粤语 `-l yue` 五处事实。
8. serve 前加 `models show`/`verify` 门禁，明确 repo 回退会下载、`--offline` 不拦 serve。
9. 补 `~/.claude/skills` / `.claude/skills` 安装链接；明写 `serve` 无内建鉴权、只能 `127.0.0.1`；把 `--gpu-memory-utilization` 改成“非 vLLM 直接 exit 2”。

---

## v2 复核

**冻结对象**：`QASR-SKILL-DESIGN.md` v2 最终版，759 行 / 42,456 bytes，SHA-256 `1cf0248d965bce222fa46925c14efcf985c3f001aec9410460662b12d40d6bf3`，实测一致（在 `9314b490…` 基础上又修了两条残余 MINOR）。
**方法**：只读；对每条修订用 `--help`、`--dry-run`、`schemas/*.json` 与源码复核；未下载模型、未跑推理。

### 5 个 BLOCKER

| # | 复核点 | 判定 | 一手证据 |
|---|---|---|---|
| 1 | token 不进 argv/history | **PASS** | 设计 321、490-491、603 已改为用户 `read -rs`/`export`；全文 grep 无 `QASR_SERVE_TOKEN=` 前缀赋值、无 `user:fake@host`；`remote.py:35-38` 只读环境 |
| 2 | 只允许 tmux | **PASS** | 设计 509-517 只给 `tmux new-session`/`attach`；全文 `nohup` 仅出现在禁止句（511、610） |
| 3 | exit 5 语义 | **PASS** | 设计 544、550 已限定为 align coverage / stream cap，并写明无产物；`align.py:392-399` 在 `:415` 写文件前抛错，`stream.py:365-371` 不写文件；`transcribe` 无 QualityError |
| 4 | align 静默覆盖 | **PASS** | 设计 271、505 明写“无 `--overwrite`、目标存在即覆盖”，要求先 dry-run；实测 `align --help` 无 `--overwrite`，`align.py:450-489` 无 `exists()` 检查；dry-run 的 `.plan.outputs[0].path` 存在 |
| 5 | serve repo 回退 | **PASS** | 设计 208、349-350、496 已写 repo 回退会下载、`--offline` 不约束子进程；实测 `serve 1.7b --dry-run` 本地完整时 `source:"local"`，`serve.py:57-72` 对不完整模型回退 repo |

### 10 个 MAJOR

| # | 复核点 | 判定 | 证据 |
|---|---|---|---|
| 6 | `--json` + `-f json` | **PASS** | 设计 184、260、601 已列冲突；批量/字幕示例改用 `-f txt,srt`/`srt,vtt`；实测 `--json -f txt,srt` rc0，`-f json` rc2 |
| 7 | manifest language/180s | **PASS** | 设计 285-297 示例含 `language:"zh"` 与 `--max-chunk-seconds 180`；实测 manifest dry-run rc0；per-item 不带 180 时仍为 270，设计已明确警示 |
| 8 | `[local]` 安装 | **PASS** | 设计 96-115 首屏给出 `uv tool install --python 3.12 "qasr-cli[local]"` 与 extras 边界；`pyproject.toml:10` 为 `>=3.10,<3.13` |
| 9 | stdin-list / URL | **PASS** | 设计 200-201、299-303 已补路由与黄金路径；`--stdin-list`/URL dry-run 均可解析（URL rc0） |
| 10 | 远程 chat warning | **PASS** | 设计 328 明确 `context` 走 chat、`-l` 不强制，要求读 `language_not_enforced_on_chat`/`server_ignored_language`；对应 `remote.py:648-652` |
| 11 | serve 前置 | **PASS** | 设计 208-209、342-351 增加 `models show/verify` + `.source=="local"` 门禁，并标明 macOS 不能启动 |
| 12 | 粤语 | **PASS** | 设计 194、256、615 已把 `yue` 从“方言不可强制”拆出；实测 `-l yue --timestamps` dry-run rc0，22 个 LID 方言标签仍不可强制 |
| 13 | `--gpu-memory-utilization` | **PASS** | 设计 337-338、463、618 改为“仅 vLLM，其他后端 exit 2”；源码 `commands/transcribe.py:366-375` 一致 |
| 14 | Claude Code 发现路径 | **PASS** | 设计 133-145、645-651 增加 `~/.claude/skills/qasr-cli` 与项目 `.claude/skills` 链接，并说明 `.agents` 只是实体目标 |
| 15 | serve 无鉴权 | **PASS** | 设计 349-350、525-526 明写 serve 无内建鉴权、`QASR_SERVE_TOKEN` 仅客户端；`serve.py:1-10,76-123` 是 argv 透传 + `execvp`，不注入 auth |

### 原 MINOR 抽查

| 项 | 判定 | 证据 |
|---|---|---|
| 1 bench `.metrics` | PASS | 设计 371 已改为顶层字段，§9 反模式同步 |
| 2 `markdown` | PASS | 设计 88、621 使用 `md` |
| 3 vLLM 微批 | PASS | 设计 337 区分 `--batch-size`（默认 8）与 `--max-inference-batch-size` |
| 5 触发集 | PASS | 设计 655-658 扩到 53 条，门槛 100%/0 |
| 6 单文件 resume | PASS | 设计 203、587 写明无 `.summary`，看 `.skipped`；`schemas/transcribe-item.json:11` 有顶层 `skipped` |
| 7 rm/config | PASS | 设计 379-386 先 `rm --dry-run`/`config path`，并注明 dev path rc2 |
| 8 align `language_source` | PASS | 设计 272 列出 `explicit/script/lid_probe` |
| 9 coverage 自检 | PASS | 设计 576-577 显式 `--coverage-min 0.8` 且先判 `.ok` |
| 10 stream 例外 | PASS | 设计 183、361 明确 stream 不能用 `--json` |
| 11 sidecar `outputs` | PASS | 设计 590、619 要求读 stdout；实测 sidecar 的 `outputs` 确为空 |
| 12 长音频 OOM | PASS | 设计 316、612 改为降 chunk/换 0.6b，并说明降 `--batch-size` 无效 |

### 6 条 MINOR 复核（v2 最终版）

| # | 原问题 | 判定 | 证据 |
|---|---|---|---|
| 1 | 安装行缺 `--python 3.12` | **PASS** | 设计 97-100 四行均带 `--python 3.12`，与 102 行一致 |
| 2 | manifest 用 CLI `--timestamps` 兜底 | **PASS** | 设计 296-297 已改为“必须显式 `--max-chunk-seconds 180`”，并明确“不要用 CLI 级 `--timestamps` 兜底” |
| 3 | §8 自检依赖 `out/a.json` | **PASS** | 设计 570-573 改为自包含的 `qasr --json transcribe a.wav -l zh --timestamps \| jq ...`，不再读 sidecar |
| 4 | jobs 账本漏 pending/running | **PASS** | 设计 588 已改为 `.state=="present"` 且 `pending/running==0`；并注明五类之和 == `.job.total` 仅在无 spec 校验失败时成立。`.job.total` 确由 `jobs.py:389` 写入 record |
| 5 | 两处行号漂移 | **PASS** | 设计 185 改 `commands/transcribe.py:299-302`（该 `UsageError` 文案本身写明“out-dir 或 json/jsonl”规则）；附录 A#19 改 `:366-375`，与源码一致 |
| 6 | completion 先写 home | **PASS** | 设计 423-424 先 `qasr completion zsh \| head` 看 stdout，再给“用户确认落点后”重定向示例 |

### 残余 MINOR 收尾（2 条）

| # | 原问题 | 判定 | 证据 |
|---|---|---|---|
| 1 | §8 jobs 判据对“校验失败输入”不成立 | **PASS** | 设计 588 已改为 `.state=="present"` 且 `.counts.pending==0`、`.counts.running==0`，并明确“五类之和 == `.job.total` 仅在没有 spec 校验失败时成立：`total` 含 `spec_errors`，而它们不写 state（`commands/transcribe.py:532-534`）”。与 `commands/jobs.py:56-66` 的计数口径一致 |
| 2 | §4.7 “所有记录”措辞过宽 | **PASS** | 设计 297 已改为“所有未显式写 `timestamps` 的记录切到时间戳模式（显式 `false` 仍是 false）”，与 `commands/transcribe.py:555` 的 `record.get("timestamps", timestamps_default)` 一致 |

**剩余问题：0 条 MINOR**（仅剩 references 尚未落地这类交付物层面的待办，不是设计缺陷）。

### 更新评分

| 维度 | 权重 | 分数 | 依据 |
|---|---:|---:|---|
| 事实准确性 | 25 | 4.8/5 | 关键断言已与源码/探针一致；两条残余自检/措辞已修 |
| 触发覆盖与误触发 | 15 | 4.5/5 | 正负向与 53 条评估计划完整 |
| 渐进披露与 token 成本 | 7 | 4/5 | 分层清晰；references 尚未落地 |
| 任务路由完整性 | 15 | 4.5/5 | 26 行覆盖 P0/P1；缺 completion 之外的边角 |
| 命令可执行性 | 12 | 4.5/5 | 抽查 dry-run 均 rc0；安装与自检示例已修 |
| 安全边界 | 15 | 5/5 | token/tmux/覆盖/serve 鉴权/下载均已明写 |
| 失败处理 | 5 | 5/5 | exit 5 与 partial/all-fail 语义正确 |
| 可维护性与版本化 | 3 | 4.5/5 | 版本锚点与核对脚本可执行；jobs 判据已自洽 |
| 可评估性 | 3 | 4.5/5 | 53 条触发集与阈值可执行；自检 jobs 判据已放宽 |

加权总分 **92.8/100**。

**最终判定：PASS**。0 BLOCKER、0 MAJOR、0 MINOR；G1/G2/G3 全部通过。冻结版 `1cf0248d…`（759 行 / 42,456 bytes）的 §8 jobs 判据与 §4.7 措辞已按建议修正；注意 references/*.md 仍需随 skill 一并落地，否则 SKILL.md 的渐进披露只有骨架。
