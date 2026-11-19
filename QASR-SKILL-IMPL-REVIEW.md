# qasr-cli skill 实现审查（Partner B）

- 对象：`skills/qasr-cli/SKILL.md`（170 行）、7 个 references（86–157 行）、`README.md`（50 行）。
- 基线：`QASR-SKILL-DESIGN.md` 759 行，sha256 `1cf0248d…d40d6bf3`；CLI 0.3.0（tag `b945bb4`）。
- 方法：只读；`--help`、`--dry-run`、非法参数、隔离 `QASR_HOME=/tmp/qzr/*`、源码交叉核对。未下载模型、未联网、未跑推理、未用真实 token。
- **结论：PASS**（原 4 MAJOR + 9 MINOR 已全部修复；0 BLOCKER / 0 MAJOR / 0 MINOR；加权约 96.7/100）。返修复验见文末。

## 阻塞问题（BLOCKER）

无。已核查 token 未进入 argv/历史/URL/config（`safety.md:33` 仅禁止示例）、无 `nohup`/裸 `&`、exit 5 未被说成“已写产物”、`align` 静默覆盖有警告、`serve` 的 repo 回退门禁为 `.source=="local" and .path!=null`、`serve` 无内建鉴权。未发现泄密、丢数据或把翻译等未实现能力写成支持。

## 重要问题（MAJOR）

### M1. `config set` 接受 `--dry-run` 但静默写盘

- 位置：`SKILL.md:46` 要求“动作前先 `--dry-run`”；`references/ops-and-models.md:58`、`references/safety.md:15` 只说“没有 dry-run”。
- 证据：`QASR_HOME=/tmp/qzr/hM1 .venv/bin/qasr --dry-run config set backend vllm` 返回 rc0，并生成 `config.json`：`{"backend":"vllm"}`；flag 放在子命令后同样写盘。`qasr_cli/commands/config.py:77-127` 的 `_set` 从不读取 `ctx.dry_run`，在 `:110-112` 直接 `write_json_atomic`。
- 影响：agent 按首屏“先 dry-run”执行，会在 macOS 上把默认 backend 改成 vllm，之后本地命令直接 exit 3。必须说明“flag 被接受但忽略”，而不只是“没有 dry-run”。
- 建议：`ops-and-models.md:58`、`safety.md:15` 写明“`--dry-run` 会被接受但完全忽略，仍立即写盘”；`SKILL.md:46` 改为“`config set` 先 `config path`，它没有 dry-run”。

### M2. `align --backend vllm` 实际被忽略，技能却统一说 exit 3

- 位置：`SKILL.md:110,117,125`；`references/deploy-and-remote.md:96-97,103`；`references/ops-and-models.md:64`；`references/troubleshooting.md:53`。
- 证据：`qasr --json --backend vllm align /tmp/qz-a.wav --text-file /tmp/qz-t2.txt -l en --dry-run` 返回 rc0、`ok:true`，无平台错误。`qasr_cli/commands/align.py:54-60` 只拒绝 remote，不调用 `require_vllm`；`qasr_cli/align.py:133-163` 的 aligner 与 ASR anchor 都固定 `build_transformers_backend`，`AlignSettings`（`commands/align.py:187-209`）没有 backend 字段。
- 影响：单窗口且语言显式时，macOS 上会成功但走 transformers/MPS；长音频 anchor 也走 transformers。技能让 agent 误以为“会 3”或“用了 vLLM”，性能与平台判断都会错。
- 建议：改为“`align` 始终走 transformers（aligner + 长音频 anchor ASR），`--backend vllm` 对 align 不生效，也不会因平台返回 3；要 vLLM 请只用 `transcribe`”。若产品希望拒绝，应改代码；0.3.0 的技能必须描述现状。

### M3. `align -l auto` 被语言规则误写成 exit 4，文档自相矛盾

- 位置：`references/commands-core.md:36`（“时间戳/`align`……`auto` → 4”）、`SKILL.md:56-57`；与之矛盾的是 `SKILL.md:85`（“align 必须给语言（`auto` 可）”）与 `commands-core.md:104`（“`auto` 会用文字脚本或音频 LID 探测”）。
- 证据：`qasr --json align /tmp/qz-a.wav --text-file /tmp/qz-t2.txt -l auto --dry-run` 返回 rc0，`language:"auto"`、`language_source:"lid_probe"`；不写 `-l` 才 rc2（“align requires --language”）；`transcribe --timestamps -l auto` 才 rc4。代码见 `qasr_cli/commands/align.py:104-118`、`qasr_cli/commands/transcribe.py:64-75`。
- 影响：agent 会跳过合法的 align auto 路径，或在混合语言素材上硬指定语言。`long-audio-and-stream.md:43` 的“`-l auto` 时需要 ASR”也过强：CJK 文稿可能先由脚本检测直接得到语言，只有检测失败才做音频 LID。
- 建议：把“`auto` → 4”限定为“`transcribe --timestamps`”；align 段明确“`-l auto` 合法：先按文稿脚本检测，失败再做音频 LID；缺 `-l` 才是 2”；`long-audio:43` 改为“脚本检测失败或窗口 >1 时需要 ASR”。

### M4. `--json` + 多格式 `-f` + `-o` 仍会写文件，技能却说“不写文件”

- 位置：`SKILL.md:134`、`references/commands-core.md:58`。
- 证据：`QASR_HOME=/tmp/qzr/home .venv/bin/qasr --json transcribe /tmp/qz-zero.wav -l zh --timestamps -f srt,vtt -o /tmp/qzr/jout` 返回 rc0，实际写出 `qz-zero.srt` 与 `qz-zero.vtt`，envelope 的 `outputs` 列出两者；`--dry-run` 的 `.plan.outputs` 也列出两个路径。`qasr_cli/transcribe.py:1462-1475` 只要 `out_dir`/`output` 就写，`--json` 只在 `:704-717` 控制 stdout；`commands-core.md:49` 自己也说“`-o` 写文件，`--json` 仍打印 envelope”。
- 影响：agent 会以为 `--json` 抑制了文件写入，脚本可能留下意外产物；目标已存在时还会 exit 4。C 自报第 5 条只在“没有 `-o`”时成立，设计稿并未这样写。
- 建议：两处改为“`--json` 下多格式 `-f` 合法；无 `-o` 时只输出 envelope、不落盘；给了 `-o` 仍会写多个文件（受 exists/4 约束）”。

## 次要问题（MINOR）

1. `references/ops-and-models.md:30` 写“`models rm ... --dry-run` 仅当模型存在时”，但存在但位于 dev path 的模型会 rc2（“refusing to remove a non-managed directory”），不存在才 rc3；且 `models rm` 输出的 `.status` 是 `"present"`，不是 `:34` 的 `complete|incomplete|missing`。建议保留设计稿 `:379` 的 `[managed]` 限定。
2. `references/ops-and-models.md:26-35` 与 `references/troubleshooting.md:101-116` 未提醒 `models show <不存在>` 也 rc0、`.ok:true`、`.status:"missing"`、`.path:null`（实测；`commands/models.py:62-72` 无条件 `EXIT_OK`）。建议加入“假成功”表：必须看 `.status`，不能只看 rc0。
3. `SKILL.md:82` 的热词行没有 `--json`，与 `SKILL.md:58`“机器消费一律 `--json`”不一致；补上即可。
4. `SKILL.md:94` 只写“（tmux 内）”，没有给出 `tmux new-session -d -s qasr-serve '...'`；只读首屏的 agent 可能前台启动并阻塞。建议内联一行或明确指向 `deploy-and-remote.md:31-35`。
5. `references/long-audio-and-stream.md:74-75` 把 `audio_s elapsed_s language text is_final` 写成“字段”，实际快照还含 `schema_version/ok/command/seq`（`qasr_cli/stream.py:209-219`）。改为“关键字段”或补全。
6. `SKILL.md:14` 的 frontmatter 负向只排除 TTS、克隆、配音、通用音频编辑；设计 §1.2 还排除翻译、说话人分离、训练、ffmpeg 转码。建议加一句“Not for translation, speaker diarization, training, or ffmpeg-style media conversion.”。
7. 设计 §10 的 53 条 trigger 评估没有落地；`skills/qasr-cli/README.md:38-47` 只验证 `--help`。建议增加 `skills/qasr-cli/evals/triggers.md`（中英文正向/负向清单与期望）。

## C 的 10 条自报修正判定

| # | C 的说法 | 判定 | 一手证据 |
|---|---|---|---|
| 1 | `--jsonl` 无 summary，`done:` 只在写文件模式 stderr | 正确 | `transcribe.py:690-703,757-759`；`SKILL.md:130` |
| 2 | glob `**` 仍需 `--recursive` | 正确 | 嵌套目录实测 2 个输入 vs 加 `--recursive` 3 个；`batch-and-resume.md:12` |
| 3 | `stream` 的 `--context` 与 `--context-file` 互斥 | 正确 | `cli.py:271-274`；rc2；`long-audio-and-stream.md:68` |
| 4 | `align -f json --json` 合法 | 正确 | 实测 rc0；`commands/align.py` 无 json/format 冲突；`commands-core.md:130` |
| 5 | `--json` 下多格式 `-f` 合法但不写文件 | **错误（需限定）** | 合法正确；带 `-o` 实测写出 srt/vtt。见 M4 |
| 6 | `serve --dry-run` 未知模型返回 `source:"local"`、`path:null`；门禁要 `.source=="local" and .path!=null` | 正确 | `serve.py:97-105`；`deploy-and-remote.md:18,22-23` |
| 7 | `models rm` dry-run 仅模型存在时出计划，不存在 rc3 | **基本正确但不完整** | dev path 存在时 rc2；见 MINOR 1 |
| 8 | `models show` 未知别名 rc0、`.status=="missing"` | 事实正确，但技能未写警告 | 实测 rc0；见 MINOR 2 |
| 9 | `jobs` 五类 counts 之和可能小于 `total`（spec 校验失败） | 正确 | `batch-and-resume.md:116-117`；`troubleshooting.md:110` |
| 10 | `--skip-existing` 只跳过非空输出 | 正确 | `transcribe.py:1442-1447`；`commands-core.md:62` |

10 条中 8 条正确；第 5 条是新引入错误（设计稿未说“不写文件”）；第 7 条丢掉了设计稿的 `[managed]` 限定；第 8 条事实正确但技能缺少“假成功”警告。

## 维度评分

| 维度 | 权重 | 得分/5 | 加权 | 主要扣分 |
|---|---:|---:|---:|---|
| 事实准确性 | 25 | 3.5 | 17.5 | M2/M3/M4 |
| 触发覆盖与误触发 | 15 | 4.5 | 13.5 | frontmatter 负向不全 |
| 渐进披露 | 7 | 5 | 7 | 结构、行数、首行均达标 |
| 任务路由完整性 | 15 | 4.5 | 13.5 | align auto/vllm 路由错误 |
| 命令可执行性 | 12 | 4.5 | 10.8 | 示例大多可执行；输出契约有误 |
| 安全边界 | 15 | 4.0 | 12.0 | M1 静默写盘 |
| 失败处理 | 5 | 4.5 | 4.5 | `models show` 假成功 |
| 可维护性 | 3 | 4.5 | 2.7 | 版本基线清楚；缺 trigger eval |
| 可评估性 | 3 | 4.0 | 2.4 | 设计 §10 未落地 |
| **合计** | **100** |  | **83.9** | **PASS WITH CHANGES** |

## 建议的最小修订集

1. `ops-and-models.md:58`、`safety.md:15`、`SKILL.md:46`：明确 `config set --dry-run` 被接受但忽略并写盘（M1）。
2. `SKILL.md:110,117,125`、`deploy-and-remote.md:96-97,103`、`ops-and-models.md:64`、`troubleshooting.md:53`：把 align 的 backend 语义改为“始终 transformers，vllm flag 不生效”（M2）。
3. `commands-core.md:36`、`SKILL.md:56-57`：把“auto → 4”限定到 `transcribe --timestamps`；align 的 auto 合法（M3）。
4. `SKILL.md:134`、`commands-core.md:58`：补“无 `-o` 才不写文件；有 `-o` 仍写”（M4）。
5. `ops-and-models.md:30,34,41-42`、`troubleshooting.md:101-116`：补 `[managed]`、rm 的 `.status:"present"`、`models show` 未知也 rc0（MINOR 1/2）。
6. 其余 MINOR 3–7：热词行补 `--json`、serve 内联 tmux、stream 字段措辞、frontmatter 负向、trigger eval 清单。

## 设计稿/README 回写项（不在本次修改范围）

- `QASR-SKILL-DESIGN.md:450` 的“显式 vllm 但平台不支持 → transcribe/align 3”与 0.3.0 的 align 实现不符，应改为“align 忽略 backend，始终 transformers”。
- 设计稿的 `config set` 安全行建议补“`--dry-run` 会被接受但忽略”，避免设计与技能都只写“没有 dry-run”。
- `README.md:195` 仍把 exit 5 写成 “alignment coverage / subtitle quality gate”；`transcribe` 没有质量门禁。技能的 `troubleshooting.md:81` 已正确纠正，README/设计表格应回写。

## 已核对无问题（抽样）

- 结构：`SKILL.md` 恰 170 行（≤170）；7 个 ref 均 ≤220 行；每个 ref 第 1 行均为“何时读我：”；`README.md` 同时给出 `~/.agents/skills` 与 `~/.claude/skills` 发现链接。
- 安装：`uv tool install --python 3.12 "qasr-cli[local]"` / `[vllm,mic]` / base 与 `pyproject.toml:10,17-23` 一致；`requires-python >=3.10,<3.13`；`[vllm]`/`[mic]`/`[hf]`/`[modelscope]` 边界正确。
- 退出码与输出：0/2/3/4/5/6/130 与 `errors.py:7-13`、`cli.py:516-524` 一致；`--jsonl` 无 summary、`done:` 仅写文件模式、sidecar `.outputs` 可能为空、`--json`/`-f` 冲突集合正确（除 M4 的“不写文件”表述）。
- serve 与安全：`models show`/`verify` → `serve --dry-run` 门禁正确；未知名字 `local+null` 被拦截；`repo` 会下载；`aligner` 为 2；缺 `qwen-asr-serve` 为 3；`--host` 默认 0.0.0.0 的警告正确；无 token 前缀赋值、无 `nohup`/裸 `&`、无真实凭据。
- 长音频与 stream：270/180/600、`--window-seconds` 10–180、`--window-overlap` 0–10、`align` 自动窗口化、coverage 低于阈值不写文件；`stream --json`/`--dry-run` 均 2，仅 Linux+CUDA+vLLM，累计快照可能回滚，cap 为 5，Ctrl-C 先 final 再 130。

---

## 返修复验（最终）

- 复核对象：C 的返修版。只读 + 隔离 `QASR_HOME` 抽查 M1–M4 各一条命令；未跑推理、未联网、未下载模型。
- **最终 verdict：PASS**（0 BLOCKER / 0 MAJOR / 0 MINOR，加权约 96.7/100）。

### 实现文件行数清单

| 文件 | 行数 | 约束 | 结论 |
|---|---:|---|---|
| `skills/qasr-cli/SKILL.md` | 170 | ≤170 | PASS |
| `references/commands-core.md` | 158 | ≤220 | PASS |
| `references/ops-and-models.md` | 149 | ≤220 | PASS |
| `references/troubleshooting.md` | 139 | ≤220 | PASS |
| `references/batch-and-resume.md` | 135 | ≤220 | PASS |
| `references/deploy-and-remote.md` | 118 | ≤220 | PASS |
| `references/long-audio-and-stream.md` | 96 | ≤220 | PASS |
| `references/safety.md` | 86 | ≤220 | PASS |
| `skills/qasr-cli/README.md` | 57 | — | PASS |
| `skills/qasr-cli/evals/triggers.md` | 63（53 条） | 新增 | PASS |

7 个 ref 的首行均为“何时读我：”。

### 原 4 MAJOR 复验

| # | 结论 | 一手证据 |
|---|---|---|
| M1 `config set` 静默写盘 | **PASS（已修）** | `SKILL.md:46`、`ops-and-models.md:60`、`safety.md:15` 均写明“`--dry-run` 被接受但忽略”。隔离复测：`QASR_HOME=/tmp/qzr/v2m1 qasr --dry-run config set backend vllm` → rc0 且写出 `{"backend":"vllm"}`，与文案一致。 |
| M2 `align --backend vllm` | **PASS（已修）** | `SKILL.md:110,117`、`deploy-and-remote.md:96-97,103`、`ops-and-models.md:66`、`troubleshooting.md:53` 统一为“align 始终 transformers；vllm 不生效、不因平台 3；remote 仍 4”。复测 rc0、`ok:true`，与文案一致。 |
| M3 `align -l auto` | **PASS（已修）** | `commands-core.md:36-37` 把 `auto→4` 限定到 `transcribe --timestamps`；`SKILL.md:57` 注明 align 例外；`long-audio-and-stream.md:43` 改为“脚本检测失败或窗口>1 才需 ASR”。复测 `align -l auto --dry-run` rc0。 |
| M4 `--json` + 多格式 `-f` + `-o` | **PASS（已修）** | `SKILL.md:135`、`commands-core.md:59-60` 改为“无 `-o` 才不落盘；给 `-o` 仍写多个文件（已存在则 4）”。 |

### 原 7 MINOR 复验

| # | 结论 | 证据 |
|---|---|---|
| 1 `models rm` 受管/状态 | **PASS** | `ops-and-models.md:30,34,41-42` 已补 `[managed]`、dev path→2、不存在→3、`.status=="present"`。 |
| 2 `models show` 假成功 | **PASS** | `ops-and-models.md:34` 与 `troubleshooting.md:109` 已写 rc0/`ok:true`/`status:"missing"`/`path:null`。 |
| 3 热词行缺 `--json` | **PASS** | `SKILL.md:82` 已补。 |
| 4 serve 未内联 tmux | **PASS** | `SKILL.md:94` 已给 `tmux new-session -d -s qasr-serve '...'`。 |
| 5 stream 字段不全 | **PASS** | `long-audio-and-stream.md:74-75` 已补 `schema_version/ok/command/seq`，并称“关键字段”。 |
| 6 frontmatter 负向不全 | **PASS** | `SKILL.md:14` 已加 translation/speaker diarization/training/ffmpeg-style。 |
| 7 trigger eval 未落地 | **PASS** | 新增 `evals/triggers.md`：53 条（36 正/17 负），含边界项 #52。 |

### 收尾修复确认（原 2 MINOR）

1. **align 的 `-f` 能力**：**PASS（已修）**。`SKILL.md:132` 与 `commands-core.md:132` 现写明“align 同样支持 `txt,json,jsonl,srt,vtt,tsv,md`（help 只列 4 种），多格式需 `-o`，全局 `--jsonl` 为 2”。复核 `align --help` 确实只列 `json,srt,vtt,txt`；`align.py:47` 校验全量 `FORMATS`（7 种），`align.py:470-488` 实现 jsonl/txt/tsv/md，文案与实现一致。
2. **`SKILL.md` 行数**：**PASS（已修）**。现为 170 行，满足设计 §2 的 ≤170；合并后未丢失 align 说明。

### C 本轮“我补”三条判定

| 改动 | 判定 | 证据 |
|---|---|---|
| `SKILL.md:132` align `-f` 枚举 | **先错后修（PASS）** | 首版误收窄为 4 种；复验发现后已改为 7 种，见“收尾修复确认”。 |
| `batch-and-resume.md:41` `audio_base64:"AAAA"` | 可接受（占位符） | `AAAA` 是合法 base64，但不是可解码音频；实测 dry-run rc4（Format not recognised）。与同块不存在的 `audio/ep01.wav` 一样属示意值，不单列问题。 |
| `ops-and-models.md:30` 补“仅受管目录” | **正确** | 与 `commands/models.py` 的 managed-root 校验一致。 |

### 更新后的维度评分

| 维度 | 权重 | 得分/5 | 加权 |
|---|---:|---:|---:|
| 事实准确性 | 25 | 4.8 | 24.0 |
| 触发覆盖与误触发 | 15 | 4.7 | 14.1 |
| 渐进披露 | 7 | 5.0 | 7.0 |
| 任务路由完整性 | 15 | 4.9 | 14.7 |
| 命令可执行性 | 12 | 4.8 | 11.5 |
| 安全边界 | 15 | 4.9 | 14.7 |
| 失败处理 | 5 | 5.0 | 5.0 |
| 可维护性 | 3 | 4.7 | 2.8 |
| 可评估性 | 3 | 4.8 | 2.9 |
| **合计** | **100** |  | **96.7** |

**最终判定：PASS。** 剩余问题 0，可定稿发布。
