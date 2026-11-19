# qasr-cli Skill 评审标准（Rubric v1）

> 评审对象：`qasr-cli` 的 Claude Code skill 设计稿（`SKILL.md` + `references/*.md`）。
> 事实基线：本仓库 `qasr 0.3.0`、`README.md`、`DESIGN.md`、`P1-REPORT.md`、`P2-TASKS.md` §7、`P2-REPORT.md`、`schemas/*.json`，以及本机实际执行 `.venv/bin/qasr --help` 和各子命令 `--help` 的输出。
> 评审立场：Reviewer 只读、对抗式审核；不以设计文档自证，不以“看起来完整”代替可执行验收。

## 0. 结论口径与硬门禁

### 0.1 结论等级

| 结论 | 条件 |
|---|---|
| **PASS** | 所有硬门禁通过；无 BLOCKER、无 MAJOR；加权分 ≥ 90 |
| **PASS WITH CHANGES** | 所有硬门禁通过；无 BLOCKER；MAJOR ≤ 5；加权分 75–89 |
| **FAIL** | 任一 BLOCKER；或硬门禁失败；或 MAJOR > 5；或加权分 < 75 |

### 0.2 严重级别

| 级别 | 定义 | 例子 |
|---|---|---|
| **BLOCKER** | 按 skill 执行会得到错误结果、丢数据、泄露凭据、把不支持的能力说成支持，或核心任务无法启动 | 把远程时间戳写成可用；把 `QASR_SERVE_TOKEN` 放进命令行；建议无确认删除模型 |
| **MAJOR** | 关键任务路径缺失、命令不可直接执行、退出码/JSON 解读错误，但不会立即造成不可逆损害 | 只写 `qasr stream --mic`（实际是 `--input mic`）；把 `doctor.ok` 当健康 |
| **MINOR** | 表述不精确、示例不完整、参考文件组织不佳，不影响正确执行 | 未说明 `--jsonl` 只对 transcribe 有意义 |

### 0.3 硬门禁（任一失败即不能 PASS）

| ID | 门禁 |
|---|---|
| G1 | 不虚构 flag、子命令、默认值、JSON 字段、退出码或能力；所有关键断言能由 `--help`、`schemas/`、README 或实际命令复现 |
| G2 | 不把 token、密码、API key 写入 argv、示例明文、config、JSON 或状态文件；不把 `QASR_SERVE_TOKEN` 描述成服务端鉴权 |
| G3 | 不自动建议下载模型、删除模型、覆盖输出或把音频发往远端；这些操作必须先说明后果并取得用户确认 |
| G4 | 触发描述覆盖 qasr 的核心 ASR 任务，且不会把 TTS、音色克隆、模型训练、说话人分离等任务误路由到 qasr |
| G5 | P0 任务至少各有一条可复制执行的命令路径；不能只写“支持某能力”而不给命令、前置条件和失败处理 |

### 0.4 事实基线（评审时必须重新核对）

当前 `qasr 0.3.0` 的顶层命令是：

`transcribe`、`align`、`serve`、`models`、`languages`、`doctor`、`config`、`jobs`、`bench`、`stream`、`completion`、`finetune`。

关键基线如下，skill 若与之冲突即为事实错误：

| 事实 | 基线 |
|---|---|
| 版本/安装 | `qasr --version` 输出 `qasr 0.3.0`；Python 要求 `>=3.10,<3.13`，主推 `uv tool install --python 3.12 "qasr-cli[local]"` |
| 轻量 base | base 只含 `numpy`、`soundfile`；本地推理必须 `[local]`，vLLM 必须 `[vllm]`，麦克风必须 `[mic]` |
| 模型目录 | 模型在 `QASR_HOME/models`，不在 `uv tool` 的 venv；macOS 默认 `~/Library/Application Support/qasr` |
| 退出码 | `0` 成功、`2` 用法/配置/manifest、`3` 模型/运行时/下载、`4` 输入/路径/能力、`5` 质量门禁、`6` 批量部分失败、`130` 中断 |
| 输出 | `--json` 一个对象；`--jsonl` 每项一行；`--format` 是文件格式，且与 `--json/--jsonl` 互斥 |
| `doctor` | `ok` 不是健康结论；健康看 `status`、`checks[]`；默认即使 `status=fail` 也可能退出 0，`--strict` 才在 warn/fail 时退出 3 |
| `align` JSON | `language`、`language_source` 在 **顶层**，不在 `result` 内；`transcribe` 的语言字段在 `result` 内 |
| 时间戳 | `--timestamps` 必须显式给 11 种可对齐语言之一；`--language auto --timestamps` 退出 4 |
| 语言表 | 30 个可强制语言、52 个 LID 标签（含 22 个方言标签）、11 个可对齐语言；方言只能由 LID 输出，不能强制 |
| 远程 | `--server` 不支持时间戳、`align` 和 `stream`，请求前即应以能力错误退出 4；远程 context 走 chat，transcription 端点不带 context |
| 本地并发 | `--jobs` 仅远程有效；本地大于 1 会 warning 并强制 1；`--batch-size` 不是解码内存上限 |
| 流式 | 仅本地 vLLM、Linux + CUDA；`--input file|mic|stdin`；`--emit jsonl` 是累计快照；与全局 `--json` 无条件互斥（退出 2） |
| 模型删除 | `models rm` 只删 `QASR_HOME/models` 下的受管目录；非 TTY 必须 `--yes` |
| 服务安全 | `qasr serve` 是 `qwen-asr-serve` 的薄包装；`QASR_SERVE_TOKEN` 是 **客户端** token，不是 `serve` 的内建鉴权 |

## 1. 维度、权重与通过标准

| 维度 | 权重 | 核心问题 |
|---|---:|---|
| 事实准确性 | 25 | flag、默认值、退出码、JSON 字段、能力边界是否与 0.3.0 一致 |
| 触发覆盖与误触发 | 15 | 该用的任务能否触发，不该用的任务是否被误触发 |
| 渐进披露与 token 成本 | 7 | 首屏是否够用，细节是否按需加载，是否复制了整份 `--help` |
| 任务路由完整性 | 15 | 用户意图是否有最小命令、前置条件、下一步和错误分支 |
| 命令可执行性 | 12 | 示例能否原样执行，参数位置和互斥关系是否正确 |
| 安全边界 | 15 | token、下载、删除、覆盖、外发、凭据、服务暴露是否安全 |
| 失败处理 | 5 | 退出码、错误 envelope、重试、断点、部分失败是否正确 |
| 可维护性与版本化 | 3 | 是否绑定版本、单一事实来源、可更新 |
| 可评估性 | 3 | 每条主张能否被测试或人工复核 |

评分方法：每个维度按 0–5 打分，`维度得分 = 权重 × 分数 / 5`，总分 100。硬门禁优先于分数；任何 BLOCKER 直接 FAIL，不因其他维度得分高而抵消。

### 1.1 事实准确性（25）

**通过标准**

- 每个 flag 都能在对应层级的 `--help` 中找到；位置正确，例如 `qasr --json models list` 和 `qasr models list --json` 可用，但 `qasr models --json list` 退出 2。
- 默认值、取值范围、互斥关系写对：`--format` 与 `--json/--jsonl` 互斥；`-o/--out-dir` 不是 `--output`；`--timestamps` 需显式可对齐语言；`--jobs` 本地无效；`stream` 与 `--json` 互斥。
- JSON 字段与 `schemas/*.json` 一致；特别区分 `align` 顶层语言字段与 `transcribe.result.language`。
- 能力矩阵准确：远程无时间戳/对齐/流式；vLLM 仅 Linux + CUDA；macOS 走 transformers/MPS；`finetune` 只 lint，不训练；不声称翻译、说话人分离。
- 版本和安装边界准确：base 轻量，`[local]` 才有 torch/qwen-asr，`[vllm]` 才有 serve/stream，Python 必须 3.12 左右。

**典型反例**

- 写 `qasr transcribe --timestamps --language auto` 可用；实际退出 4，提示必须给显式语言。
- 写 `qasr align --server ...` 或 `qasr transcribe --server ... --timestamps` 可用；实际均为能力错误退出 4。
- 写 `qasr stream --server ...` 或“macOS 也能 stream”；实际远程/非 vLLM 拒绝，macOS 无 vLLM。
- 把 `qasr stream --mic` 当命令；实际参数是 `qasr stream --input mic`。
- 把 `--format json` 与 `--json` 混用；实际互斥并退出 2。
- 把 `doctor` 的 `ok=true` 当“健康”；实际健康字段是 `status` 和 `checks[]`，且默认退出码可能仍是 0。
- 把 `models list` 的 `present=true` 当“模型完整”；必须看 `status=complete`，必要时 `models verify`。
- 把 `QASR_SERVE_TOKEN` 当服务端鉴权；实际它只被客户端读取，`qasr serve` 本身没有内建 token 校验。

### 1.2 触发覆盖与误触发（15）

**通过标准**

- frontmatter 的 `description` 以 `qasr`/`Qwen3-ASR` 和核心动作开头，包含转写、LID、时间戳、对齐、批处理、远程/流式、模型诊断等高信号词。
- 中文和英文的核心说法都能触发；对“音频转文字”“转写”“字幕时间轴”“强制对齐”“热词偏置”“断点续跑”等至少各有一个入口。
- 与相邻的 `qtts-cli` 明确分工：TTS、配音、音色克隆、有声书归 qtts；qasr 只处理“音频 → 文本/时间轴”。
- 误触发可控：TTS、训练、说话人分离、翻译、纯视频剪辑、通用音频降噪等不应被描述成 qasr 能力。
- description 以正向能力为主；负向边界放在 `SKILL.md` 顶部，不在 description 里堆一长串否定关键词。

**典型反例**

- description 只写“Qwen ASR helper”，没有 `transcribe`、timestamps、align、batch 等动作词，导致“把会议录音转成 SRT”不触发。
- 把“语音合成”“配音”“克隆音色”写进正向触发，导致 qtts 任务误入 qasr。
- 把“训练/微调 ASR”写成 qasr 能力；实际只有 `finetune lint-jsonl`。
- 把“给视频加字幕”无条件触发；如果用户要的是字幕翻译或烧录，应由用户澄清，不能声称 qasr 完成。
- description 过长且关键词堆砌，模型只记住“ASR”而忽略 qasr 的本地/批处理边界。

### 1.3 渐进披露与 token 成本（7）

**通过标准**

- `SKILL.md` 是可执行的操作手册，建议 ≤ 200 行、≤ 10 KB；首屏包含最小可用路径、三条硬安全规则、退出码表和任务路由表。
- 详细 flag、JSON schema、远程/长音频/流式细节放入 `references/`，每个 reference 有明确“何时读”的入口。
- 不复制完整 `--help`；用“命令 + 关键 flag + 指向 `qasr <cmd> --help`”的方式保持可维护。
- 常见任务只加载 1–2 个 reference；不要让用户为了转写一个文件先读完整套 60 KB 文档。
- 安全、能力边界、错误处理在首屏可见，不藏在最后一个 reference。

**典型反例**

- `SKILL.md` 把 12 个子命令的全部 flag 和所有 JSON 字段都抄一遍，超过 30 KB，且与 `--help` 漂移。
- 首屏只有“qasr 很强大，详见 references”，没有任何可直接运行的最小命令。
- 每个任务都要求读 6 个 reference，导致 token 成本和延迟不可控。
- 把 `schemas/*.json` 全文嵌入 skill，而不是在需要时读取仓库 schema。

### 1.4 任务路由完整性（15）

**通过标准**

- 至少覆盖以下路由：单文件转写、多文件/目录/glob、URL、stdin-list、manifest/resume、LID/语言、context 偏置、timestamps、独立 align、长音频、远程 `--server`、`serve`、本地 vLLM、`stream`、models/doctor、config/jobs/bench、completion、finetune lint。
- 每个路由给出：最小命令、必要前置条件（模型/extra/平台/显式语言）、预期输出、下一步（例如时间戳后生成 SRT、失败后看 `jobs show`）。
- 明确区分：`transcribe --timestamps` 与 `align`；`serve` 与 `--server`；`stream` 与普通 `transcribe`；`bench` 与准确率评测。
- 错误路由完整：缺模型 → `doctor`/`models verify`/`pull`；模型半包 → `verify`；批量部分失败 → `state.jsonl`/`jobs`；退出 5 → coverage/quality；130 → `--resume`。
- 路由表要能让新用户在不读 DESIGN 的情况下选对命令。

**典型反例**

- 只写“用 transcribe 处理音频”，不区分目录、manifest、URL 和 stdin-list 的前置条件。
- 用户已有参考文本，却路由到 `transcribe`；实际应使用 `align`。
- 远程场景推荐 `--timestamps` 或 `align`；实际远程不提供这些能力。
- 把 `bench` 当作识别准确率评测；实际只测吞吐/RTF。
- 把 `models pull` 写成自动发生在首次转写时；实际应显式下载并说明体积。

### 1.5 命令可执行性（12）

**通过标准**

- 所有示例都能在 0.3.0 上复制执行；占位符用明确的大写变量，且不伪装成 flag。
- 参数层级正确：全局 flag 可放在顶层命令前或叶子命令后，但不要放在中间层。例如 `qasr --json models list`、`qasr models list --json` 可用；`qasr models --json list` 退出 2。
- 互斥关系在示例中不冲突：`--format` 不与 `--json/--jsonl` 同用；`--out-dir` 与 `--output` 不同用；`stream` 不用 `--json`。
- 路径含空格时正确引用；stdin 示例使用真实支持的格式；麦克风示例使用 `--input mic`。
- 示例明确平台/extra 前提：vLLM/stream 标 Linux+CUDA；mic 标 `[mic]`；serve 标 `[vllm]`；本地 transformers 标 `[local]`。

**典型反例**

- `qasr transcribe a.wav --out out.srt`：`-o` 是 `--out-dir`，单文件输出应使用 `--output`。
- `qasr transcribe a.wav -f json --json`：文件格式与 stdout JSON 互斥。
- `qasr stream --input mic --json`：`--json` 与 stream 无条件互斥，退出 2。
- `qasr models --json list`：全局/叶子 flag 放错层级，退出 2。
- 在 macOS 示例中直接写 `qasr --backend vllm stream ...`，没有说明 Linux+CUDA 前提。
- 用 `QASR_SERVE_TOKEN=... qasr ...` 作为推荐写法；token 会进入 shell history/argv，应引导用户在交互环境中设置环境变量。

### 1.6 安全边界（15）

**通过标准**

- `QASR_SERVE_TOKEN` 只从环境变量读取；示例不把 token 放进命令参数、配置文件、JSON、日志或状态文件。
- `qasr serve` 没有内建鉴权；skill 必须建议显式 `--host 127.0.0.1`（或反向代理/防火墙），不能暗示“有 token 就能安全绑定 `0.0.0.0`”。
- `models pull` 会下载数 GB 权重并产生供应链/磁盘风险；先说明来源、体积、`--revision`，不要自动执行。
- `models rm` 是删除操作；非 TTY 必须 `--yes`，先 `--dry-run`，只允许受管目录，不能删除用户任意路径。
- `transcribe` 的已有输出默认拒绝覆盖；`--overwrite`/`--resume` 才会替换。skill 不得说“默认静默覆盖”。
- 远程 `--server` 会把音频上传到远端；先说明隐私/网络/服务端信任边界。
- 输出路径、manifest 的 `output` 字段、`--out-dir` 不得指向输入、模型目录、state/lock 文件或其他源文件。
- 示例中不出现真实 token、真实内网地址、真实个人路径或可执行的危险通配符。

**典型反例**

- `QASR_SERVE_TOKEN=secret qasr --server ...` 或 `export` 后写进 skill 示例；token 泄露到 shell history。
- 建议 `qasr serve --host 0.0.0.0` 并称“client token 已保护服务”；实际服务端没有该鉴权。
- 首次转写前自动 `qasr models pull 1.7b`，不提示 4 GB+ 下载和磁盘占用。
- 建议 `qasr models rm 1.7b` 而不说明 `--yes`、只删受管目录、不可恢复。
- 建议 `--overwrite` 清理旧输出；实际可能覆盖用户已有字幕/JSON，且默认本来会拒绝。
- 把 `--server http://gpu-box:8000` 作为默认路径，不说明音频外发。

### 1.7 失败处理（5）

**通过标准**

- 给出完整退出码表，并说明何时看 `error.kind`、`error.retryable`、`hint`、`details`。
- 区分 `3`（模型/运行时）、`4`（输入/能力/路径）、`5`（质量门禁）、`6`（部分失败）、`130`（中断）；不要把 `5` 当“没有产物”，也不要把 `6` 当全部失败。
- 远程 408/429、超时、连接错误按可重试处理；其他 4xx 按输入错误；`--retry` 只重试 retryable。
- `--resume` 依据 job key + 输出存在 + 输出 SHA，不是“目录存在就跳过”；模型/参数变化应重跑。
- `--skip-existing` 只跳过已有且非空输出，不等于结果正确。
- `stream` 启动前错误走普通 envelope；启动后错误是 terminal snapshot；cap 触发退出 5；SIGINT 先发 final 再 130。
- `doctor` 失败时先看 `status`/`checks`，再用 `--strict` 做 CI 门禁。

**典型反例**

- 写“失败就看 stderr 文本”，不解析 `--json` error envelope。
- 把 `--jsonl` 最后一行的 summary 当批次结果；实际 jsonl 不输出 summary，summary 只在 stderr 人类输出。
- 把 `--resume` 说成“跳过所有已存在文件”；实际还要校验 job key/output SHA。
- 把 `--retry` 用于 400/401/403；实际只应重试 retryable 错误。
- 把 `stream` 的 `is_final` 当成“每个快照都可能是 final”；实际最终只应出现一次。

### 1.8 可维护性与版本化（3）

**通过标准**

- 明确写“tested with qasr 0.3.0”；版本变化时先跑 `qasr --version` 和 help 差异。
- 以 `qasr --help`、`schemas/`、README 为单一事实来源；references 不复制会快速漂移的完整 flag 清单。
- 每个 reference 有清晰职责边界和“何时读”；不出现同一事实在多个文件里互相矛盾。
- 提供更新检查清单：新增/删除命令、flag 改名、默认值变化、schema_version、平台/extra 变化。
- 不把临时实验、未交付功能或未来 P2 计划写成当前能力。

**典型反例**

- 写“最新版 qasr 都支持”，没有版本号。
- 在三个 reference 里分别维护三份 flag 列表，升级后互相矛盾。
- 把 DESIGN §12.4 提到的多 GPU、视频抽轨、watch、Windows 写成 0.3.0 已交付；P2-REPORT 已明确这些不在范围。
- 把模型大小、RTF、Python 版本等易变数字硬编码且无“实测日期”。

### 1.9 可评估性（3）

**通过标准**

- 每条关键主张都能落到一个可执行命令、一个 schema 字段或一个退出码。
- 提供固定的 trigger 测试集（中英文正向/负向）和命令验收清单。
- 验收标准是“运行什么、看什么字段、期望什么退出码”，不是“感觉清楚”。
- 能用无 GPU、无模型环境验证的部分明确标注；需要模型/GPU 的部分单独列出，不混在一起。

**典型反例**

- 写“支持稳定 JSON”，但不说明 schema_version、字段位置和错误形态。
- 写“支持长音频”，但不说切块阈值、时间戳语言限制、resume 行为。
- 用“已验证”代替版本、平台、模型和命令。

## 2. Skill 必须覆盖的能力清单

### 2.1 P0：核心单机闭环

以下内容应能在 `SKILL.md` 首屏或核心 reference 中直接找到，并有可执行示例：

| 能力面 | 必须覆盖 |
|---|---|
| 安装与入口 | `qasr --version`；`uv tool install --python 3.12 "qasr-cli[local]"`；可用 `.venv/bin/qasr` 作为项目内 fallback |
| 环境诊断 | `doctor --json` 的 `status`/`checks`；`--strict` 的 0/3 差异；`QASR_HOME` 可写性 |
| 模型 | `models list/path/verify/pull`；`complete/incomplete/missing`；半包 `.aria2`/safetensors 检查；下载体积与 `--revision` |
| 单条转写 | `transcribe a.wav --json`；stdout 文本；`result.text/language/warnings`；空音频成功且 `language:null` |
| 多输入 | 多文件、目录 `--recursive`、glob、URL、`--stdin-list`；多输入需要 `--out-dir`；稳定排序与去重 |
| 输出格式 | `-f txt,json,jsonl,srt,vtt`；`-o/--out-dir`；`--output`；`--format` 与 stdout JSON 互斥；SRT/VTT 需要时间戳 |
| LID/语言 | `--language auto`；30 个可强制语言；52 个 LID 标签/22 个方言；方言不可强制；强制只是提示/标签覆盖 |
| 上下文偏置 | `--context` 可重复；`--context-file`；manifest 逐条 context；远程 context 走 chat |
| 时间戳 | `--timestamps` + 显式 11 语言之一；`--granularity sentence|word|char`；`--language auto --timestamps` 退出 4 |
| 独立对齐 | `align` 的 `--text/--text-file/--srt`；11 语言；coverage；P0 ≤180s；`--server` 拒绝 |
| 批处理/resume | manifest 字段与严格校验；`--out-dir`；`.qasr/state.jsonl`；`--resume/--overwrite`；`--skip-existing`；退出 6 |
| 契约 | `--json` vs `--jsonl` vs `--format`；stdout/stderr 纪律；退出码表；`schema_version` |
| dry-run/config | `--dry-run` 不加载权重；`config show/path`；配置优先级；token 不是配置键 |

### 2.2 P1：运行时、部署与长音频

以下内容至少要在路由表和对应 reference 中覆盖；不需要全部塞进首屏：

| 能力面 | 必须覆盖 |
|---|---|
| 远程客户端 | `--server`/`QASR_SERVE_URL`；`/v1/models` 模型解析；`--jobs` 1–32；`--timeout`/`--retries` |
| 远程能力边界 | 无 timestamps、无 align、无 stream；context 走 chat；transcription 的 language 只做服务端校验 |
| `serve` | `qwen-asr-serve` 薄包装；`[vllm]`；`--host/--port`；未知参数透传；无内建鉴权 |
| 本地 vLLM | `--backend vllm` 仅 Linux+CUDA；`auto` 选择/回退；`--gpu-memory-utilization`；batch 与 timestamps |
| 长音频 | 纯文本默认 270s、时间戳 180s；`--max-chunk-seconds`；`align --window-seconds` 10–180、overlap 0–10；窗口化与单调合并 |
| 长音频限制 | aligner 不超过 180s；跨 chunk 英文粘词 warning；时间戳 canonical ASR 在 resume 时会重跑；coverage 门槛 |
| `align` 自动语言 | `--language auto` 只做脚本判定（Han/Kana/Hangul/Cyrillic）；Latin 及其他脚本要求显式语言，不能把日文汉字判成中文、韩文判成英文 |
| 高级批处理 | `--skip-existing`、`--retry`、`tsv/md`、`jobs list/show`、`bench`；本地 `--jobs` 强制 1 |
| 配置/模型运维 | `config get/set`；`models show/rm`；`rm --yes`/`--dry-run`；符号链接/受管目录限制 |
| 错误映射 | 4xx 与 408/429/5xx 的区分；重试；错误 envelope；`backend_fallback`、`language_not_enforced_on_chat`、`server_ignored_language` 等 warning |

### 2.3 P2：流式与生态（仅 v0.3.0 已交付）

| 能力面 | 必须覆盖 |
|---|---|
| `stream` | 仅本地 vLLM/Linux+CUDA；`--input file|mic|stdin`；stdin `s16le` 16 kHz；`--chunk-ms`/`--realtime`/`--max-stream-seconds` |
| 流式契约 | `--emit jsonl` 累计快照；`seq` 单调；`is_final` 只一次；cap 退出 5；SIGINT 130；与 `--json` 互斥 |
| `completion` | `bash|zsh|fish`，stdout 一个尾换行；只生成脚本，不安装 |
| `finetune lint-jsonl` | 只校验 SFT JSONL；不训练、不构建数据集；`--strict`、`--no-check-files`；退出 4 |
| 打包 | `[local]`/`[vllm]`/`[mic]`/`[all]`；macOS MPS；Linux CUDA；模型目录与 venv 分离 |

**明确不能写成 qasr 能力**：翻译、说话人分离、音频降噪/源分离、视频剪辑/烧字幕、TTS/音色克隆、ASR 训练、自动数据集构建、Windows 原生 vLLM、远程流式、远程时间戳/对齐、多 GPU 分片、视频抽轨、watch。DESIGN §12.4 中这些属于未来 P2 设想，`P2-REPORT.md` 已说明不在 0.3.0 范围。

## 3. 建议触发词

### 3.1 正向：中文

`qasr`、`Qwen3-ASR`、`Qwen3 ASR`、`通义千问语音识别`、`本地语音识别`、`本地 ASR`、`音频转文字`、`语音转文字`、`转写`、`转录`、`听写`、`会议录音转写`、`播客转写`、`访谈转写`、`批量转写`、`目录转写`、`音频字幕`、`SRT`、`VTT`、`时间戳`、`字级时间戳`、`强制对齐`、`文本对齐音频`、`语种识别`、`语言识别`、`LID`、`方言识别`、`粤语识别`、`上下文偏置`、`热词`、`断点续跑`、`批量任务状态`、`远程转写`、`OpenAI 兼容 ASR`、`自建 ASR 服务`、`GPU 转写`、`vLLM ASR`、`流式识别`、`麦克风转写`、`模型校验`、`QASR_HOME`。

### 3.2 正向：英文

`qasr`、`Qwen3-ASR`、`Qwen3 ASR CLI`、`local ASR`、`speech-to-text`、`transcribe audio`、`audio transcript`、`batch transcription`、`directory transcription`、`podcast transcription`、`meeting transcription`、`language identification`、`LID`、`dialect detection`、`Cantonese ASR`、`context biasing`、`hotwords`、`vocabulary bias`、`word timestamps`、`forced alignment`、`SRT/VTT from audio`、`subtitle timing`、`resumable batch`、`partial failure`、`job state`、`remote ASR server`、`OpenAI-compatible transcription`、`self-hosted vLLM ASR`、`GPU transcription`、`streaming ASR`、`microphone transcription`、`model verify`、`qasr doctor`。

### 3.3 负向：不应触发

`TTS`、`语音合成`、`文字转语音`、`配音`、`音色克隆`、`voice clone`、`有声书`、`qtts`、`Qwen3-TTS`、`训练 ASR`、`微调 ASR`、`训练语音模型`、`构建 SFT 数据集`、`说话人分离`、`speaker diarization`、`字幕翻译`、`音频翻译`、`translate audio`、`降噪`、`去伴奏`、`音频源分离`、`视频剪辑`、`烧字幕`、`生成视频`、`LLM 会议总结`、`安装 CUDA 驱动`、`Whisper`/`FunASR` 操作。

### 3.4 边界/歧义场景

| 用户说法 | 正确路由 |
|---|---|
| “给这段视频加字幕” | 先问是否已有音频/是否需要转写；音频转写 + 时间轴可走 qasr，字幕翻译/烧录不是 qasr |
| “把这段录音转成文字并翻译” | qasr 只做转写；翻译由用户另选工具，skill 不承诺 |
| “给这段文字配音” | qtts，不是 qasr |
| “用 Qwen3-ASR 微调一个模型” | 不在 qasr 范围；最多 `finetune lint-jsonl` 校验数据 |
| “识别说话人是谁” | qasr 不做 diarization，明确拒绝 |
| “下载 ASR 模型” | `qasr models pull` 可以，但先说明体积、来源、revision |

## 4. 审核方法

### 4.1 准备与冻结

1. 记录候选 skill 的路径和 SHA-256，记录 `qasr --version`、`python -V`、`uname -a`、`git status`；审核期间不修改候选文件。
2. 先读 `SKILL.md` 的 frontmatter，再读正文和 references；不要先读设计稿的解释，避免被作者措辞带偏。
3. 建立“主张清单”：每条命令、flag、默认值、退出码、JSON 字段、能力边界各占一行，后面附验证证据。

### 4.2 用 `--help` 核对命令面

逐个执行并保存输出：

```bash
qasr --help
for c in transcribe align serve models languages doctor config jobs bench stream completion finetune; do
  qasr "$c" --help
done
qasr models list --help
qasr models path --help
qasr models verify --help
qasr models show --help
qasr models rm --help
qasr models pull --help
qasr config show --help
qasr config path --help
qasr config get --help
qasr config set --help
qasr jobs list --help
qasr jobs show --help
qasr finetune lint-jsonl --help
```

核对项：命令是否存在；flag 是否属于该层级；默认值和 choices；互斥组；`--json/--jsonl/--dry-run` 是否真的可用；示例是否把全局 flag 放错层级。

### 4.3 用安全探针核对退出码

以下探针不下载模型、不跑推理，适合本机/CI：

| 探针 | 期望 |
|---|---|
| `qasr --version` | 0，输出 `qasr 0.3.0` |
| `qasr --json models list` | 0，stdout 一个 JSON 对象 |
| `qasr models list --json` | 0，stdout 一个 JSON 对象 |
| `qasr models --json list` | 2，usage envelope；用于抓 flag 层级错误 |
| `qasr transcribe --timestamps --language auto /tmp/nope.wav --json` | 4，capability；验证时间戳限制 |
| `qasr align --server http://127.0.0.1:9 /tmp/nope.wav --text x --json` | 4，capability；验证远程对齐拒绝 |
| `qasr stream --input mic --json` | 2，usage；验证 stream 与 `--json` 互斥 |
| 无 `sounddevice` 环境：`qasr stream --input mic` | 4，capability；验证 mic extra |
| `qasr doctor --json` | 0（即使 `status=fail`）；`--strict` 在 warn/fail 时 3 |
| `qasr models verify no-such --json` | 3，missing/incomplete |
| `qasr transcribe /tmp/nope.wav --json` | 4，input |
| `qasr finetune lint-jsonl /tmp/nope.jsonl --json` | 4，input |
| `qasr models rm 1.7b --dry-run --json` | 模型在 `QASR_HOME/models` 下时为 0、`dry_run:true`；若模型来自工作区/外部路径则为 2（拒绝非受管目录）；两种情况都不得删除 |
| `qasr serve --dry-run --json` | 可解析模型时 0（只输出 argv，不启动服务）；缺模型时 3 |

注意：`5`（coverage/quality）、`6`（部分失败）、`130`（SIGINT）通常需要模型、音频或 fake server；用仓库 `tests/`、`review_tests*` 的契约测试验证，不要为了审核下载模型。

### 4.4 核对 JSON 契约

1. 用 `--json` 运行 `models list`、`languages`、`config show`、`doctor`、`serve --dry-run`、`transcribe --dry-run`，用 `schemas/*.json` 做 `jsonschema` 校验。
2. 检查 stdout 只有一个 JSON 对象或规定行数；stderr 不混入结果；第三方日志/进度不污染 stdout。
3. 构造错误：缺失输入、未知 flag、能力不支持；确认进程退出码与 `error.code` 一致，`ok:false`，`kind`/`retryable`/`hint` 合理。
4. 特别检查 `align`：`language`/`language_source` 在顶层；`transcribe` 在 `result` 内。skill 若把两者混写，至少 MAJOR。
5. 检查 `stream`：`--emit jsonl` 的每行可独立解析；`seq` 连续；`is_final` 只出现一次；错误行是独立 oneOf 变体。
6. 检查批量：`--json` 是多条 envelope，`--jsonl` 是每项一行且不附 summary；错误行与成功行都可解析。

### 4.5 无 GPU/无模型的验证边界

可以验证：help、版本、config、models list/verify（用现有模型目录或缺失目录）、languages、doctor、dry-run、错误退出码、JSON schema、参数互斥、token 静态审计、输出路径规则、trigger 文本。

不能假装验证：真实转写准确率、MPS 显存峰值、1.7B+aligner 稳定性、vLLM 流式实时性、麦克风采集、远程真实服务。需要这些证据时，明确标注“需模型/GPU/服务器”，不要用 dry-run 代替。

### 4.6 安全静态审计

对候选 skill 全文执行以下检查：

```bash
rg -n "QASR_SERVE_TOKEN|token|secret|password|api[_-]?key" SKILL.md references/
rg -n "0\\.0\\.0\\.0|--host|--port|serve" SKILL.md references/
rg -n "models pull|models rm|--yes|--overwrite|--force|--resume" SKILL.md references/
rg -n "--server|https?://|audio_base64|--stdin-list" SKILL.md references/
```

逐条判断：是否把 token 放进 argv；是否把 `serve` 说成有鉴权；是否自动下载/删除/覆盖；是否说明远端音频外发；是否把输出指向输入或状态文件。

### 4.7 触发测试

1. 在干净会话中只加载候选 skill（或使用等价的 skill 选择机制），不要同时加载 qtts-cli，避免串味。
2. 使用固定 prompt 集：中文正向 15 条、英文正向 15 条、负向 15 条、歧义 8 条；覆盖单文件、目录、manifest、timestamps、align、远程、stream、models、TTS、训练、翻译、diarization。
3. 每条至少跑 2–3 次，记录“是否触发 / 首条建议命令 / 是否加载正确 reference”。
4. 通过线：正向核心 100% 触发；负向高风险（TTS、训练、翻译、说话人分离）0 次误触发；歧义场景必须澄清而不是硬答。
5. 若无法自动测量 skill 选择，至少由两名评审独立按同一 prompt 集判定，并记录分歧。

### 4.8 渐进披露与可维护性检查

1. `wc -lc SKILL.md references/*.md`；对照 token 预算：`SKILL.md` ≤ 10 KB/200 行，单个 reference 建议 ≤ 20 KB。
2. 模拟只读 `SKILL.md`：能否完成“单文件转写 + JSON 输出 + 看退出码”的最小路径。
3. 模拟只读一个 reference：能否完成“远程转写”或“流式”这类单主题任务。
4. 检查每个 reference 是否只有一个职责；同一 flag/默认值是否在多处重复维护。
5. 检查版本声明和更新清单；用 `qasr --version` 与 `--help` 差异验证“tested with 0.3.0”是否真实。

### 4.9 评分与输出

对每个维度打 0–5 分并写一句证据；所有 BLOCKER/MAJOR/MINOR 都附候选 skill 的文件/章节定位、一手证据和修复建议。最终只给出一个结论：`PASS`、`PASS WITH CHANGES` 或 `FAIL`，不要用“基本可用”替代判定。
