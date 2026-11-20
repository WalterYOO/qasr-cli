何时读我：准备覆盖/删除文件、下载模型、把音频发到远程、启动服务或跑长任务之前。

# 安全与破坏性操作

版本基线 `qasr 0.3.2`。原则：**联网、外发、覆盖、删除、持久化配置都必须由用户决定**；
agent 只负责把计划和影响摆清楚。

## 判定表

| 操作 | 默认行为 | 必须做什么 |
|---|---|---|
| `transcribe -o` / `--output` | 目标存在 → exit 4，拒绝覆盖 | 先说明冲突；用户同意后才 `--overwrite`；批量优先 `--resume`/`--skip-existing` |
| `align -o` / `--output` | **静默覆盖**，没有 `--overwrite` | 先 `--dry-run` 看 `.plan.outputs[].path`；确认或换新目录 |
| `models rm` | 只删受管目录；非 TTY 需 `--yes` | 先 `models show` + `rm --dry-run`，确认后再 `--yes` |
| `config set` | 立即写盘；`--dry-run` 被接受但忽略 | 先 `config path`/`config show` 给用户看落点；不要用 `--dry-run` 预演；项目级用 `--project` |
| `models pull` | 下载数 GB 权重，联网 | 说明 repo/revision/来源/体积；用户同意后执行 |
| `--server` | 上传音频到远端 | 说明外发内容、目标端、隐私边界；用户同意后执行 |
| URL 输入 | 下载到 `QASR_HOME/cache/url`，联网 | 确认来源可信、大小（≤2 GiB）与网络可达 |
| `serve` | 常驻进程；缺失模型可能下载 | 先证明 `models show/verify` 为 `complete`；`source:"repo"` 时停下确认 |
| `serve --host` | qasr 不注入 host，上游默认 `0.0.0.0` | 默认显式 `--host 127.0.0.1`；对外必须另有防火墙/反代/TLS |
| 长任务（>~10 min） | — | 只用 tmux；禁止 `nohup` 与裸 `&` |
| token | 仅从环境读，客户端使用 | 用户自己在交互 shell 设置；agent 不读/不打印/不代填 |

## Token

- `QASR_SERVE_TOKEN` 只从环境变量读取；`qasr serve` 本身**不校验**它。
- 用户在自己的交互 shell 执行：

```bash
read -rs QASR_SERVE_TOKEN; export QASR_SERVE_TOKEN
```

- 禁止：命令行前缀赋值（`QASR_SERVE_TOKEN=... qasr ...`）、URL userinfo
  （`https://token@host`）、argv、config、JSON、state、日志、README/示例。
- 不要把“客户端带了 token”说成“服务端已鉴权”；对外暴露必须另加网络层保护。

## 下载与外发

- `transcribe`/`align` 不会自动下载模型；缺失/半包是 3。
- `serve` 是例外：本地缺失/半包会回退 repo id，`qwen-asr-serve` 启动后可能下载；
  `--offline` 不约束这个子进程。serve 前必须 `complete` + `source:"local"` 且 `path` 非 null。
- `models pull`、`--server`、URL 输入都会联网；先说明来源、体积、目标端。
- 远程会收到音频内容与（可能）context；不要把敏感音频默认发到不可信服务。
- 下载来源优先固定 `--revision`；半包用 `pull --resume`，不要手工拼目录。

## 覆盖与删除

- `transcribe` 的覆盖冲突是在**写盘阶段**报 4；`--dry-run` 不检查目标是否存在。
  因此计划阶段只能看 `.plan.outputs[]`，不能保证写入时无冲突。
- `align` 没有任何覆盖开关；默认 `atomic_write_bytes` 直接替换。重复运行前必须确认路径。
- `--resume` 会允许替换该 job 的旧输出；它的安全性来自 job key + 输出 SHA 校验，不是“不会写”。
- `--overwrite` 会无条件替换；不要在用户未确认时使用。
- `models rm` 只删 `QASR_HOME/models`（或 `--model-dir`）下的一级受管目录；
  符号链接、嵌套目录、`..`、dev path 都会被拒绝。删除不可恢复。
- `config set` 立即持久化；写错 backend/device 会影响后续所有命令。先展示 `config path`。

## 长任务

只允许 tmux（远程服务器也一样）：

```bash
tmux new-session -d -s qasr-job \
  'qasr --jsonl transcribe ./audio -o out/ --resume > qasr.jsonl 2> qasr.err'
tmux capture-pane -pt qasr-job | tail -20
tmux attach -t qasr-job
```

- `serve` 是常驻进程，必须在 tmux 内跑。
- 中断（130）后先看 `<out-dir>/.qasr/state.jsonl` 与 `jobs show`，再决定 `--resume`；
  不要重复提交同一批任务。
- 远程长任务同样放 tmux；重连后用 `jobs show` 恢复上下文。

## 路径与文件

- 输出路径不要指向输入文件、模型目录、`<out-dir>/.qasr/` 或其他源文件。
- manifest 的 `output` 必须是相对 `--out-dir` 的路径，不能绝对、不能含 `..`。
- 不要修改 `<out-dir>/.qasr/` 下的 state/manifest/lock；锁冲突时先确认进程是否还活着。
- 目录输入会镜像相对子路径；跑前用 `--dry-run` 看 `.plan.outputs[]` 是否落在预期位置。

## 执行边界

- 不替用户执行：`models pull`、`models rm`、`serve` 公网绑定、`finetune` 训练（本 CLI 也没有训练）。
- 不自动下载模型来“让命令成功”；缺模型就报告 3 并给 `models pull` 建议。
- 不把 `--offline` 当作对所有子进程的保证（尤其 serve）。
- 不在未确认时改配置、删输出、覆盖字幕、上传音频。
- 不把真实 token、内网地址、个人路径写进示例或报告。
