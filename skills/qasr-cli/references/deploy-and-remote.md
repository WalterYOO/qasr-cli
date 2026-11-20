何时读我：要起 `qasr serve`、用 `--server` 连远程、启用本地 vLLM，或排查 GPU/鉴权/网络问题时。

# 部署与远程

版本基线 `qasr 0.3.3`。三条边界先记住：

- `serve` 只在 Linux + CUDA + `[vllm]` 下真正可用；`serve --dry-run` 跨平台但只证明 argv。
- 远程只做文本/LID/context/批量；**没有**时间戳、`align`、`stream`。
- `qasr serve` **没有内建鉴权**；`QASR_SERVE_TOKEN` 只被客户端读取，不会被服务端校验。

## serve

### 前置检查

```bash
qasr --json models show 1.7b | jq -e '.status == "complete"'
qasr --json models verify 1.7b | jq -e '.status == "complete"'
qasr --json serve 1.7b --dry-run | jq -e '.source == "local" and .path != null'
```

- `models show` 的 `.status` 必须是 `complete`；`present:true` 不够。
- `serve --dry-run` 的 `.source` 为 `local` 且 `.path` 非 null 才安全；`repo` 表示本地缺失/半包，
  `qwen-asr-serve` 启动后会去下载。未知名字也可能返回 `source:"local"` + `path:null`，同样不可用。
  **`--offline` 不约束这个子进程**。
- `source:"repo"` 时停下，向用户说明会下载数 GB 权重，得到确认后再继续（通常先 `models pull`）。
- `serve` 需要 `qwen-asr-serve` 可执行文件；缺失是 3。
- 只能 serve ASR 模型（`1.7b`/`0.6b`/ASR 路径）；`aligner` 会 2。

### 启动

```bash
tmux new-session -d -s qasr-serve 'qasr serve 1.7b --host 127.0.0.1 --port 18000'
tmux capture-pane -pt qasr-serve | tail -20
tmux attach -t qasr-serve
```

- `MODEL` 必须在任何未知的 `qwen-asr-serve` flag 之前；未知 flag 会原样透传。
- `qasr serve` 不注入 `--host`；上游 vLLM 默认绑定 `0.0.0.0`。
  **必须显式 `--host 127.0.0.1`**，除非用户明确要求对外且已有防火墙/反代/TLS。
- `--gpu-memory-utilization 0.3–0.6` 可传给服务端；共享 GPU 不要用默认占满。
- 进程用 `execvp` 替换为 server，故 `qasr serve` 的退出码就是 server 的退出码。
- `serve` 是常驻进程，只允许 tmux；不要 `nohup`/裸 `&`。

### 鉴权与暴露

- `qasr serve` 本身不校验 token；不要把“客户端带了 token”当成服务端已安全。
- 对外暴露必须自己加一层：防火墙、反向代理、TLS、网络 ACL；否则保持 loopback。
- 客户端 token 由用户在交互 shell 设置：

```bash
read -rs QASR_SERVE_TOKEN; export QASR_SERVE_TOKEN
```

agent 不读取、不打印、不代填；不要把 token 写进 argv、URL userinfo、config、state 或日志。

## 远程客户端（--server）

```bash
qasr --json --server http://127.0.0.1:18000 transcribe a.wav --jobs 4
qasr --json --server http://127.0.0.1:18000 transcribe a.wav --context "Qwen3"
```

- `--server URL` 隐含 `--backend remote`；也可用 `QASR_SERVE_URL`。
- `--server` 与显式 `--backend transformers|vllm` 冲突（2）；不要同时给。
- `QASR_SERVE_TOKEN` 存在时作为 Bearer 头发送；仅从环境读取。
- `--jobs` 1–32，默认 4；`--batch-size` 在远程恒为 1（>1 只 warning）。
- `--timeout` 默认 300 s；`--retries` 默认 2。408/429/5xx/连接失败可重试，
  4xx（含 401/403）是输入/凭据错误，不要盲目重试。`Retry-After` 会进 `details`。
- 单文件上传上限 25 MB；超出是 4。`--max-chunk-seconds >600` 会降为 600 并 warning。
- 不传 `--model` 时，客户端读 `/v1/models` 的第一个 id；服务端有多个模型时会 warning。
  传了 `--model` 则按 alias/repo/文件名匹配；JSON `.model.id` 是实际使用的 id。
- 远程能力：文本、LID、context、批量。以下在发请求前直接 4：
  `--timestamps`、`--granularity`（非 `sentence`）、`align`、`stream`。
- `--context` 走 `/v1/chat/completions`，`-l` 不保证生效。读 `.result.warnings[]`：
  `language_not_enforced_on_chat`、`server_ignored_language`。
- 音频会上传到远端；先说明隐私、网络与服务端信任边界，由用户确认。

### 远程排错

```bash
qasr --json doctor | jq '.checks[] | select(.name == "backend")'
qasr --json --server http://127.0.0.1:18000 transcribe a.wav --dry-run   # 只验证参数，不联网
```

- 连接失败/超时：先增大 `--timeout`，再降 `--jobs`；不要先加 `--retries`。
- 401/403：检查用户 shell 的 `QASR_SERVE_TOKEN` 与服务端配置，不要把 token 贴进命令。
- 404/400：检查 URL 前缀、服务端 served-model-name、音频格式/大小。

## 本地 vLLM

```bash
qasr --json --backend vllm transcribe a.wav --batch-size 8
qasr --json --backend vllm --gpu-memory-utilization 0.5 transcribe a.wav
```

- 仅 Linux + CUDA + `qasr-cli[vllm]`；其他平台显式 `--backend vllm`：
  `transcribe` → 3，`stream` → 4；`align` 忽略 vllm（始终 transformers；remote 仍 4），不因平台 3。
- `--backend auto`：Linux+CUDA 且 vllm 可导入 → vllm；否则 transformers（含 macOS/MPS/CPU）。
  探测到 vLLM 但导入失败会回退，JSON 顶层 `backend_fallback:true`，并带 warning。
- `--batch-size` 对 vLLM 默认 8；`--max-inference-batch-size` 是模型内部批，默认等于 batch。
- `--gpu-memory-utilization` 只允许 vLLM，取值 `(0,1]`；给其他后端是 2（不是静默忽略）。
  共享 GPU 用 0.3–0.6；这是 vLLM 预留比例，不是硬上限。
- `align` 的 aligner 与长音频 anchor ASR **始终走 transformers**；`--backend vllm` 对 `align`
  不生效，也不会因平台返回 3。transcribe 的时间戳仍由 transformers aligner 完成；`stream` 必须 vLLM。
- `--device cuda:N` 会在 import torch 前设 `CUDA_VISIBLE_DEVICES`；
  `--device mps` 配 vLLM 会失败。`dtype=auto`：CUDA→bf16、MPS→fp16、CPU→fp32。
- vLLM 对 `transformers==4.57.6`、`vllm==0.14.0` 有钉死要求；用 `uv tool` 隔离，
  不要把 `[vllm]` 装进 macOS 主环境。

## 快速决策

| 需求 | 选型 |
|---|---|
| 本机少量文件、要时间戳 | transformers（`[local]`，`backend=auto`） |
| Linux GPU 高吞吐、要时间戳 | vLLM 跑 ASR + transformers aligner |
| 多机/客户端只想取文本 | `--server` 远程 |
| 要实时/流式 | 仅 Linux+CUDA+`[vllm]` 的 `stream` |
| 远程要时间戳/对齐 | 不可能；改本地 |
