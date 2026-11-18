# qasr-cli P1 实现报告

> 任务书：`P1-TASKS.md`（含 2026-11-18 a53859c 澄清）。
> 状态：进行中（N1 已完成，N2–N5 待做）。

## 1. 里程碑对照

| 里程碑 | 状态 | 内容 |
|---|---|---|
| N1 | ✅ 完成 | `serve` 薄包装、全局 `--server`、`doctor --server`、token 边界 |
| N2 | ⏳ 进行中 | 远程客户端、context/chat、并发与错误映射 |
| N3 | ⏳ 待做 | vLLM 后端与 auto 选择 |
| N4 | ⏳ 待做 | 长音频窗口化、`align --language auto` |
| N5 | ⏳ 待做 | 批量运维命令、tsv/md、schema/README |

## 2. 测试命令与结果

```bash
uv run ruff check qasr_cli tests        # All checks passed
uv run pytest -q                        # 217 passed, 6 skipped
```

集成测试（需要本机模型）使用：

```bash
RUN_INTEGRATION=1 QASR_TEST_MODEL_DIR=... pytest -m integration
```

## 3. GPU 实测

待 N2/N3 完成后由协调者安排 gpu06 联调；本机不做 GPU 操作。

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

## 5. 已知限制

- 远程不返回时间戳，也不提供对齐端点（capability error，退出 4）。
- 远程 RTF 含网络时间，仅作记录，不作门禁。
- `QASR_SERVE_TOKEN` 只从环境变量读取；不写入配置、state、jobs 或日志。

## 6. 审查入口

- 假 OpenAI 服务与 serve/doctor 契约测试：`tests/contract/test_serve_doctor.py`
- 远程客户端测试：N2 完成后补充 `tests/contract/test_remote_*.py`
- 长音频纯函数测试：N4 完成后补充 `tests/unit/test_long_audio.py`
- 全量：`uv run pytest -q`；GPU：`RUN_INTEGRATION=1 ... pytest -m integration`
