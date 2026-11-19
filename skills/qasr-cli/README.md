# qasr-cli skill（仓库内源）

本目录是 `qasr` CLI 的 Claude Code skill 源文件，随 CLI 版本（0.3.x）维护。

```
skills/qasr-cli/
├── SKILL.md
├── evals/
│   └── triggers.md
└── references/
    ├── commands-core.md
    ├── batch-and-resume.md
    ├── long-audio-and-stream.md
    ├── deploy-and-remote.md
    ├── ops-and-models.md
    ├── safety.md
    └── troubleshooting.md
```

## 安装（由用户执行）

`~/.agents/skills` 作为实体目标，`~/.claude/skills` 是 Claude Code 的发现入口：

```bash
mkdir -p ~/.agents/skills ~/.claude/skills
ln -sfn "$PWD/skills/qasr-cli" ~/.agents/skills/qasr-cli
ln -sfn ../../.agents/skills/qasr-cli ~/.claude/skills/qasr-cli
```

项目级（在仓库根目录执行）：

```bash
mkdir -p .agents/skills .claude/skills
ln -sfn "$PWD/skills/qasr-cli" .agents/skills/qasr-cli
ln -sfn ../../.agents/skills/qasr-cli .claude/skills/qasr-cli
```

Windows 或不允许符号链接时，复制目录到 `~/.claude/skills/qasr-cli` 亦可；升级时重新复制。

## 触发词评估

`evals/triggers.md` 是设计稿 §10 的 53 条清单（36 条正向核心 + 17 条高风险负向）。
改动 `SKILL.md` 的 description 后，用新会话只给 description 重跑，门槛见该文件。

## 校验

```bash
wc -l SKILL.md references/*.md evals/*.md
qasr --version
qasr --help >/dev/null
for c in transcribe align serve models languages doctor config jobs bench completion finetune; do
  qasr "$c" --help >/dev/null || echo "BROKEN: $c"
done
```

skill 只描述 `qasr 0.3.x` 已实现的能力；CLI 升级后先对照 `qasr <cmd> --help`
与 `qasr --json languages` 更新 `SKILL.md` 与 `references/`。
