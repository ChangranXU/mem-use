---
description: "评审智能体端点、预算与命令行参数。"
icon: sliders
---

# 配置

{% hint style="info" %}
只有 injudge 需要配置，通过 `.env` 文件（memgraph 只用命令行参数）。复制 `.env.example`——完整注释见该文件——并填写；每个键都会被读取和校验，没有无效旋钮。评审是一个智能体，因此旋钮刻意很少：端点三件套、两个预算、采样参数。
{% endhint %}

## 安装

评审智能体是 `agent/` 下的 TypeScript（基于 pi agent 包），由 Python 驱动：

```bash
uv sync                    # Python 侧（tqdm、pytest）
npm --prefix agent install    # TS 侧执行一次（需要 PATH 上有 node >= 22.19）
```

## 端点

| 键 | 默认值 | 用途 |
|---|---|---|
| `MODEL` | 必填 | 评审模型；`--judge-model` 可覆盖 |
| `API_KEY`、`BASE_URL` | 必填 | 评审端点；`OPENAI_API_KEY` / `OPENAI_BASE_URL` 可遮蔽 |
| `API` | `openai-chat` | 线格式：`openai-chat`（任意 OpenAI 兼容端点，`BASE_URL` 指向 `/chat/completions` 根）或 `anthropic`（`BASE_URL` 指向 API 主机根） |
| `TEMPERATURE` | `0.0` | 采样温度 |
| `THINKING_LEVEL` | `off` | 智能体思考层级：`off`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max` |

## 两个预算

两个预算都以**强制作答**收尾——智能体会被要求停止调用工具、基于已收集的证据作答——绝不会在运行中途失败。

| 键 | 默认值 | 用途 |
|---|---|---|
| `MAX_STEPS` | `40` | 首次调查的 LLM 调用上限（含最终作答轮）。每次恢复调查最多 `min(MAX_STEPS, 4)` 次调用；修复仅一次。 |
| `CONTEXT_WINDOW` | `131072` | 评审模型的上下文窗口（token）。水位线（窗口减 `MAX_OUTPUT_TOKENS` 和强制作答消息的估算大小）在上下文溢出前触发强制作答。 |
| `MAX_OUTPUT_TOKENS` | `65536` | 每次 LLM 调用的输出 token 上限。是上限不是预留：按实际生成计费。加上强制作答消息估算后，须在 `CONTEXT_WINDOW` 内为输入留出空间；无效配置在加载轨迹前拒绝。`--judge-max-tokens` 可覆盖。 |

中途被截断的判定不是失败类别：该答案只是无法通过契约解析，纠正重试会继续同一对话重新作答。

## 重试与并行

| 键 | 默认值 | 用途 |
|---|---|---|
| `MAX_ATTEMPTS` | `5` | 每个区间的运行器阶段总数：调查、单调用修复，以及仅在明确请求缺失证据后恢复的调查。修复仍不合法则停止。`--max-attempts` 可覆盖。 |
| `TIMEOUT` | `300` | 单次 LLM 调用超时（秒）。`--timeout` 可覆盖。 |
| `JUDGE_PARALLEL` | `1` | 同一轨迹内并行评判的区间数（上限 32）。每个在飞区间是一条独立的智能体对话。 |

## 命令行参数

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--env PATH` | checkout 的 `.env` | 指定另一份 env 文件 |
| `--judge-model` | `MODEL` | 覆盖评审模型 |
| `--timeout` | `300` | 单次 LLM 调用超时（秒） |
| `--max-attempts` | `MAX_ATTEMPTS` | 每区间的调查/修复阶段总数 |
| `--judge-max-tokens` | `MAX_OUTPUT_TOKENS` | 单次调用输出上限 |
| `--limit N` | 不设 | 最多用智能体评判 *N* 个区间（其余为 `unjudged`）。必须为正。规则判定的 auto-neutral 区间仍会出判定；它们不消耗智能体运行。 |
| `--dry-run` | 关 | 打印每个原本会使用智能体的区间的证据文档、文件索引与开场消息；不发起调用、不写文件 |
| `--golden PATH` | 边车 | 每个智能体评审区间的证据所附 golden answer 正文 |
| `--correctness resolved\|unresolved` | 边车 | 每个智能体评审区间所附的任务结果，仅作佐证 |
| `--gold-metadata PATH` | 不设 | verified-metadata.json：按各轨迹记录的 `instance_id` 连接该实例的 gold patch（优先级介于 `--golden` 与边车之间） |
| `--harness-report PATH` | 不设 | 整个运行共用一份 SWE-bench 风格评测报告，按 `instance_id` 逐轨迹连接（优先级介于 `--correctness` 与边车之间） |
| `-o DIR` | `output/` | 产物容器——每次运行写入自己的 `<run-stamp>/` 子目录；送入智能体的区间会增加 `evidence/` 与 `agent-runs/` 条目 |
| `--html` | 关 | 同时写出 `memgraph-verdicts.html` |

### 离线准备

使用 `--prepare-only` 可在不加载 `.env`、Node 或网络的情况下按 bridge
episode 分区并写出 `injudge.episode-inputs/v1`：

```bash
uv run -m injudge <paths> --prepare-only -o output/
```

该模式支持 `--limit`（限制物化的 episode 数量）以及带键的 gold/harness
附件。`--dry-run`、`--html` 和付费评审参数会被拒绝。若边界或归属需要复核，
仍先写出带诊断的包，再以非零状态退出。

{% hint style="warning" %}
评审 API key 仅通过 `JUDGE_API_KEY` 环境变量传给智能体运行器——绝不经过作业文件或产物——且只发往评审端点。它不得出现在产物、报告、日志或错误信息中。回显了请求的端点输出会在任何文本被保留前先打码。
{% endhint %}
