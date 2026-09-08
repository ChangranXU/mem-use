---
description: "评审端点、评分预算、扫描键与命令行参数。"
icon: sliders
---

# 配置

{% hint style="info" %}
只有 injudge 需要配置，通过 `.env` 文件（memgraph 只用命令行参数）。复制 `.env.example`——完整注释见该文件——并填写；每个键都会被读取和校验，没有无效旋钮。
{% endhint %}

## 端点

| 键 | 默认值 | 用途 |
|---|---|---|
| `MODEL` | 必填 | 评审模型；`--judge-model` 可覆盖 |
| `API_KEY`、`BASE_URL` | 必填 | 评审端点；`OPENAI_API_KEY` / `OPENAI_BASE_URL` 可遮蔽 |
| `API` | `openai-chat` | 线格式：`openai-chat`、`responses` 或 `anthropic` |
| `TEMPERATURE` | `0.0` | 采样温度 |
| `MAX_TOKENS` | `65536` | 评审响应的输出 token 上限（是上限，不是预留）。推理模型会把其中大部分花在 reasoning 上；`--judge-max-tokens` 可覆盖。 |
| `MAX_ATTEMPTS` | `5` | 每区间在工作层级的生成尝试次数。层级发现免费。 |
| `OUTPUT_FORMAT` | 自动 | 固定合规层级：`structured`、`json` 或 `prompt`。不设则从 `structured` 起走链，并记忆首个可用层级。 |
| `THINKING` | 提供方默认 | `enabled` / `disabled`；映射到提供方原生字段 |
| `THINKING_EFFORT` | 不设 | `none` … `max`，按提供方原生透传 |
| `THINKING_BUDGET` | — | 仅 anthropic。`API=anthropic` 且 `THINKING=enabled` 时必填。须小于 `--judge-max-tokens`。 |

## 评分预算与并行

| 键 | 默认值 | 用途 |
|---|---|---|
| `JUDGE_PARALLEL` | `1` | 同一轨迹内并行评判的区间数（上限 32）。在飞调用数可达 `JUDGE_PARALLEL × SCAN_PARALLEL`。 |
| `JUDGE_INPUT_TOKENS` | `800000` | 评分请求的 token 上限，按长上下文评审模型（约 1M token 窗口减去输出余量）设定。低于它单次全文调用；超过则走分块证据扫描。 |


默认输入预算在 1M-token 窗口内为 65,536-token 输出上限、请求/schema 开销和分词器差异留出余量。它只是估算，并非端点能力探测：小窗口模型应调低，部署端点允许时可调高。投递原文、全部查询、任务、gold 和完整步骤流均未经预览截断即计入预算。单块证据不再有独立 token 限制；只有组装后的评分请求超预算时才做一次压缩。`MAX_TOKENS` 和 `SCAN_MAX_TOKENS` 限制生成输出，应为推理和全部条目保留足够空间；输出截断明确记为错误，绝不判为中性。

## 扫描（超预算路径）

这些键在逐字评分输入超过 `JUDGE_INPUT_TOKENS` 时生效。提取运行在评分配置的**副本**上，因此提取过程中学到的层级不会泄漏到评分。

| 键 | 默认值 | 用途 |
|---|---|---|
| `SCAN_PARALLEL` | `4` | 提取并发数（上限 32；`1` 为全串行）。评分始终为单次调用。 |
| `SCAN_CHUNK_STEPS` | `0` | 可选的单块步数上限；`0` 表示仅按 token 分块。 |
| `SCAN_CHUNK_TOKENS` | `输入预算的一半` | 单个提取请求 user 体的 token 上限，无额外固定上限。显式设置时须小于 `JUDGE_INPUT_TOKENS`；提取端点窗口较小时需相应调低。 |
| `SCAN_NEIGHBORHOOD` | `2` | 评分输入中注入步骤两侧各保留的 verbatim 步数（`0` 表示只保留该步） |
| `SCAN_MODEL` · `SCAN_API` · `SCAN_BASE_URL` · `SCAN_API_KEY` | 继承 | 提取端点覆盖键（`SCAN_OPENAI_BASE_URL` / `SCAN_OPENAI_API_KEY` 可遮蔽）。不设则继承评分值。 |
| `SCAN_THINKING` | `disabled` | 唯一不继承的覆盖键：提取默认不消耗推理预算，显式开启才启用。 |
| `SCAN_MAX_TOKENS` · `SCAN_TEMPERATURE` · `SCAN_MAX_ATTEMPTS` · `SCAN_THINKING_EFFORT` · `SCAN_THINKING_BUDGET` · `SCAN_OUTPUT_FORMAT` | 继承 | 其余提取覆盖键，与评分键一一对应。 |

## 命令行参数

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--env PATH` | checkout 的 `.env` | 指定另一份 env 文件 |
| `--judge-model` | `MODEL` | 覆盖评审模型 |
| `--timeout` | `300` | 请求超时（秒） |
| `--max-attempts` | `MAX_ATTEMPTS` | 每区间在工作层级的生成尝试次数 |
| `--judge-max-tokens` | `MAX_TOKENS` | 响应上限，不是预留。推理模型会把其中大部分花在 reasoning 上。 |
| `--limit N` | 不设 | 最多用 LLM 评判 *N* 个区间（其余为 `unjudged`）。必须为正。规则判定的 auto-neutral 区间仍会出判定；它们不消耗调用。 |
| `--dry-run` | 关 | 打印评审输入；不发起调用、不写文件 |
| `--golden PATH` | 边车 | 每次评分调用附上的 golden answer 正文 |
| `--correctness resolved\|unresolved` | 边车 | 已记录的任务结果，仅作佐证 |
| `--gold-metadata PATH` | 不设 | verified-metadata.json：按各轨迹记录的 `instance_id` 连接该实例的 gold patch（优先级介于 `--golden` 与边车之间） |
| `--harness-report PATH` | 不设 | 整个运行共用一份 SWE-bench 风格评测报告，按 `instance_id` 逐轨迹连接（优先级介于 `--correctness` 与边车之间） |
| `-o DIR` | `output/` | 产物目录 |
| `--html` | 关 | 同时写出 `memgraph-verdicts.html` |

{% hint style="warning" %}
评审 API key 只发往评审端点，不得出现在产物、报告、日志或错误信息中。回显了请求的端点输出会在任何文本被保留前先打码。
{% endhint %}
