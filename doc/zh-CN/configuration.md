# 配置

[文档目录](README.md) · [English](../en/configuration.md)

只有实际运行 injudge 评审时才需要 `.env`。memgraph、`--dry-run` 和
`--prepare-only` 不读取评审凭证。复制仓库根目录的 `.env.example` 为 `.env`，
填写 `MODEL`、`API_KEY`、`BASE_URL`；其余设置按需修改，默认值见下表。

## 安装

评审智能体是 `agent/` 下的 TypeScript（基于 pi agent 包），由 Python 驱动：

在仓库根目录执行：

```bash
uv sync                    # Python 侧（tqdm、pytest）
npm --prefix agent install    # TS 侧执行一次（需要 PATH 上有 node >= 22.19）
```

## 端点

| 键 | 默认值 | 用途 |
|---|---|---|
| `MODEL` | 必填 | 评审模型；`--judge-model` 可覆盖 |
| `API_KEY`、`BASE_URL` | 必填 | 评审端点；`OPENAI_API_KEY` / `OPENAI_BASE_URL` 可遮蔽 |
| `API` | `openai-chat` | API 格式：`openai-chat`（任意 OpenAI 兼容端点，`BASE_URL` 指向提供 `/chat/completions` 的 API 根地址）或 `anthropic`（`BASE_URL` 指向 API 主机根） |
| `TEMPERATURE` | `0.0` | 采样温度 |
| `THINKING_LEVEL` | `off` | 智能体思考层级：`off`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max` |

思考层级是请求，并不保证各提供方具有不同的推理预算。当前 pi 对自定义模型将 `xhigh` 和 `max` 限定为 `high`；[DeepSeek](https://api-docs.deepseek.com/guides/thinking_mode/) 也将 `medium` 和 `high` 映射为同一强度，因此比较这两种设置不能检验不同推理预算。`off` 经提供方兼容层明确请求禁用思考。

## 两个预算

对话开始后，任一预算都可以触发基于已读证据的**强制作答**。首次付费调用前，若开场消息或必读证据超过估算的输入预算，运行器就会拒绝该区间；见[评审智能体](judge-client.md#预算与强制作答)。

| 键 | 默认值 | 用途 |
|---|---|---|
| `MAX_STEPS` | `40` | 首次调查的 LLM 调用上限（含最终作答轮）。答案被拒后的每次跟进最多 `min(MAX_STEPS, 4)` 次调用。 |
| `CONTEXT_WINDOW` | `131072` | 评审模型的上下文窗口（token）。水位线扣除 pi 的上下文预留、`MAX_OUTPUT_TOKENS` 和强制作答消息的估算大小，再触发最终作答。 |
| `MAX_OUTPUT_TOKENS` | `32768` | 每次 LLM 调用的输出 token 上限。上下文水位线会为输出预留这部分空间，因此输入证据的可用空间相应减少；计费仍按实际生成量。加上 pi 的上下文预留和强制作答消息估算后，须在 `CONTEXT_WINDOW` 内为输入留出空间；无效设置或超限的必读输入在付费调用前拒绝。`--judge-max-tokens` 可覆盖。 |

中途被截断的判定不是失败类别：该答案只是无法通过契约解析，纠正重试会继续同一对话重新作答。

## 重试与并行

| 键 | 默认值 | 用途 |
|---|---|---|
| `MAX_ATTEMPTS` | `3` | 每个区间的运行器阶段总数：首次调查，加上每次答案被拒后的跟进。`--max-attempts` 可覆盖。 |
| `TIMEOUT` | `300` | 等待响应头或下一个正文块的最大空闲秒数。持续返回数据的流可以运行更久，不设总耗时上限。每次重试重新计时。`--timeout` 可覆盖。 |
| `JUDGE_PARALLEL` | `1` | 跨所有输入轨迹并行评判的区间数（上限 32）。每个在飞区间是一条独立的智能体对话。 |

## 命令行参数

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--env PATH` | checkout 的 `.env` | 指定另一份 env 文件 |
| `--judge-model` | `MODEL` | 覆盖评审模型 |
| `--timeout` | `300` | 等待响应头或下一个正文块的空闲超时秒数 |
| `--max-attempts` | `MAX_ATTEMPTS` | 每区间的运行器阶段总数（调查加跟进） |
| `--judge-max-tokens` | `MAX_OUTPUT_TOKENS` | 单次调用输出上限 |
| `--episode ID` | 不设 | 只评判已录制 episode id、benchmark 身份或执行 id 等于 *ID* 的 episode（可重复）。其余区间保留紧凑的 `unjudged` 审计记录，不计入汇总统计、summary 计数和 dry-run 区间计数。policy 总量仍覆盖完整轨迹。无匹配的 *ID* 在任何调用前快速失败。不可与 `--prepare-only` 同用。 |
| `--limit N` | 不设 | 最多用智能体评判 *N* 个合格区间；机械判定的区间不占此额度；先校验全部 episode 与附件的适用范围。`--prepare-only` 则改为限制物化的 episode 数量。小于 1 的值会快速失败。 |
| `--prepare-only` | 关 | 写出离线 episode 证据包；不加载评审配置或 Node，不调用模型。见[离线准备](#离线准备)。 |
| `--dry-run` | 关 | 打印每个付费区间任务的目标文档、文件索引与开场消息；不发起调用、不写文件。不可与 `--html` 同用。 |
| `--golden PATH` | 不设 | 只适用于恰好一个 episode 的 golden answer；其他情况请用带键的元数据 |
| `--correctness resolved\|unresolved` | 不设 | 只适用于恰好一个 episode 的已记录结果，仅作佐证 |
| `--gold-metadata PATH` | 不设 | verified-metadata.json：按已记录的 episode benchmark 身份连接 gold patch |
| `--harness-report PATH` | 不设 | 按 episode 身份连接的 harness 结果；缺少执行级溯源时，重复执行或不同实验组会被拒绝 |
| `-o DIR` | `output/` | 产物容器——每次运行写入自己的 `<run-stamp>/` 子目录；送入智能体的区间会增加 `evidence/` 与 `agent-runs/` 条目 |
| `--html` | 关 | 同时写出 `memgraph-verdicts.html` |

### 离线准备

使用 `--prepare-only` 可在不加载 `.env`、Node 或网络的情况下按 bridge
episode 分区并写出 `injudge.episode-inputs/v5`。将 `path/to/recordings` 替换为实际录制路径：

```bash
uv run -m injudge path/to/recordings --prepare-only -o output/
```

该模式接受 `--limit`（按物化的 episode 计数）、两个单 episode 附件
（`--golden` / `--correctness`），以及带键的 gold 与 harness 附件选项。

`--dry-run`、`--html` 和评审参数（`--env`、`--judge-model`、`--timeout`、
`--max-attempts`、`--judge-max-tokens`）会被拒绝。范围、归属、episode 证据或
相互冲突的结果附件需要复核，或存在 error 级诊断时，先写出包，再以状态 2 退出。

普通警告本身不导致失败；退出状态由这些实际条件决定，不取决于警告文字或输入标识符。
`--limit` 只限制物化数量，不会隐藏输入中其他 episode 的问题。

评审 API key 不得出现在产物、报告、日志或错误信息中；[凭证](judge-client.md#凭证)说明了它如何传给运行器，以及回显了请求的端点输出如何被打码。
