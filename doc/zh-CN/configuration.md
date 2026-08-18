# 配置

injudge 的评审端点经 `.env` 配置（复制 `.env.example` 为 `.env` 并填写；
完整注释见该文件）。每个键都会被读取和校验——没有摆设。

| 键 | 默认值 | 用途 |
|---|---|---|
| `MODEL` | 必填 | 评审模型；`--judge-model` 可覆盖 |
| `API_KEY` / `BASE_URL` | 必填 | 评审端点；`OPENAI_API_KEY` / `OPENAI_BASE_URL` 可遮蔽 |
| `API` | `openai-chat` | 线格式：`openai-chat` · `responses` · `anthropic` |
| `TEMPERATURE` | `0.0` | 采样温度 |
| `MAX_ATTEMPTS` | `5` | 每区间在工作层级的生成尝试预算；层级发现免费 |
| `OUTPUT_FORMAT` | 自动 | 固定合规层级 `structured` / `json` / `prompt`；不设则从 `structured` 自动降级并记忆可用层级 |
| `THINKING` | 提供方默认 | `enabled` / `disabled`；映射到提供方原生字段 |
| `THINKING_EFFORT` | 不设 | `none` … `max`，按提供方原生透传 |
| `THINKING_BUDGET` | — | 仅 anthropic；`API=anthropic` 且 `THINKING=enabled` 时必填，须小于 `--judge-max-tokens` |
| `JUDGE_PARALLEL` | `1` | 同一轨迹内并行评判的区间数（上限 32）；在飞调用数可达 `JUDGE_PARALLEL` × `SCAN_PARALLEL` |
| `JUDGE_INPUT_TOKENS` | `50000` | 评分请求的 token 上限：低于它单次全文调用；超过则走分块证据扫描 |
| `SCAN_PARALLEL` | `4` | 提取并发数（上限 32；`1` 为全串行）；评分始终为单次调用 |
| `SCAN_CHUNK_STEPS` | `8` | 每个提取块的最大 agent 步数 |
| `SCAN_CHUNK_TOKENS` | min(`20000`, 预算的一半) | 单个提取请求 user 体的 token 上限；不设时从 `JUDGE_INPUT_TOKENS` 派生，显式设置时须小于它 |
| `SCAN_NEIGHBORHOOD` | `2` | 评分输入中注入步骤两侧各保留的 verbatim 步数（`0` = 仅注入步本身） |
| `SCAN_EVIDENCE_TOKENS` | `4096` | 单块返回证据项的 token 提示上限；超限时做一次摘要压缩，绝不截断 |
| `SCAN_MODEL` · `SCAN_API` · `SCAN_BASE_URL` · `SCAN_API_KEY` | 继承 | 提取端点覆盖键（`SCAN_OPENAI_BASE_URL` / `SCAN_OPENAI_API_KEY` 可遮蔽）；不设则继承评分值 |
| `SCAN_THINKING` | `disabled` | 唯一不继承的覆盖键：提取默认不烧推理预算，显式 enabled 才开启 |
| `SCAN_MAX_TOKENS` · `SCAN_TEMPERATURE` · `SCAN_MAX_ATTEMPTS` · `SCAN_THINKING_EFFORT` · `SCAN_THINKING_BUDGET` · `SCAN_OUTPUT_FORMAT` | 继承 | 其余提取覆盖键，与评分键一一对应 |

## 命令行参数

`--timeout`（默认 120 秒）、`--max-attempts`、`--judge-max-tokens`
（默认 24000——是上限
不是预留；推理模型会把它的大部分花在 reasoning 上）、`--limit N`（必须为
正整数——0 或负数直接报错）、`--env PATH`、`--golden PATH`、
`--correctness resolved|unresolved`。

评审 API key 只发往评审端点，绝不出现在产物、报告、日志或错误信息中
——回显了请求的端点输出会在任何文本被保留前先打码。
