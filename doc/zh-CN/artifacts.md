# 产物

injudge 的所有产物写入输出目录（`-o DIR`，默认 `output/`）：

- `injudge-report.json` — `injudge.report/v5` 运行时报告，每条轨迹完成
  即重写，崩溃也保留已完成部分：逐区间的判定与评分信号、分析、证据、
  投递成本，每个操作的 `input` 记录（预算门路径 `full`/`scan` 及其
  扫描计数、gold 规则体系与超预算原样发送标记）、轨迹投递策略与汇总
  指标（含 gate 区间上的 `input_paths` 与 `gold_scored` 计数）。可用
  memgraph 的 `--verdicts` 回灌。
- `summary.json` — `injudge.summary/v2` 最终统计集合：计数、比率、评分信号
  分布、成本、置信度、精简的逐记忆使用统计（按轨迹与整体），外加无叙述
  的区间索引（判定 / 置信度 / 使用标签）。评判叙述与过程日志只保留在
  report 产物中。
- `memgraph-verdicts.html` — 仅在 `--html` 时产出（默认关闭）：召回节点
  按区间判定着色的 memgraph 可视化。

memgraph 自己的 `--json` 产物是 `{tool, trajectories, ops, memories,
edges, validation}`——与 HTML 内嵌的同一个 dict。

## 读懂 summary.json

`summary.json` 是一次运行的最终统计集合——供阅读和比较记忆系统的数字。
（评判叙述——评分 note、分析、证据、`judge_raw`、validation 日志——只保留在
`injudge-report.json` 里；区间的 `operation_id` 即指向它的索引。）计数规则：
比率只覆盖**已评判区间**（分母 = positive + negative + neutral）；`error` 与
`unjudged` 只计数、永不进分母；规则判定的 auto-neutral 区间计为 neutral；
重述的重复投递并入其区间、从不出现在比率中。

顶层：`tool.schema`（产物版本）、`generated_at`、`overall`（全部轨迹合并）、
`trajectories`（逐轨迹块，含同样的统计外加区间索引）。每条轨迹在有记录时
带上 `instance_id` / `model` / `adapter`。没有已评判样本的块会省略空的
rate / headline / rubric / confidence / cost 壳（仅 error 或零注入的轨迹
仍保留判定计数）。

每个聚合块（`overall` 与每条轨迹）：

| 字段 | 含义 |
|---|---|
| `n_injections` | 投递的召回注入数，含重述的重复投递 |
| `n_spans` | 注入区间数——评审单元（相同投递的最大连续段） |
| `restated_injections` | 相同内容的重复投递（计数，但不进比率） |
| `positive` / `negative` / `neutral` | 区间判定：注入的足迹改善 / 损害 / 未体现在工作中 |
| `error` / `unjudged` | 评审失败的区间 / 被 `--limit` 截掉的区间 |
| `rates` | 已评判区间上的判定占比（无评判时省略） |
| `headline.judged_spans` | 比率的分母（为 0 时整个 `headline` 省略） |
| `headline.wasted_token_share` | neutral+negative 区间的注入 token 成本占已评判区间成本的比例——花在没有帮助的注入上的预算份额 |
| `rubric.deliveries` | 被打分的记忆投递数（区间 × 记忆 key）；为 0 时整个 `rubric` 省略 |
| `rubric.counts` / `rubric.rates` | 已评判投递上各信号的取值分布与投影比率；null 信号永不稀释比率。比率口径：`relevance` = 相关占比；`accuracy` = 未被证伪及以上占比；`novelty` = 非冗余占比；`influence` = 有任何足迹的占比；`positive_effect` = 仅在有影响的投递中 improved 的占比（无影响时按契约为 null） |
| `rubric.no_footprint_reasons` | 被忽略记忆未留足迹的原因：`irrelevant` / `redundant` / `confirming_ignored` / `ignored_novel`（空时省略） |
| `rubric.inaccurate_keys` / `irrelevant_keys` | 排查清单 `{key, spans}`——评审认定被证伪 / 不相关的记忆；仅在非空时写出 |
| `cost` | 确定性的投递成本（不经评审）：`content_chars_total`、`approx_tokens_total`（chars/4 估算）、按判定类别归集的 `approx_tokens_by_verdict`（为零时省略） |
| `confidence.mean_by_verdict` / `confidence.buckets` | 评审置信度：按判定类别的均值（省略 null 类）与 `[0,0.5)` / `[0.5,0.8)` / `[0.8,1.0]` 分桶计数，覆盖已评判区间（没有任何评分时整个 `confidence` 省略） |
| `input_paths` | 仅 overall：预算门（`JUDGE_INPUT_TOKENS`）后走全文 / 分块扫描输入的 gate 区间数 |
| `gold_scored` | 仅 overall：在 gold 评分规则下评判的区间数（附了 golden answer） |
| `over_budget` | 仅 overall：评分输入超过 `JUDGE_INPUT_TOKENS` 预算仍原样发送的区间数 |
| `n_trajectories` | 仅 overall：本次运行的轨迹数 |
| `memory_usage` | 逐记忆使用统计：`key` + 非零 `tallies`；记录内容跨区间变化时加 `value_drift: true`。标签：`used` = 被采纳且改善；`misused` = 被采纳但无效果或有害；`noted` = 仅被提及；`ignored` = 无足迹；`unclear` = 信息不足 |
| `low_confidence` | 置信度 < 0.5 的区间 id（仅在非空时写出） |
| `policy` | 投递策略：`n_injections`、`n_steps`、`injection_density`（每 agent 步的注入数）、`restated_ratio`、`est_injected_tokens`、`prompt_tokens_total`（计费用量）、`injected_token_share`、`n_searches`；overall 还带 `est_injected_tokens_billed`——仅部分轨迹有计费记录时，份额的分子取它 |

`rubric.counts` 中出现的信号枚举值即 [injudge](injudge.md#评分信号)
评分信号表中的值。当多个身份共用同一显示键时，summary 的 `key` 字段
（排查清单与 `memory_usage`）会带上 `key#<item 或 namespace>` 后缀——取
8 字符前缀，前缀冲突时取完整 base——两条记忆绝不会被并成一条；report
始终保留完整身份。每条轨迹的 `spans` 是指向 report 的无叙述索引：

| 字段 | 含义 |
|---|---|
| `operation_id` | 区间 id——在 `injudge-report.json` 中可查完整记录 |
| `verdict` / `confidence` | 区间判定与评审置信度（`confidence` 为 null 时省略） |
| `steps` | 该区间投递落在的 `[首个, 末个]` agent 步 |
| `n_injections` | 区间内的投递数 |
| `usage` | 逐记忆的使用标签（与 `memory_usage` 同一套标签；空时省略） |
