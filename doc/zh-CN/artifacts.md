---
description: "injudge.report/v5、injudge.summary/v2，以及每项比率如何计数。"
icon: box-archive
---

# 产物

injudge 的全部产物写入输出目录（`-o DIR`，默认 `output/`）：

| 文件 | Schema | 作用 |
|---|---|---|
| `injudge-report.json` | `injudge.report/v5` | 运行时评审记录。每条轨迹完成后重写，崩溃也保留已完成部分。用 `--verdicts` 喂给 memgraph。 |
| `summary.json` | `injudge.summary/v2` | 最终统计集合：计数、比率，以及无叙述的区间索引。 |
| `memgraph-verdicts.html` | — | 仅在 `--html` 时产出（默认关闭）：召回节点按区间判定着色的 memgraph。 |

memgraph 的 `--json` 产物是 `{tool, trajectories, ops, memories, edges, validation}`——与 HTML 内嵌的同一个 dict。见 [memgraph](memgraph.md#图模型)。

## Report 与 summary

report 是评审记录。每个操作带有含 note 的评分、分析、证据、失败时的 `judge_raw`、投递成本、预算门区间上的 `input` 路径，以及 validation 日志。区间的 `operation_id` 就是指向该文件的索引。

summary 是供阅读和比较记忆系统的数字。评判叙述只留在 report 中。

当多个身份共用同一显示键时，summary 的 `key` 字段（排查清单与 `memory_usage`）会带上 `key#<item 或 namespace>` 后缀——取 8 字符前缀，前缀冲突时取完整 base——两条记忆绝不会被并成一条。report 始终保留完整身份。

## 数字如何推导

从评分信号到使用标签与区间判定的映射见 [判定推导](verdict-derivation.md)。本节是这些标签所进入的计数契约。

**总体。**

| 总体 | 成员 |
|---|---|
| 已评判区间 | `positive` + `negative` + `neutral`。含规则判定的 auto-neutral 区间。判定率的分母。 |
| LLM 已评判区间 | 无 `auto` 标记的已评判区间。评分比率、使用汇总与置信度的总体。 |
| `error` / `unjudged` | 只计数。永远不是比率分母。 |
| 重述注入 | 并入所属区间。计入 `n_injections` 与投递成本。不出现在比率中。 |

没有已评判样本的块会省略空的 `rates` / `headline` / `rubric` / `confidence` / `cost` 壳。仅 error 或零注入的轨迹仍保留判定计数。

**Headline。** `judged_spans` 是比率分母。`wasted_token_share` 是 neutral 与 negative 区间的注入 token 成本占已评判区间成本的份额：花在没有帮助的注入上的预算。

**评分投影。** `rubric.deliveries` 是 LLM 已评判区间上被打分的记忆投递（`区间 × key`）。null 信号永不稀释比率。布尔风格的比率是：

- `relevance` — 相关占比
- `accuracy` — 未被证伪及以上占比
- `novelty` — 非冗余占比
- `influence` — 有任何足迹的占比
- `positive_effect` — 在有 effect 的投递中 improved 的占比（其余按契约为 `null`）

`rubric.counts` 中的信号枚举即 [评分信号表](injudge.md#评分信号) 中的值。精确分子与分母见 [判定推导](verdict-derivation.md#从标签到汇总)。

## summary.json 字段

顶层：`tool.schema`、`generated_at`、`overall`（全部轨迹合并）、`trajectories`（逐轨迹块，含同样的统计外加区间索引）。每条轨迹在有记录时带上 `instance_id` / `model` / `adapter`。

每个聚合块（`overall` 与每条轨迹）：

| 字段 | 含义 |
|---|---|
| `n_injections` | 投递的召回注入数，含重述的重复投递 |
| `n_spans` | 注入区间数——评审单元 |
| `restated_injections` | `n_injections − n_spans` |
| `positive` / `negative` / `neutral` | 区间判定：注入的足迹改善 / 损害 / 未体现在工作中 |
| `error` / `unjudged` | 评审失败的区间 / 被 `--limit` 截掉的区间 |
| `rates` | 已评判区间上的判定占比（无评判时省略） |
| `headline.judged_spans` | 比率的分母（为 0 时整个 `headline` 省略） |
| `headline.wasted_token_share` | neutral + negative 的注入 token 成本占已评判区间成本的份额 |
| `rubric.deliveries` | 被打分的记忆投递数；为 0 时整个 `rubric` 省略 |
| `rubric.counts` / `rubric.rates` | 各信号的取值分布与投影比率 |
| `rubric.no_footprint_reasons` | 被忽略记忆未留足迹的原因：`irrelevant` / `redundant` / `confirming_ignored` / `ignored_novel`（空时省略） |
| `rubric.inaccurate_keys` / `irrelevant_keys` | 排查清单 `{key, spans}`——评审认定被证伪 / 不相关的记忆；仅在非空时写出 |
| `cost` | 确定性投递成本：`content_chars_total`、`approx_tokens_total`（`chars // 4`）、按区间判定类别归集的 `approx_tokens_by_verdict`（为零时省略） |
| `confidence.mean_by_verdict` / `confidence.buckets` | 按判定类别的均值（省略 null 类）与 `[0, 0.5)` / `[0.5, 0.8)` / `[0.8, 1.0]` 分桶计数，覆盖 LLM 已评判区间 |
| `input_paths` | 仅 overall：预算门后走全文 / 分块扫描输入的区间数 |
| `gold_scored` | 仅 overall：在 gold 评分规则下评判的区间数 |
| `over_budget` | 仅 overall：评分输入超过 `JUDGE_INPUT_TOKENS` 仍原样发送的区间数 |
| `n_trajectories` | 仅 overall |
| `memory_usage` | 逐记忆使用统计：`key` + 非零 `tallies`；已录制内容跨区间变化时加 `value_drift: true`。标签定义见 [判定推导](verdict-derivation.md#使用标签)。 |
| `low_confidence` | 置信度 `< 0.5` 的区间 id（仅在非空时写出） |
| `policy` | 投递策略：`n_injections`、`n_steps`、`injection_density`（每 agent 步的注入数）、`restated_ratio`、`est_injected_tokens`、`prompt_tokens_total`（计费用量）、`injected_token_share`、`n_searches`。overall 还带 `est_injected_tokens_billed`——仅部分轨迹有计费记录时，份额的分子取它。 |

## 区间索引

每条轨迹的 `spans` 是指向 report 的无叙述索引：

| 字段 | 含义 |
|---|---|
| `operation_id` | 区间 id——在 `injudge-report.json` 中可查完整记录 |
| `verdict` / `confidence` | 区间判定与评审置信度（`confidence` 为 null 时省略） |
| `steps` | 该区间投递落在的 `[首个, 末个]` agent 步 |
| `n_injections` | 区间内的投递数 |
| `usage` | 逐记忆的使用标签（空时省略）。auto-neutral 的占位 `unclear` 不会投影到这里。 |

## 仅 report 中的字段

`injudge-report.json` 每个操作还带有：`memories`（完整已录制引用）、评分 note、`analysis`、`evidence`、`step_gaps`、`new_keys` / `removed_keys`、`auto`、`error`、`judge_raw`，以及预算门区间上的 `input`（`path` 为 `full` 或 `scan`，外加扫描计数，适用时还有 `"gold"`、`"over_budget"`）。评判意外失败的区间会保留不含扫描计数的再推导 `{path, gold?}` 记录，使路径计数仍包含该区间。

`injections` 数组列出每一次原始投递及其所属区间、`role`（`first` 或 `restated`）和 `content_chars`。
