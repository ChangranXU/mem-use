---
description: "injudge.report/v11、injudge.summary/v7、全部字段，以及每项比率如何计数。"
icon: box-archive
---

# 产物

{% hint style="info" %}
injudge 的全部产物写入输出目录（`-o DIR`，默认 `output/`）。
{% endhint %}

| 文件 | Schema | 作用 |
|---|---|---|
| `injudge-report.json` | `injudge.report/v11` | 运行时评审记录。每条轨迹完成后重写，崩溃也保留已完成部分。用 `--verdicts` 喂给 memgraph。 |
| `summary.json` | `injudge.summary/v7` | 最终统计集合：计数、比率，以及无叙述的区间索引。 |
| `memgraph-verdicts.html` | — | 仅在 `--html` 时产出（默认关闭）：召回节点按区间判定着色的 memgraph。 |
| `evidence/<span-dir>/` | — | 每个送入评审智能体的区间一个文件夹（含 error 结局）：智能体检索过的原始文件。 |
| `agent-runs/<span-dir>.jsonl` | — | 每个送入评审智能体的区间一条轨迹：评判对话的 pi 事件流记录（见[评审智能体](judge-client.md#轨迹记录)）。 |

`<span-dir>` 是文件系统安全的区间 id——非 `[A-Za-z0-9._-]` 字符替换为 `_`、截到 80 字符，再加 8 字符摘要后缀（`span_dir_name`）——任意区间 id 都不会冲突。report 的 `input.transcript` 指向同一个派生名称。

memgraph 的 `--json` 产物是 `{tool, trajectories, ops, memories, edges, validation}`——与 HTML 内嵌的同一个 dict。见 [memgraph](memgraph.md#图模型)。

## Report 与 Summary

**report** 是评审记录。每个操作带有含 note 的评分、`analysis`、`evidence`、失败时的 `judge_raw`、投递成本，以及送入智能体运行的区间（含 error 结局）上的 `input` 智能体运行记录。report 顶层另带 validation 日志。区间的 `operation_id` 就是在该文件中查找所用的 id。

**summary** 是供阅读和比较记忆系统的数字。评判叙述只留在 report 中。

当多个身份共用同一显示键时，summary 的 `key` 字段（排查清单与 `memory_usage`）会带上 `key#<item 或 namespace>` 后缀——取 8 字符前缀，前缀冲突时取完整 base——两条记忆绝不会被并成一条。report 始终保留完整身份。

## 数字如何推导

从评分信号到使用标签与区间判定的映射见[评分规则与判定推导](verdict-derivation.md)。本节是这些标签所进入的计数契约；同一组函数（`injudge/aggregate.py`）产出两份产物的数字。

### 总体

| 总体 | 成员 |
|---|---|
| 已评判区间 | `positive` + `negative` + `neutral`。判定率的分母。 |
| `unassessable` | 规则判定的 auto-neutral 区间（没有任何 agent 回合被归类——无从观察后续）。只计数，其投递成本计入独立的 token 桶：观察缺失不等于测得的零效应，因此不进入任何方向性分母，也不计入浪费 token 份额。非零时由 `headline.unassessable_spans` / `unassessable_tokens` 标出。 |
| LLM 已评判区间 | 无 `auto` 标记的已评判区间。评分比率、使用汇总与置信度的总体。 |
| `error` / `unjudged` | 只计数。永远不是比率分母。 |
| 重述注入 | 并入所属区间。计入 `n_injections` 与投递成本。不出现在比率中。 |

没有已评判样本的块会省略空的 `rates` / `headline` / `rubric` / `confidence` 壳。`cost` 只在两个总计都为零时省略——仅有 error 的运行只要有过注入仍带成本，计入 `error` 类。

### 判定率

分母是已评判区间；重述的重复注入绝不出现在比率中，它们已并入所属区间。

$$
\mathrm{rates}[v] = \frac{\mathrm{count}[v]}{\mathrm{positive} + \mathrm{negative} + \mathrm{neutral}}
$$

`headline.judged_spans` 就是该分母。`headline.wasted_token_share` 是 neutral 与 negative 区间的注入 token 成本占已评判区间成本的份额——一个**区间级"无收益投递"代理**：neutral 区间被引用的记忆计入浪费，positive 区间被无视的记忆计入产出，混合捆绑整体按其判定计费；因此份额的分子与分母（`wasted_tokens` / `judged_tokens`）随行命名，被排除的 error/unjudged token 在非零时以 `error_unjudged_tokens` 呈现（unassessable token 同理以 `unassessable_tokens` 呈现）；`headline.caveat` 随行于每一个报告的份额，重申两种已知的粗糙性（区间级代理，以及判定的不对称：被无视的垃圾是 neutral；只有 acted-on 且无效或有害才是 negative）——不会因为份额偏低或存在少量 negative 区间而消失。

### 评分比率

总体是 LLM 已评判区间上的已打分投递（`区间 × 投递条目`，按条目的区间本地 id 键控）。全空的覆盖缺口行会被跳过；为 null 的信号值绝不进入分母。

| 比率 | 分子 | 分母 | 读作 |
|---|---|---|---|
| `relevance` | `true` | `true` + `false` | 相关占比 |
| `accuracy_corroborated` / `accuracy_uncontradicted` / `accuracy_contradicted` | 各取值 | 全部三种 accuracy 已判定投递 | 三路拆分——绝不再合成单一的"accurate"比率：`uncontradicted` 不说明任何问题，折进去会让大量无法核实的运行报告 ≈1.0 |
| `novelty` | `novel` + `confirming` | 上述 + `redundant` | 非冗余占比 |
| `influence` | `referenced` + `acted_on` + `decisive` | 上述 + `none` | 有任何足迹的占比 |
| `positive_effect` | `improved` | `improved` + `no_effect` + `harmed` | 有 effect 的投递中 improved 的占比 |

`positive_effect` 只在有 effect 判断的投递上计算——除非 influence 为 acted-on，该信号为 null。`rubric.counts` 中的信号枚举即[评分信号表](verdict-derivation.md#评分信号)中的值。

### 排查清单

`inaccurate_keys` 收集 `accuracy` 为 `contradicted` 的身份；`irrelevant_keys` 收集 `relevance` 为 `false` 的身份。每条带上产生它的区间 id。每个投递条目恰好映射回其自身投递的一个 `(namespace, item)` 身份，因此一个 key 由多个身份投递时，每个身份都会入账。未携带已录制身份（无 `item_id`、无 key——如 redacted 匿名引用）的条目没有可联结的身份对：按区间作用域入账（其 `item_id` 记录操作与区间内 id），两个区间的无身份条目绝不合并。

### 逐记忆使用

只在 LLM 已评判区间上按 `(namespace, item)` 合并。对多 item 投递的联合判断会在每个身份下计数。当首尾两个*已录制* value 都存在且不同时，`value_drift` 为 true；redact（未录制）的末端不是 drift。无身份条目与上文排查清单一样按区间作用域入账。

### 交付成本与策略

投递成本按确定性规则核算：每个区间的成本是其全部投递（含重述）的已录制放置内容大小之和，按 `chars / 4` 估算 token，计入该区间的判定类别。未记录大小的 redact 放置计 0 token，并产生警告。error 与 unjudged 的 token 计入 `cost`，但不进 `wasted_token_share` 的分母。

投递策略度量记忆系统注入的轻重，按轨迹与整体统计：注入密度（每 agent 步的注入数）、重述比例，以及注入占轨迹已计费 prompt token 预算的份额——覆盖所有泳道的用量记录，包括记忆泳道的提取调用。仅部分轨迹有计费记录时，整体份额的分子取 `est_injected_tokens_billed`。

### 置信度

均值与分桶（`[0,0.5)`、`[0.5,0.8)`、`[0.8,1.0]`）只覆盖 LLM 已评判区间。置信度从不加权计数。置信度 `< 0.5` 的区间列入 `low_confidence`。

### 输入路径计数

仅 overall：带 `input` 记录的区间——每个送入智能体运行的区间，含 error 结局——计入 `input_paths.agent`、`gold_scored` 与 `forced_answers`（`forced` 标记：对话的最终答案出自预算耗尽的强制作答轮）。只有规则判定的 auto-neutral 与 unjudged 区间没有 `input`，因此只有它们不出现在这些计数中。

## summary.json 字段

顶层：`tool.schema`、`tool.judge_model`、`tool.config`（本次运行实际使用的、已解析且无密钥的评审智能体配置）、`generated_at`、`overall`（全部轨迹合并）、`trajectories`（逐轨迹块，含同样的统计外加区间索引）。每条轨迹在有记录时带上 `instance_id` / `model` / `adapter`。

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
| `headline.positive_span_rate` / `negative_span_rate` | 判定率，两者始终一起呈现；wasted-token share 在可表示时随行于旁 |
| `headline.wasted_token_share` | neutral + negative 的注入 token 成本占已评判区间成本的份额，随行命名 `wasted_tokens` / `judged_tokens` |
| `headline.error_unjudged_tokens` | 不进任何比率或份额分母的 token 桶（仅非零时写出） |
| `headline.unassessable_spans` / `unassessable_tokens` | 规则判定的无后续回合区间及其投递成本（仅非零时写出）；不进入任何方向性分母 |
| `headline.caveat` | 随行于每一个报告的浪费份额：指明区间级代理的粗糙性与判定的不对称 |
| `rubric.deliveries` | 被打分的记忆投递数；为 0 时整个 `rubric` 省略 |
| `rubric.counts` / `rubric.rates` | 各信号的取值分布与投影比率 |
| `rubric.no_footprint_reasons` | 被忽略记忆未留足迹的原因：`irrelevant` / `redundant` / `confirming_ignored` / `ignored_novel` / `ignored_unknown_novelty`（空时省略） |
| `rubric.inaccurate_keys` / `irrelevant_keys` | 排查清单 `{key, spans}`——评审认定被证伪 / 不相关的记忆；仅在非空时写出 |
| `cost` | 确定性投递成本：`content_chars_total`、`approx_tokens_total`（`chars // 4`）、按区间判定类别归集的 `approx_tokens_by_verdict`（为零时省略） |
| `confidence.mean_by_verdict` / `confidence.buckets` | 按判定类别的均值（省略 null 类）与 `[0,0.5)` / `[0.5,0.8)` / `[0.8,1.0]` 分桶计数，覆盖 LLM 已评判区间 |
| `input_paths` | 仅 overall：送入评审智能体运行的区间数（`agent`），含 error 结局 |
| `gold_scored` | 仅 overall：附带 golden answer 运行的区间数（gold 评分规则） |
| `forced_answers` | 仅 overall：最终答案出自预算耗尽强制作答轮的区间数，含 error 结局 |
| `judge_compliance` | 评审契约合规度：`pe_pairing_corrected_spans` / `llm_judged_spans` / `pe_pairing_correction_rate`——LLM 已评判区间中评分表需要 positive_effect↔influence 配对纠正的占比（存在 LLM 已评判区间时写出） |
| `n_trajectories` | 仅 overall |
| `memory_usage` | 逐记忆使用统计：`key` + 非零 `tallies`；已录制内容跨区间变化时加 `value_drift: true`。标签定义见[判定推导](verdict-derivation.md#使用标签)。 |
| `low_confidence` | 置信度 `< 0.5` 的区间 id（仅在非空时写出） |
| `policy` | 投递策略：`n_injections`、`n_steps`、`injection_density`（每 agent 步的注入数）、`restated_ratio`、`est_injected_tokens`、`prompt_tokens_total`（计费用量）、`injected_token_share`、`n_searches`，以及 `n_extractions` / `extraction_failures`（录制到的生成数 vs 被标记为失败/不完整的数量——提取可靠性；overall 同样汇总，与评审的 error 区间相互独立）。overall 还带 `est_injected_tokens_billed`——仅部分轨迹有计费记录时，份额的分子取它。 |

## 区间索引

每条轨迹的 `spans` 是指向 report 的无叙述索引：

| 字段 | 含义 |
|---|---|
| `operation_id` | 区间 id——在 `injudge-report.json` 中可查完整记录 |
| `verdict` / `confidence` | 区间判定与评审置信度（`confidence` 为 null 时省略） |
| `steps` | 该区间投递落在的 `[首个, 末个]` agent 步（无可归属步骤时省略） |
| `n_injections` | 区间内的投递数 |
| `usage` | 逐记忆的使用标签（空时省略）。auto-neutral 的占位 `unclear` 不会投影到这里。 |

## 仅 Report 中的字段

`injudge-report.json` 每个操作还带有：`memories`（完整已录制引用，每条标注其区间本地 `id`）、`memory_exposure`（逐条目：placement 对该值是 `full`、`full_normalized`——整体载入但仅在空白符折叠 / markdown 转义剥离后匹配——、`clipped` 到 N 字符前缀，还是 `unknown`——无法核实，故评审输入中隐去规范值；含引号、反斜杠或换行结构的值只做逐字比对）、`memory_windows`（逐条目 id：该条目动作 `anchors` 校验所依据的足迹窗口结束步——从未被重投递时为 `null`，即其上界是步骤流末尾；同一步投递冲突会把仍开放的窗口闭合在该区间自己的末次投递步；无 agent 回合归类时省略）、评分 note 与各条目的动作 `anchors`、`analysis`、`evidence`、`step_gaps`、`new_keys` / `removed_keys` / `previous_keys` / `changed_keys`、`auto`、`error`、`judge_raw`、`pe_pairing_corrections`（若有），以及送入智能体运行的区间（含 error 结局）上的 `input`（`path` 为 `agent`，外加 `attempts`、`llm_calls`、`tool_calls`、`transcript`——指向 `agent-runs/` 下该对话的轨迹 JSONL，相对输出目录——适用时还有 `"gold"` / `"forced"`——`forced` 标记步数或上下文预算以强制作答轮结束的对话）。`tool.config` 记录已解析、无密钥的评审智能体配置（净化为 `scheme://host[:port][/path]`），便于从产物本身复现运行。

`injections` 数组列出每一次原始投递及其所属区间、`role`（`first` 或 `restated`）、`verdict`、`content_chars`，以及 `op_id` / `seq` / `step`。

## Schema 变更：Report v11 / Summary v7

当已录制基线无法确立 novel、confirming 或 redundant 时，`novelty.value`
现在允许 `null`。这表示证据缺失，不是第四个档位；强行指定档位会制造结论。
仅 novelty 和 positive_effect 允许 null。影响判断、动作锚点校验、usage
推导与区间判定规则保持不变。

两个产物都将已评审条目的未知新颖性计入 `rubric.counts.novelty.null`，
从 novelty 比率分母中排除这些条目，同时保留其其他信号和 usage 统计。
相关但被忽略且 novelty 为 null 的条目使用 `ignored_unknown_novelty`，
不再误计为 `ignored_novel`。标准输出明确列出已判定档位的总体与未知数量；
全部未知时 novelty 比率为 null。

本次取代 report v10 / summary v6。memgraph 仅接受当前报告 schema，
不提供迁移或兼容读取层；重新运行 injudge 生成新产物。
