---
description: "injudge.report/v11、injudge.summary/v9、全部字段，以及每项比率如何计数。"
icon: box-archive
---

# 产物

{% hint style="info" %}
injudge 把一次运行的全部产物写入输出容器（`-o DIR`，默认 `output/`）下该次运行自己的 `<run-stamp>/` 文件夹——一次运行绝不覆盖另一次运行的产物。
{% endhint %}

| 文件 | Schema | 作用 |
|---|---|---|
| `injudge-report.json` | `injudge.report/v11` | 运行时评审记录。每条轨迹完成后重写，崩溃也保留已完成部分。用 `--verdicts` 喂给 memgraph。 |
| `summary.json` | `injudge.summary/v9` | 最终统计集合：计数、比率，以及无叙述的区间索引。 |
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
| `unassessable` | 规则判定的 auto-neutral 区间（所有投递都无法归属到 agent 步骤——无法确定下游足迹）。只计数，其投递成本计入独立的 token 桶：观察缺失不等于测得的零效应，因此不进入任何方向性分母，也不计入浪费 token 份额。非零时由 `headline.unassessable_spans` / `unassessable_tokens` 标出。 |
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

`positive_effect` 只在有 effect 判断的投递上计算——除非 influence 为 acted-on，该信号为 null。未知新颖性（`null`）计入 `rubric.counts.novelty.null`，不进入 novelty 比率分母；全部未知时 novelty 比率为 null。相关但被忽略且 novelty 为 null 的条目使用 `ignored_unknown_novelty`。`rubric.counts` 中的信号枚举即[评分信号表](verdict-derivation.md#评分信号)中的值。

### 排查清单

`inaccurate_keys` 收集 `accuracy` 为 `contradicted` 的身份；`irrelevant_keys` 收集 `relevance` 为 `false` 的身份。每条带上产生它的区间 id。每个投递条目恰好映射回其自身投递的一个 `(namespace, item)` 身份，因此一个 key 由多个身份投递时，每个身份都会入账。未携带已录制身份（无 `item_id`、无 key——如 redacted 匿名引用）的条目没有可联结的身份对：按区间作用域入账（其 `item_id` 记录操作与区间内 id），两个区间的无身份条目绝不合并。

### 逐记忆使用

只在 LLM 已评判区间上按 `(namespace, item)` 合并。对多 item 投递的联合判断会在每个身份下计数。当首尾两个*已录制* value 都存在且不同时，`value_drift` 为 true；redact（未录制）的末端不是 drift。无身份条目与上文排查清单一样按区间作用域入账。

操作本地显示 ID 不作为汇总身份。规范内容身份生成 `content:<sha256>` 形式的条目 ID，原始引用保持完整。memgraph 的候选、投递与删除引用另带 `grouping_id`，用于浏览器与图注册表联结。

### 交付成本与策略

投递成本按确定性规则核算：每个区间的成本是其全部投递（含重述）的已录制放置内容大小之和，按 `chars / 4` 估算 token，计入该区间的判定类别。未记录大小的 redact 放置计 0 token，并产生警告。error 与 unjudged 的 token 计入 `cost`，但不进 `wasted_token_share` 的分母。

投递策略度量记忆系统注入的轻重，按轨迹与整体统计：注入密度（每 agent 步的注入数）、重述比例，以及注入占轨迹已计费 prompt token 预算的份额——覆盖所有泳道的用量记录，包括记忆泳道的提取调用。仅部分轨迹有计费记录时，整体份额的分子取 `est_injected_tokens_billed`。

计费预算在三个总量中取最大有效值：`run.json` 的 `usage.prompt_tokens`、`agent_end` 的 `usage.prompt_tokens`、所有 live assistant-message 的用量总和。两种快照均包含不产生对话轮次的 embedding 调用，替代消息总量而不与之相加。无效或较小的快照会警告；两种快照均缺失时，消息用量只是下界，已绑定 embedding 角色时会提示漏计风险。

### 置信度

均值与分桶（`[0,0.5)`、`[0.5,0.8)`、`[0.8,1.0]`）只覆盖 LLM 已评判区间。置信度从不加权计数。置信度 `< 0.5` 的区间列入 `low_confidence`。

### 输入路径计数

仅 overall：带 `input` 记录的区间——每个送入智能体运行的区间，含 error 结局——计入 `input_paths.agent`、`gold_scored` 与 `forced_answers`（`forced` 标记：对话的最终答案出自预算耗尽的强制作答轮）。只有规则判定的 auto-neutral 与 unjudged 区间没有 `input`，因此只有它们不出现在这些计数中。

## summary.json 字段

文件按“结果 → 明细 → 运行信息”排列，使用两个空格缩进：

| 区域 | 阅读内容 |
|---|---|
| `overall` | 先看 headline，再看运行计数、判定计数/比率、投递成本/策略、评审诊断；不插入长记忆列表。 |
| `trajectories` | 按输入顺序排列的精简对比行：`id`、已记录的 `instance_id` / `model` / `adapter`、`counts`、`verdicts`。 |
| `details.overall` | 全局 rubric 分布、排查清单和记忆使用汇总。 |
| `details.trajectories.<id>` | 用对比行的 `id` 查找该轨迹的 headline、投递统计、评审诊断、rubric、记忆使用和区间索引；不重复行内的计数与判定。 |
| `generated_at` | 源报告的时间戳，不表示重新评审的时间。 |
| `tool` | `schema`、`source_report_schema`、`judge_model` 和已解析、无密钥的 `config`；不含 API key、URL 用户信息或查询参数。 |

例如，先比较 `trajectories[0].verdicts.counts`，再用该行的 `id`（如 `T1`）
查阅 `details.trajectories.T1.spans`，最后通过 `operation_id` 定位完整报告。
比率仍是数值型小数，不转成百分比字符串；`null` 仍表示未确定，而不是零。

### 计数与判定

以下字段位于 `overall` 和每条轨迹的对比行中：

| 字段 | 含义 |
|---|---|
| `counts.n_trajectories` | 轨迹数；仅全局 |
| `counts.n_injections` | 召回注入投递数，含重述的重复投递 |
| `counts.n_spans` | 注入区间数——评审单元 |
| `counts.restated_injections` | `n_injections − n_spans` |
| `verdicts.counts.positive` / `negative` / `neutral` | 区间判定：已采取行动且有益 / 已采取行动但无益 / 两者均未建立。仅引用记忆不构成 negative。 |
| `verdicts.counts.error` / `unjudged` / `unassessable` | 评审失败 / 被 `--limit` 跳过 / 无可观察的后续回合；保留计数，但不进入方向性比率。 |
| `verdicts.rates` | 已评判区间中 positive/negative/neutral 的占比；无已评判区间时省略 |

### Headline、投递与评审

以下块位于 `overall` 和 `details.trajectories.<id>`：

| 字段 | 含义 |
|---|---|
| `headline.judged_spans` | 比率的分母；为 0 时整个 `headline` 省略 |
| `headline.positive_span_rate` / `negative_span_rate` | 两者始终一起呈现；wasted-token share 可表示时随行于旁 |
| `headline.wasted_token_share` | neutral + negative 的注入 token 成本占已评判区间成本的份额，随行命名 `wasted_tokens` / `judged_tokens` |
| `headline.error_unjudged_tokens` | 不进入任何比率或份额的 error/unjudged token 桶；仅非零时写出 |
| `headline.unassessable_spans` / `unassessable_tokens` | 规则判定的无后续回合区间及其投递成本；仅非零时写出，不进入方向性分母 |
| `headline.caveat` | 每一个浪费份额都附带的说明：区间级代理的粗糙性与判定的不对称 |
| `delivery.cost` | 确定性投递成本：`content_chars_total`、`approx_tokens_total`（`chars // 4`）、按区间判定归类的 `approx_tokens_by_verdict`；总量为零时省略 |
| `delivery.policy` | `n_injections`、`n_steps`、`injection_density`、`restated_ratio`、`est_injected_tokens`、`prompt_tokens_total`、`injected_token_share`、`n_searches`、`n_extractions` / `extraction_failures`。全局另含 `est_injected_tokens_billed`，用于只有部分轨迹有 usage 记录时的份额分子。提取失败与评审失败分开统计。 |
| `judge.confidence.mean_by_verdict` / `buckets` | LLM 已评判区间的非空分类均值，以及 `[0,0.5)` / `[0.5,0.8)` / `[0.8,1.0]` 桶计数 |
| `judge.input_paths` | 仅全局：通过评审智能体（`agent`）运行的区间，含 error 结果 |
| `judge.gold_scored` | 仅全局：附带黄金答案运行的区间 |
| `judge.forced_answers` | 仅全局：最终回答来自预算耗尽强制回合的区间，含 error 结果 |
| `judge.compliance` | `pe_pairing_corrected_spans` / `llm_judged_spans` / `pe_pairing_correction_rate`；存在 LLM 已评判区间时写出 |
| `judge.low_confidence` | 置信度 `< 0.5` 的区间 id；为空时省略 |

### Rubric 与记忆明细

以下块位于 `details.overall` 和 `details.trajectories.<id>`：

| 字段 | 含义 |
|---|---|
| `rubric.deliveries` | 被打分的记忆投递数；为 0 时整个 `rubric` 省略 |
| `rubric.counts` / `rubric.rates` | 各信号的取值分布与投影比率 |
| `rubric.no_footprint_reasons` | `irrelevant` / `redundant` / `confirming_ignored` / `ignored_novel` / `ignored_unknown_novelty`；为空时省略 |
| `rubric.inaccurate_keys` / `irrelevant_keys` | 被判定为受反证 / 不相关的记忆排查清单 `{key, spans}`；为空时省略 |
| `memory_usage` | `key` + 非零 `tallies`；跨区间记录的内容发生变化时带 `value_drift: true`。标签见[判定推导](verdict-derivation.md#使用标签)。 |

## 区间索引

`details.trajectories.<id>.spans` 是指向 report 的无叙述索引：

| 字段 | 含义 |
|---|---|
| `operation_id` | 区间 id；据此在 `injudge-report.json` 查找完整记录 |
| `verdict` / `confidence` | 区间结果与评审置信度；confidence 为 null 时省略。auto-neutral 使用 `unassessable`，与汇总计数一致。 |
| `auto` | 不可评估区间的规则原因，如 `no_subsequent_turn`；已评审区间省略 |
| `steps` | 区间投递所在的 `[first, last]` agent 步；未归属到 step 时省略 |
| `n_injections` | 区间中的投递数 |
| `usage` | 逐记忆使用标签；为空时省略，不投影 auto-neutral 的占位 `unclear` |

## 仅 Report 中的字段

`injudge-report.json` 每个操作还带有：`memories`（完整已录制引用，每条标注其区间本地 `id`）、`memory_exposure`（逐条目：placement 对该值是 `full`、`full_normalized`——整体载入但仅在空白符折叠 / markdown 转义剥离后匹配——、`clipped` 到 N 字符前缀，还是 `unknown`——无法核实，故评审输入中隐去规范值；含引号、反斜杠或换行结构的值只做逐字比对）、`memory_windows`（逐条目 id：该条目动作 `anchors` 校验所依据的足迹窗口结束步——从未被重投递时为 `null`，即其上界是步骤流末尾；同一步投递冲突会把窗口截断在该区间自己的末次投递步，即使此前已有更晚的有限上界；所有投递都无法归属到 agent 步骤时省略）、评分 note 与各条目的动作 `anchors`、`analysis`、`evidence`、`step_gaps`、`new_keys` / `removed_keys` / `previous_keys` / `changed_keys`、`auto`、`error`、`judge_raw`、`pe_pairing_corrections`（若有），以及送入智能体运行的区间（含 error 结局）上的 `input`（`path` 为 `agent`，外加 `attempts`、`llm_calls`、`tool_calls`、`transcript`——指向 `agent-runs/` 下该对话的轨迹 JSONL，相对输出目录——适用时还有 `"gold"` / `"forced"`——`forced` 标记步数或上下文预算以强制作答轮结束的对话）。`tool.config` 记录已解析、无密钥的评审智能体配置（净化为 `scheme://host[:port][/path]`），便于从产物本身复现运行。

`injections` 数组列出每一次原始投递及其所属区间、`role`（`first` 或 `restated`）、`verdict`、`content_chars`，以及 `op_id` / `seq` / `step`。
