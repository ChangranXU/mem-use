---
description: "injudge.report/v15、injudge.summary/v13、全部字段，以及每项比率如何计数。"
icon: box-archive
---

# 产物

{% hint style="info" %}
injudge 把一次运行的全部产物写入输出容器（`-o DIR`，默认 `output/`）下该次运行自己的 `<run-stamp>/` 文件夹——一次运行绝不覆盖另一次运行的产物。
{% endhint %}

| 文件 | Schema | 作用 |
|---|---|---|
| `injudge-report.json` | `injudge.report/v15` | 运行时评审记录。每个区间完成后保存，待评区间标为 `unjudged`；即使一条多 episode 轨迹中途被打断，也保留已完成部分。用 `--verdicts` 喂给 memgraph。 |
| `summary.json` | `injudge.summary/v13` | 最终统计集合：计数、比率，以及无叙述的区间索引。 |
| `memgraph-verdicts.html` | — | 仅在 `--html` 时产出（默认关闭）：召回节点按区间判定着色的 memgraph。 |
| `evidence/<span-dir>/` | — | 评审智能体实际运行的每个区间一个文件夹：它当时可用的原始文件。 |
| `agent-runs/<span-dir>.jsonl` | — | 每个送入评审智能体的区间一条轨迹：评判对话的 pi 事件流记录（见[评审智能体](judge-client.md#轨迹记录)）。 |

离线准备运行会在自己的运行目录下写出另一套产物：

| 文件 | Schema | 作用 |
|---|---|---|
| `episode-inputs.json` | `injudge.episode-inputs/v1` | episode 索引、执行/基准身份、边界来源、准备状态、附件状态、计数、审计投影与未归属诊断；不包含 verdict，其 `tool` 块带 `mode: "prepare-only"`。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/episode.md` | — | 紧凑的 episode 头信息与固定文件导航。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/task.md` | — | 已录制的用户任务消息与后续更新，附来源。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/context.json` | — | 每个回合可见的消息出现与显式重放缺口。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/deliveries.md` | — | 按时间排列的 delivery、搜索、槽位与曝光证据，也包含空 delivery。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/searches.json` | — | 该 episode 已记录的搜索及其计数。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/manifest.json` | — | 该文件夹各文件的索引，含种类、来源边界，以及 episode 的 step/搜索/投递计数。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/steps/` | — | 原始全局 step 与工具结果可用时间。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/gold.md` / `outcome.json` | — | 记录到附件时的内容：黄金答案，以及结果状态与正确性及其联结键。 |

正常评审在每个区间自己的 evidence root 内复用这些已准备的 episode 文件；任何评审
都无法读取另一个 episode 或仅供审计的规范记忆索引。report 记录 episode 执行身份、
任务/附件来源、证据边界，以及分阶段的评审健康。

审计记录使用 `gold_join` 与 `outcome_join` 分别保存黄金答案和评测结果的来源。
仅按基准实例索引的 harness report 不能跨重复执行或不同实验组关联。结果冲突
或缺失会明确保留，不会被轨迹目录中的 sidecar 覆盖。

`<span-dir>` 是文件系统安全的区间 id——非 `[A-Za-z0-9._-]` 字符替换为 `_`、截到 80 字符，再加 8 字符摘要后缀（`span_dir_name`）——不同区间 id 基本不会共用同一个文件夹名。准备产物用同样的方式命名 episode 文件夹（`episode_dir_name`）。report 的 `input.transcript` 指向同一个派生名称。

memgraph 的 `--json` 产物是 `{tool, trajectories, ops, memories, edges, validation}`——与 HTML 内嵌的同一个 dict。见 [memgraph](memgraph.md#图模型)。

## Report 与 Summary

**report** 是评审记录。每个操作带有含 note 的评分、`analysis`、`evidence`、失败时的 `judge_raw`、投递成本，以及评审智能体实际运行的区间（含 error 结局）上的 `input` 智能体运行记录。report 顶层另带 validation 日志。区间的 `operation_id` 就是在该文件中查找所用的 id。

**summary** 是供阅读和比较记忆系统的数字。评判叙述只留在 report 中。

计数在轨迹/区间计数旁给出 episode 数量。每条轨迹的详情包含 episode 到区间的简洁
索引。CLI 汇总录制诊断的数量；其完整消息保留在 `injudge-report.json`，带 `--html`
时也在渲染出的图中。空投递与消费者未观测按独立类别（`empty`、`unassessable`）计数，
不进入任何方向性分母。`memgraph --verdicts` 只在报告的 `tool.schema` 为
`injudge.report/v15` 时接受它；它是读取时唯一校验的产物版本。

当多个身份共用同一显示键时，summary 的 `key` 字段（排查清单与 `memory_usage`）会带上 `key#<item 或 namespace>` 后缀——取 8 字符前缀，前缀冲突时取完整 base——两条记忆绝不会被并成一条。report 始终保留完整身份。

### Report 的 aggregates

`aggregates.per_trajectory.<id>` 与 `aggregates.overall` 以 `injudge/aggregate.py`
的扁平形状保存同一批未投影的数字：

| 字段 | 含义 |
|---|---|
| `n_trajectories` / `n_episodes` / `n_injections` / `n_spans` / `restated_injections` | 总体计数；`n_trajectories` 仅全局 |
| `positive` / `negative` / `unresolved` / `neutral` / `error` / `unjudged` / `unassessable` / `empty` | 判定计数——这里是顶层键，不是 summary 的 `verdicts.counts` |
| `rates` | 已评判区间中 positive/negative/unresolved/neutral 的占比 |
| `headline` / `cost` / `rubric` / `memory_usage` / `judge_compliance` / `mean_confidence_by_verdict` / `confidence_buckets` / `low_confidence` | summary 所投影的块，重新嵌套并精简 |
| `input_paths` / `gold_scored` / `forced_answers` / `policy` | 仅全局 |

## 数字如何推导

从评分信号到使用标签与区间判定的映射见[评分规则与判定推导](verdict-derivation.md)。本节是这些标签所进入的计数契约；同一组函数（`injudge/aggregate.py`）产出两份产物的数字。

### 总体

| 总体 | 成员 |
|---|---|
| 已评判区间 | `positive` + `negative` + `unresolved` + `neutral`。判定率的分母。`unresolved` 指已采取行动但评审无法判定效果——已确立的足迹、方向未知，与 neutral 的“未确立影响”分开。 |
| `empty` | 未投递任何条目（`auto: empty_delivery`）。不调用评审；单独计数和记录投递成本，不进入方向性比率或无已证收益份额。 |
| `unassessable` | 规则判定的 auto-neutral 区间（所有投递都无法归属到 agent 步骤——无法确定下游足迹）。只计数，其投递成本计入独立的 token 桶：观察缺失不等于测得的零效应，因此不进入任何方向性分母，也不计入无已证收益份额。有已评判总体、因而存在 `headline` 时，由其中的 `unassessable_spans` / `unassessable_tokens` 标出。 |
| LLM 已评判区间 | 无 `auto` 标记的已评判区间。评分比率、使用汇总与置信度的总体。 |
| `error` / `unjudged` | 只计数。永远不是比率分母。 |
| 重述注入 | 并入所属区间。计入 `n_injections` 与投递成本。不出现在比率中。 |

没有已评判样本的块会省略空的 `rates` / `headline` / `rubric` / `confidence` 壳。`cost` 只在两个总计都为零时省略——仅有 error 的运行只要有过注入仍带成本，计入 `error` 类。

### 判定率

分母是已评判区间；重述的重复注入绝不出现在比率中，它们已并入所属区间。

$$
\mathrm{rates}[v] = \frac{\mathrm{count}[v]}{\mathrm{positive} + \mathrm{negative} + \mathrm{unresolved} + \mathrm{neutral}}
$$

`headline.judged_spans` 就是该分母。`headline.share_without_demonstrated_benefit` 是 negative、neutral 与 unresolved 区间的注入 token 成本占已评判区间成本的份额——一个**估算的区间级“收益未得到证明”代理**，绝非测得的浪费、损害或净收益——其分子与分母（`tokens_without_demonstrated_benefit` / `judged_tokens`）随行命名，其中 unresolved 部分（`unresolved_tokens`：归因或效果未定）单独标出；被排除的 error/unjudged token 在非零且报告了该份额时以 `error_unjudged_tokens` 呈现（unassessable token 同理以 `unassessable_tokens` 呈现），`headline.caveat` 随行于每一个报告的份额，而不只在份额偏高时出现。

### 评分比率

总体是 LLM 已评判区间上的已打分投递（`区间 × 投递条目`，按条目的区间本地 id 键控）。全空的覆盖缺口行会被跳过；为 null 的信号值绝不进入分母。

| 比率 | 分子 | 分母 | 读作 |
|---|---|---|---|
| `relevance` | `true` | `true` + `false` | 相关占比 |
| `accuracy_corroborated` / `accuracy_uncontradicted` / `accuracy_contradicted` | 各取值 | 全部三种 accuracy 已判定投递 | 三路拆分——绝不再合成单一的"accurate"比率：`uncontradicted` 不说明任何问题，折进去会让大量无法核实的运行报告 ≈1.0 |
| `novelty` | `novel` + `confirming` | 上述 + `redundant` | 非冗余占比 |
| `influence` | `referenced` + `selected` + `acted_on` | 上述 + `none` | 有任何足迹的占比 |
| `positive_effect` | `improved` | `improved` + `no_effect` + `harmed` | 有 effect 的投递中 improved 的占比 |

`positive_effect` 只在有 effect 判断的投递上计算——除非 influence 为 acted-on，该信号为 null。未知新颖性（`null`）计入 `rubric.counts.novelty.null`，不进入 novelty 比率分母；全部未知时 novelty 比率为 null。相关但被忽略且 novelty 为 null 的条目使用 `ignored_unknown_novelty`。`rubric.counts` 中的信号枚举即[评分信号表](verdict-derivation.md#评分信号)中的值。

### 排查清单

`inaccurate_keys` 收集 `accuracy` 为 `contradicted` 的身份；`irrelevant_keys` 收集 `relevance` 为 `false` 的身份。每条带上产生它的区间 id。`shared_anchors` 按身份列出被两个及以上区间记为 `acted_on` 的同一动作步骤（`{traj, step, spans}`）：同一记忆以不同文本投递时窗口可以重叠，各区间判定照常保留，但共享步骤是一个动作，不是两份收益。每个投递条目恰好映射回其自身投递的一个 `(namespace, item)` 身份，因此一个 key 由多个身份投递时，每个身份都会入账。未携带已录制身份（无 `item_id`、无 key——如 redacted 匿名引用）的条目没有可联结的身份对：按区间作用域入账（其 `item_id` 记录操作与区间内 id），两个区间的无身份条目绝不合并。

### 逐记忆使用

只在 LLM 已评判区间上按 `(namespace, item)` 合并。对多 item 投递的联合判断会在每个身份下计数。当首尾两个*已录制* value 都存在且不同时，`value_drift` 为 true；redact（未录制）的末端不是 drift。

操作本地显示 ID 不作为汇总身份。规范内容身份生成 `content:<sha256>` 形式的条目 ID，原始引用保持完整。memgraph 的候选、投递与删除引用另带 `grouping_id`，用于浏览器与图注册表联结。

### 交付成本与策略

投递成本按确定性规则核算：每个区间的成本是其全部投递（含重述）的已录制放置内容大小之和，按 `chars // 4` 估算 token，计入该区间的判定类别。未记录大小的 redact 放置计 0 token，并产生警告。error 与 unjudged 的 token 计入 `cost`，但不进 `share_without_demonstrated_benefit` 的分母。

投递策略度量记忆系统注入的轻重：按轨迹与整体，对照该轨迹全部已计费的 prompt token 预算，所有泳道的用量记录都计入，包括记忆泳道的提取调用。

计费预算在三个总量中取最大有效值：`run.json` 的 `usage.prompt_tokens`、`agent_end` 的 `usage.prompt_tokens`、所有 live assistant-message 的用量总和。两种快照均包含不产生对话轮次的 embedding 调用，替代消息总量而不与之相加。无效或较小的快照会警告；两种快照均缺失时，消息用量只是下界，已绑定 embedding 角色时会提示漏计风险。

### 置信度

均值与分桶（`[0,0.5)`、`[0.5,0.8)`、`[0.8,1.0]`）只覆盖 LLM 已评判区间。置信度从不加权计数。置信度 `< 0.5` 的区间列入 `low_confidence`。

### 输入口径

仅 overall：带 `input` 记录的区间——评审智能体实际运行的每个区间，含 error 结局——计入 `input_paths.agent`、`gold_scored` 与 `forced_answers`（`forced` 表示最终答案出自预算耗尽的强制作答轮）。按规则计数的空投递或不可归属区间，以及 unjudged 区间没有 `input`。

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

比率仍是数值型小数，不转成百分比字符串；`null` 仍表示未确定，而不是零。

### 计数与判定

以下字段位于 `overall` 和每条轨迹的对比行中：

| 字段 | 含义 |
|---|---|
| `counts.n_trajectories` | 轨迹数；仅全局 |
| `counts.n_injections` | 召回注入投递数，含重述的重复投递 |
| `counts.n_spans` | 注入区间数——评审单元 |
| `counts.n_episodes` | 该轨迹各区间被归属到的 episode 分组数；全局为其总和 |
| `counts.restated_injections` | `n_injections − n_spans` |
| `verdicts.counts.positive` / `negative` / `unresolved` / `neutral` | 区间判定：已采取行动且有益 / 使用造成伤害 / 归因或效果未定 / 没有已建立的收益或伤害。仅引用记忆不构成 negative。 |
| `verdicts.counts.error` / `unjudged` / `unassessable` | 评审失败 / 被 `--limit` 跳过 / 没有可归属的 agent 步骤；保留计数，但不进入方向性比率。 |
| `verdicts.counts.empty` | 未投递任何内容；只计数，不进入任何方向性分母与无已证收益份额 |
| `verdicts.rates` | 已评判区间中 positive/negative/unresolved/neutral 的占比；无已评判区间时省略 |

### Headline、投递与评审

以下块位于 `overall` 和 `details.trajectories.<id>`：

| 字段 | 含义 |
|---|---|
| `headline.judged_spans` | 比率的分母；为 0 时整个 `headline` 省略 |
| `headline.positive_span_rate` / `negative_span_rate` / `unresolved_span_rate` | 始终一起呈现；无已证收益份额可表示时随行于旁 |
| `headline.share_without_demonstrated_benefit` | negative + neutral + unresolved 的注入 token 成本占已评判区间成本的份额，随行命名 `tokens_without_demonstrated_benefit` / `judged_tokens`，并单独标出其中的 `unresolved_tokens` |
| `headline.error_unjudged_tokens` | 不进入任何比率或份额的 error/unjudged token 桶；仅非零且报告了该份额时写出 |
| `headline.unassessable_spans` / `unassessable_tokens` | 规则判定的无后续回合区间及其投递成本；仅非零且 `headline` 存在时写出，不进入方向性分母 |
| `headline.caveat` | 每一个报告的份额都附带的说明：它统计什么、它不是浪费、损害或净收益，以及区间级整体计费的粗糙性 |
| `delivery.cost` | 确定性投递成本：`content_chars_total`、`approx_tokens_total`（`chars // 4`）、`approx_tokens_by_verdict`（每个计数类别都补零填充）。两个总计都为零时省略该块。 |
| `delivery.policy` | `n_injections`、`n_steps`、`injection_density`（每 agent 步的注入数）、`restated_ratio`、`est_injected_tokens`、`prompt_tokens_total`、`injected_token_share`（注入占已计费 prompt 预算的份额）、`n_searches`、`n_extractions` / `extraction_failures`。全局另含 `est_injected_tokens_billed`，用于只有部分轨迹有 usage 记录时的份额分子。提取失败与评审失败分开统计。 |
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
| `rubric.shared_anchors` | 多个区间记到同一记忆上的动作步骤排查清单 `{key, anchors: [{traj, step, spans}]}`；为空时省略 |
| `memory_usage` | `key` + 非零 `tallies`；跨区间记录的内容发生变化时带 `value_drift: true`。标签见[判定推导](verdict-derivation.md#使用标签)。 |

## 区间索引

`details.trajectories.<id>.spans` 是指向 report 的无叙述索引：

| 字段 | 含义 |
|---|---|
| `operation_id` | 区间 id；据此在 `injudge-report.json` 查找完整记录 |
| `episode_execution_id` | 所属 episode 的执行身份；边界不明确时省略 |
| `verdict` / `confidence` | 区间结果与评审置信度；confidence 为 null 时省略。自动标记的区间读作其计数类别：空投递为 `empty`，消费者未观测为 `unassessable`。 |
| `auto` | 不可评估区间的规则原因，如 `no_subsequent_turn`；已评审区间省略 |
| `steps` | 区间投递所在的 `[first, last]` agent 步；未归属到 step 时省略 |
| `n_injections` | 区间中的投递数 |
| `usage` | 逐记忆使用标签；为空时省略，不投影 auto-neutral 的占位 `unclear` |

## 仅 Report 中的字段

`injudge-report.json` 的每个操作还带有：`kind`（`recall_injection_span`）、`traj`、`seq`、`n_injections`、`steps`（无投递归属到 step 时省略）、`payload_summary` 和 `delivery_cost`（`deliveries`、`content_chars_total`、`approx_tokens_total`）；`episode_execution_id` 与 `episode_last_step`；`memories`（带区间本地 id 的完整已录制引用）；`memory_exposure`——与 `memories` **按位置平行**的列表（下标 `i` 描述 `memories[i]`），每条为 `{state, exposed_chars, value_chars}`，state 取 `full`、`full_normalized`、`clipped` 或 `unknown`，未核实的规范值不会进入评审证据；以及逐条目的 `memory_windows`（全局步骤上界，至迟止于该 episode 的最后可用步骤；同 episode 的后续投递也可提前截断），`memory_window_starts` 记录同一步投递冲突的例外（未新增已核实文本的同步副本在共享步骤之后开窗），`memory_variant_steps` 记录窗口内同一记忆以不同或更窄文本再次投递的步骤（不会截断窗口），`memory_followups` 记录经过工具结果来源证明的显式选取 `{step, delivery_step, tool_call_id, operation_id}`，动作步骤与结果投递步骤分开记录。与 `memory_exposure` 不同，`rubric`、`memory_usage` 和 `memory_windows` 按区间本地 id 键控。评分 note、动作 `anchors`、`analysis`、`evidence`、`step_gaps`、`new_keys` / `removed_keys` / `previous_keys` / `changed_keys`、`auto`、`error`、`judge_raw` 和适用时的 `pe_pairing_corrections` 仍保留。评审智能体实际运行的区间——含其自身的 error 结局——带有 `input`：`path: agent`、尝试次数、保留的 `llm_calls` / `tool_calls` 及成功读取且仍保留的步骤文件数（`retained_evidence`；`read_steps` 列出完整文本仍保留的全局步骤，包括 baseline 中索引过的完整副本——每个被引用的锚点及结构化 evidence 步骤都必须在其中）、实际执行调用数、包括回滚调用的上报用量、上下文回滚、`any_phase_forced`，以及相对于运行文件夹的 transcript 路径；在智能体运行前就被拒的区间（episode 边界不安全或不完整，或父 episode 没有已录制任务）取 error 判定，没有 `input`、没有 evidence 文件夹、也没有 transcript。`gold` 与仅代表最终答案的 `forced` 在适用时出现。轨迹行带有 `id`、`run_id`、`path`、`instance_id`、`model`、`adapter`、`n_steps`、`n_injections`、`n_spans`、其 `episodes` 行以及该轨迹的 `policy`；episode 行记录边界、任务、附件与证据来源，但不产生 episode verdict。

report 的 `tool` 块给出工具名（`name: "injudge"`）与其 `schema`，并带有已解析、无密钥的评审智能体配置（`config`，URL 规范化为 `scheme://host[:port][/path]`）以及 `judge_model`。

`injections` 数组列出每一次原始投递及其所属区间、该区间的 `episode_execution_id`（若有时）、`role`（`first` 或 `restated`）、`verdict`、`content_chars`，以及 `op_id` / `seq` / `step`。

## v15 的来源记录与解释

`tool.provenance` 保存 implementation_sha256 和 prompts_sha256；执行过的区间
input 保存 evidence_sha256 和 system_prompt_sha256。它们标识当前代码、提示词及
实际评审输入，不含凭据，也不证明预置记忆库的来源日期或无污染。

influence null 单独计数，不进入已评估影响的分母。selected 的效果为 null，不能
继承后续读取的收益；no_effect 派生为中性的 unproductive，misused 必须为 harmed。
neutral 包括选择、提及、无收益使用和未建立影响；unresolved 包括归因或效果不确定。
符号计数与 token 代理指标不是可相加的系统净收益。对泛化的结论须区分同任务答案
复用与跨任务迁移，不能只凭文本重合或日期猜测来源。
