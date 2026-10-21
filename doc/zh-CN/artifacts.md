---
description: "injudge.report/v26、injudge.summary/v18、全部字段，以及每项比率如何计数。"
icon: box-archive
---

# 产物

当前 schema 为 report v26 和 summary v18。Report v26 在后续副本显示相同或更短文本时（如仍可见完整读取的预览）也保留最早归属，
并将所有条目都没有合格动作步骤的区间按规则判为中性（`auto: no_eligible_step`），不调用评审。
判定计数与评审本应推导的结果一致，但评分规则、用法与置信度汇总只覆盖评审实际打分的区间，不能与 v25 直接比较。
同一回合新增可见选择标识符的副本保留该步的动作资格。影响规则也将同条目窗口交接后的共有内容归后续窗口，
无需先发生选择；受影响的判定可能改变，不应与旧提示词的结果混合比较。
Report v25 按评审实际读取范围区分未知影响：
完整读取窗口后的未知影响为 `ambiguous`（归因不明），判为中性；否则仍为 `unclear`，
判为未决。判定计数不能与 v24 直接比较（未决减少、中性增加；正面与负面不变）。
Report v24 在消费者输入中仍有已证明副本时，
为相同曝光文本保留最早的记功归属，并用 `memory_window_owners` 记录被抑制的副本。
无法证明连续可见时，后续投递另起区间并使用自己的输入基线。
不同文本及已证明的选取后完整读取仍沿用原交接规则。影响、效果和判定计数不能与 v23 直接比较。
Report v23 将引用的结果章节纳入完整步骤读取证明。
它也收紧了效果证据要求：智能体虚构的检索标识符本身不能证明记忆危害，泛化的测试通过计数
不能证明收益。v22 及更早版本的效果和判定计数不能与 v23 直接比较。
Report v22 引入已完成区间的检查点日志，未改变 rubric 标签。Report v20 改变了影响归因：
只有当行动使用了暴露前记录未陈述的内容时才归功于记忆，
influence null 不再表示"另一来源同样能解释"（见[判定推导](verdict-derivation.md)）。
v19 及更早版本的影响、效果与判定计数不能与 v20 及以后版本比较。v20 当时保留了
产物结构与 summary schema v17；v19 已删除 aggregates 中的 `headline` 块。一个记忆包可以同时包含有益
和有害条目，因此不能用整包判定衡量哪些 token 带来了收益。

Report v22 也移除了重复的 `input.retained_llm_calls` 和 `input.retained_tool_calls`。
`llm_calls` 与 `tool_calls` 统计最后阶段结束后保留的对话；该阶段没有结果时记为 null。
实际执行调用数仍覆盖所有阶段，包括已丢弃的轮次。

Report 和 summary 区分记忆指导造成的危害和执行中的失败。由记忆误导、
随后放弃或纠正且没有产生被使用发现的弯路计为危害。智能体自身的笔误、缺失工具
或无关环境失败本身不能证明记忆危害；应追踪纠正后的结果。Report v17 引入了这一区分，
因此 v16 的危害计数不能直接比较。不要混合比较不同评分
版本的效果标签或比率。旧报告保留为历史产物；memgraph 和 calibration 仅接受 report v26。
读取历史报告需使用与其版本匹配的旧代码。

{% hint style="info" %}
injudge 把一次运行的全部产物写入输出容器（`-o DIR`，默认 `output/`）下该次运行自己的 `<run-stamp>/` 文件夹——一次运行绝不覆盖另一次运行的产物。
{% endhint %}

| 文件 | Schema | 作用 |
|---|---|---|
| `injudge-report.json` | `injudge.report/v26` | 初始快照将待评区间标为 `unjudged`，运行结束时写出完整快照。评审期间，`checkpoint_journal` 指定保存已完成结果的文件。用 `--verdicts` 喂给 memgraph。 |
| `injudge-checkpoints.jsonl` | `injudge.checkpoint/v1` | 每个已完成区间追加一条记录。报告带有 `checkpoint_journal` 字段时必须保留；运行完成后作为审计记录保留。 |
| `summary.json` | `injudge.summary/v18` | 精简最终统计：计数、比率、投递估算及评审运行总量。 |
| `summary.md` | — | 同一批统计的简短可读摘要，列出应优先查看的区间、诊断计数，并链接到 JSON 文件。 |
| `memgraph-verdicts.html` | — | 仅在 `--html` 时产出（默认关闭）：召回节点按区间判定着色的 memgraph。 |
| `evidence/<span-dir>/` | — | 评审智能体实际运行的每个区间一个文件夹：它当时可用的原始文件。 |
| `agent-runs/<span-dir>.jsonl` | — | 每个送入评审智能体的区间一条轨迹：评判对话的 pi 事件流记录（见[评审智能体](judge-client.md#轨迹记录)）。 |

每条检查点记录包含 `schema`、`generated_at`、已完成的 `operation`、其 `injections`
及 `validation` 诊断。每次完成只追加该区间的数据，不重新构建或重写完整报告。

报告、检查点和离线包的 `generated_at` 使用 UTC。摘要复制报告时间戳；运行文件夹名使用本地时间。

正常完成或中断被处理时，injudge 写出完整报告并移除 `checkpoint_journal`。此时报告
可以独立使用，保留的日志不再参与加载。若进程在此之前被强制结束，应将初始报告与
同目录的日志一起保留。memgraph 和 calibration 的共享读取器应用完整日志记录，按
源区间顺序恢复诊断，并一次性重建汇总。末尾没有换行的记录会被忽略；完整但无效的
记录会使加载失败。恢复后的报告仅在内存中返回，加载不改写文件。若直接读取 JSON
且不重放日志，在完整报告写出前只能看到初始快照。

离线准备运行会在自己的运行目录下写出另一套产物：

| 文件 | Schema | 作用 |
|---|---|---|
| `episode-inputs.json` | `injudge.episode-inputs/v5` | episode 索引、执行/基准身份、边界来源、准备状态、附件状态、计数、审计投影与未归属诊断；不包含 verdict，其 `tool` 块带 `mode: "prepare-only"`。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/task.md` | — | 已录制的用户任务消息与后续更新，附来源。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/messages.md` | — | 必要时保存任务和步骤文件之外的输入消息，包含来源位置及首次可见的回合/步骤。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/deliveries.md` | — | 按时间排列的 delivery、搜索、槽位与曝光证据，也包含空 delivery。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/searches.json` | — | 该 episode 已记录的搜索及其计数。 |
| `episodes/<safe-episode-id>-<8 字符摘要>/manifest.json` | — | episode 身份、准备状态与诊断、计数，以及每个文件的种类、大小和一行说明（步骤文件另带来源边界）。 |
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

先读 `summary.md` 可快速了解运行结果。它使用与 `summary.json` 相同的计数，
显示运行日期、评审模型、判定表、条目 influence 与 acted-on effect 计数。有相关条目时，
还显示这些条目的 accuracy 与 novelty 分布计数，包括未知 novelty。最多列出
十个非 neutral 或出错的区间（含 episode id 与评审分析或错误）供优先查看；最多列出
三条不同的警告/错误诊断。输入诊断覆盖整份录制，包括选择范围外的 episode。
评审和录制中的文本在 `summary.md` 中经过转义，显式的 Markdown 链接、图片和 HTML 均保留为文本。

Episode 数与轨迹/区间数一起呈现。空投递和不可观察消费者分别计为 `empty`、
`unassessable`，不进入判定比率分母。完整诊断、记忆身份、episode 行和区间记录
保留在 report。`memgraph --verdicts` 只接受 `injudge.report/v26`。

### Report 的 aggregates

`aggregates.per_trajectory.<id>` 与 `aggregates.overall` 以 `injudge/aggregate.py`
的扁平形状保存同一批未投影的数字：

| 字段 | 含义 |
|---|---|
| `n_trajectories` / `n_episodes` / `n_injections` / `n_spans` / `restated_injections` | 总体计数；`n_trajectories` 仅全局 |
| `positive` / `negative` / `unresolved` / `neutral` / `error` / `unjudged` / `unassessable` / `empty` | 判定计数——这里是顶层键，不是 summary 的 `verdicts.counts` |
| `rates` | 已评判区间中 positive/negative/unresolved/neutral 的占比 |
| `cost` / `rubric` / `memory_usage` / `judge_compliance` / `mean_confidence_by_verdict` / `confidence_buckets` / `low_confidence` | 详细 report 块；summary 仅选择精简统计 |
| `input_paths` / `gold_scored` / `forced_answers` / `policy` | 仅全局 |

## 数字如何推导

从评分信号到使用标签与区间判定的映射见[评分规则与判定推导](verdict-derivation.md)。本节是这些标签所进入的计数契约；同一组函数（`injudge/aggregate.py`）产出两份产物的数字。

### 总体

| 总体 | 成员 |
|---|---|
| 已评判区间 | `positive` + `negative` + `unresolved` + `neutral`。判定率的分母。`unresolved` 指已采取行动但评审无法判定效果，或未在完整窗口上评估归因。中性包括选取、引用、归因不明、无效使用和未确立影响，以及按规则判定、不调用评审的无合格动作步骤区间（`auto: no_eligible_step`）。 |
| `empty` | 未投递任何条目（`auto: empty_delivery`）。不调用评审；单独计数和记录投递成本，不进入判定比率。 |
| `unassessable` | 所有投递都无法归属到 agent 步骤（`auto: no_subsequent_turn`），录制内容无法支持下游影响判断。计数与投递成本保留在独立类别中，不进入判定比率。缺少观察不代表测得零效应。 |
| LLM 已评判区间 | 无 `auto` 标记的已评判区间。评分比率、使用汇总与置信度的总体。 |
| `error` / `unjudged` | 只计数。永远不是比率分母。 |
| 重述注入 | 并入所属区间。计入 `n_injections` 与投递成本。不出现在比率中。 |

没有已评判样本时省略空的 `rates` / `rubric` / `confidence` 块。`cost` 只在两个总计都为零时省略。仅有 error 的运行只要有过投递仍带成本，计入 `error` 类。

### 判定率

分母是已评判区间；重述的重复注入绝不出现在比率中，它们已并入所属区间。

$$
\mathrm{rates}[v] = \frac{\mathrm{count}[v]}{\mathrm{positive} + \mathrm{negative} + \mathrm{unresolved} + \mathrm{neutral}}
$$

### 评分比率

总体是有效评审答案中的投递条目（`区间 × 投递条目`，按条目的区间本地 id 键控）。失败答案与机械跳过的区间不计入；为 null 的信号值绝不进入分母。

| 比率 | 分子 | 分母 | 读作 |
|---|---|---|---|
| `relevance` | `true` | `true` + `false` | 相关占比 |
| `accuracy_corroborated` / `accuracy_uncontradicted` / `accuracy_contradicted` | 各取值 | 有 accuracy 值的相关条目 | 三路拆分——绝不再合成单一的"accurate"比率：`uncontradicted` 不说明任何问题，折进去会让大量无法核实的运行报告 ≈1.0 |
| `novelty` | `novel` + `confirming` | 上述 + `redundant`，仅相关条目 | 相关条目中的非冗余占比 |
| `influence` | `referenced` + `selected` + `acted_on` | 上述 + `none` | 有任何足迹的占比 |
| `positive_effect` | `improved` | `improved` + `no_effect` + `harmed` | 有 effect 的投递中 improved 的占比 |

novelty 与 accuracy 比率只使用相关条目，其分布记在 `rubric.relevant_counts`。无关条目默认是新的且无法核实，计入它们只会衡量检索失误，而不是相关记忆带来了什么。`rubric.counts` 保留全部条目。`positive_effect` 只在有 effect 判断的投递上计算——除非 influence 为 acted-on，该信号为 null。未知新颖性（`null`）计入 `rubric.counts.novelty.null`，不进入 novelty 比率分母；全部未知时 novelty 比率为 null。相关但被忽略且 novelty 为 null 的条目使用 `ignored_unknown_novelty`。`rubric.counts` 中的信号枚举即[评分信号表](verdict-derivation.md#评分信号)中的值。

### 排查清单

`inaccurate_keys` 收集 `accuracy` 为 `contradicted` 的身份；`irrelevant_keys` 收集 `relevance` 为 `false` 的身份。每条带上产生它的区间 id。`shared_anchors` 按身份列出被两个及以上区间记为 `acted_on` 的同一动作步骤（`{traj, step, spans}`）：同一记忆以不同文本投递时窗口可以重叠，各区间判定照常保留，但共享步骤是一个动作，不是两份收益。每个投递条目恰好映射回其自身投递的一个 `(namespace, item)` 身份，因此一个 key 由多个身份投递时，每个身份都会入账。未携带已录制身份（无 `item_id`、无 key——如 redacted 匿名引用）的条目没有可联结的身份对：按区间作用域入账（其 `item_id` 记录操作与区间内 id），两个区间的无身份条目绝不合并。

### 逐记忆使用

只在 LLM 已评判区间上按 `(namespace, item)` 合并。对多 item 投递的联合判断会在每个身份下计数。当首尾两个*已录制* value 都存在且不同时，`value_drift` 为 true；redact（未录制）的末端不是 drift。

操作本地显示 ID 不作为汇总身份。规范内容身份生成 `content:<sha256>` 形式的条目 ID，原始引用保持完整。memgraph 的候选、投递与删除引用另带 `grouping_id`，用于浏览器与图注册表联结。

可通过 report 中的 `aggregates.overall.memory_usage` 将注入证据追溯到存储条目：每个 `(namespace, item_id)` 包含使用计数及对应区间 ID。在 `operations[].memories` 中定位该条目的区间本地 ID，再读取对应 `rubric` 的相关性、准确性和动作锚点，以及 `memory_exposure` 同一位置的曝光记录。这衡量的是采样任务中的可观察效用，不是存储内容的内在质量。被忽略且不相关的投递提示检索不匹配；相关但被忽略的内容可能重复或呈现不佳，两者都不足以决定删除记忆。未投递条目未被评估，裁剪预览不能评估隐藏内容，汇总前也要检查版本变化与共享动作锚点。

### 交付成本与策略

每个区间先累加全部投递（含重复投递）的已录制放置内容大小，再按 `chars // 4`
估算 token。整份估算计入该区间的判定类别。未记录大小的 redact 放置计 0 token，
并产生警告。error 与 unjudged 区间的成本保留在各自类别中。

negative 记忆包可能有一条记忆帮助了任务，另一条造成损害；positive 记忆包也
可能包含被忽略的条目。按判定分类的成本只表示各类别投递了多少文本，不能衡量
每条记忆的 token 收益，也不能衡量相对无记忆运行的额外成本。

投递策略度量记忆系统注入的轻重：按轨迹与整体，对照该轨迹全部已计费的 prompt token 预算，所有泳道的用量记录都计入，包括记忆泳道的提取调用。

计费预算取 `run.json`、`agent_end` 和已记录的 assistant 消息中最大的有效 prompt-token 总量。
两种快照均包含不产生对话轮次的 embedding 调用，用于替代消息总量，不与之相加。
无效或较小的快照会产生警告。两种快照均缺失时，消息用量只是下界；
已绑定 embedding 角色时会提示漏计风险。

历史重写会从输入中移除消息，但保留这些消息的费用。消息撤回只取消该槽位当前记录的费用。
槽位中后来的用户或工具消息不能取消更早的 assistant 费用。

### 置信度

均值与分桶（`[0,0.5)`、`[0.5,0.8)`、`[0.8,1.0]`）只覆盖 LLM 已评判区间。置信度从不加权计数。置信度 `< 0.5` 的区间列入 `low_confidence`。

### 输入口径

仅 overall：带 `input` 记录的区间——评审智能体实际运行的每个区间，含 error 结局——计入 `input_paths.agent`、`gold_scored` 与 `forced_answers`（`forced` 表示最终答案出自预算耗尽的强制作答轮）。按规则计数的空投递或不可归属区间，以及 unjudged 区间没有 `input`。

## summary.json 字段

Summary v18 只保留精简最终统计。条目、区间、评分说明、配置和来源记录仍在
`injudge-report.json` 中。

| 区域 | 内容 |
|---|---|
| `overall` | 计数、判定、rubric、估算投递 token 和评审运行计数。 |
| `trajectories` | 每条输入轨迹一行：`id`、`run_id`、已记录的 `instance_id` / `model` / `adapter`、计数、判定，以及投递大小非零时的 `approx_delivery_tokens`。 |
| `recording_policy` | 录制全量策略指标；`scope: whole_input_recordings` 包括评审选择范围外的 episode。驻留指标不含原始字符总量。 |
| `diagnostics` | 输入校验与评审诊断的 `error`、`warn`、`info` 记录数。完整消息保留在 report 中。 |
| `selection` | 源报告的选择范围；存在时保留。 |
| `notes` | 估算口径与区间判定范围的简短说明。 |
| `report` | 同一运行目录中的详细记录 `injudge-report.json`。 |
| `generated_at` | 源报告的 UTC 时间戳，原样复制，不转换时区。 |
| `tool` | 仅 `name`、`schema`、`source_report_schema` 和 `judge_model`。 |

比率保持数值小数。未知值保持 `null`，不写成零。投递与驻留 token 按 `chars // 4`
估算，不是计费 token，也不是相对无记忆运行的额外开销。区间判定衡量记忆影响，
不是 benchmark 任务成功率。

### 计数与判定

`overall` 和每条轨迹行都含以下字段：

| 字段 | 含义 |
|---|---|
| `counts.n_trajectories` | 输入轨迹数；仅全局。 |
| `counts.n_episodes` / `n_spans` / `n_injections` / `restated_injections` | 选择范围内的 episode、区间、投递及重复投递数。 |
| `verdicts.counts` | positive、negative、unresolved、neutral、error、unjudged、unassessable 和 empty 区间数。 |
| `verdicts.judged_spans` | positive + negative + unresolved + neutral；判定比率的分母。 |
| `verdicts.rates` | 四类已评判区间的占比；没有已评判区间时省略。 |

empty、unassessable、error 和 unjudged 只保留计数，不进入判定比率分母。
重复投递计入投递成本，但不增加已评判区间数。

### 全局 Rubric、投递与评审

| 字段 | 含义 |
|---|---|
| `rubric.judged_entries` | 有有效评审答案的投递条目数。 |
| `rubric.counts` / `rubric.relevant_counts` / `rubric.rates` | 五信号分布、相关条目的 novelty 与 accuracy 分布，以及比率。每项比率以该信号的已评估条目为分母，排除 null；novelty 与 accuracy 比率只用相关条目。 |
| `rubric.no_footprint_reasons` | 可选计数：不相关、重复、确认但被忽略、新颖但被忽略、新颖性未知且被忽略。 |
| `rubric.shared_action_groups` | 含有被多个区间共同记入的动作的记忆组数，不是共享步骤数；可选。完整锚点保留在 report。 |
| `delivery.approx_tokens_total` / `approx_tokens_by_verdict` | 估算投递 token 总量及各判定类别的分量；录制投递大小非零时保留，即使估算值舍入为零。 |
| `judge.attempted_spans` / `gold_scored` | 评审智能体实际运行的区间数，含 error；附带 gold 运行的区间数。 |
| `judge.spans_with_followups` | `attempts > 1` 的区间：至少一个被拒答案进行了跟进。 |
| `judge.forced_spans` | 任一阶段被强制作答的区间数，不限于最终阶段。 |
| `judge.context_rollbacks` | 所有评审阶段被回滚的轮数。 |
| `judge.executed_llm_calls` / `executed_tool_calls` | 所有阶段实际执行的调用数，含回滚工作。 |
| `judge.reported_usage` | 上报的 `input`、`output`、`cache_read`、`cache_write` token。 |
| `judge.effect_pairing_corrected_spans` / `low_confidence_spans` | 效果配对被纠正的区间数、置信度低于 0.5 的区间数；非零时可选。 |

没有已评判条目时省略 `rubric`。评审智能体未运行时省略 `judge`。
任一评审阶段缺少某项执行计数或 token 用量，或上报值无效时，该项总量为 `null`。
后续有效值不能补齐缺口。runner 没有返回可读取结果时，该区间的实际调用数、回滚数
和上报 token 总量全部为 `null`，包括先前阶段已上报的总量。任一已运行区间的某项
总量未知时，summary 中该项总量也为 `null`。stdout 将这些未知执行计数显示为
`unknown`。上报用量与投递估算衡量不同数量，不能相加为一项成本。

`recording_policy` 保留注入密度、重复比例、投递 token 比率、搜索/提取数、提取失败
和驻留估算。其完整录制范围不同于 `overall` 的评审选择范围。已记录的 null 保留；
估算说明只在顶层 `notes` 出现一次。

## 仅 Report 中的字段

`injudge-report.json` 的每个操作还带有：`kind`（`recall_injection_span`）、`traj`、`seq`、`n_injections`、`steps`（无投递归属到 step 时省略）、`payload_summary` 和 `delivery_cost`（`deliveries`、`content_chars_total`、`approx_tokens_total`）；`episode_execution_id` 与 `episode_last_step`；`memories`（带区间本地 id 的完整已录制引用）；`memory_exposure`——与 `memories` **按位置平行**的列表（下标 `i` 描述 `memories[i]`），每条为 `{state, exposed_chars, value_chars}`，state 取 `full`、`full_normalized`、`clipped` 或 `unknown`，未核实的规范值不会进入评审证据；以及逐条目的 `memory_windows`（全局步骤上界，至迟止于该 episode 的最后可用步骤；同 episode 的后续投递也可提前截断），`memory_window_starts` 记录延后或为空的动作窗口，`memory_window_owners` 将受影响条目映射至更早归属的 `{span_id, memory_id}`（独立的重新投递会另起区间并重建基线），`memory_variant_steps` 记录窗口内同一记忆以不同或更窄文本再次投递的步骤（不会截断窗口），`memory_followups` 记录经过工具结果来源证明的显式选取 `{step, delivery_step, tool_call_id, operation_id}`，动作步骤与结果投递步骤分开记录。与 `memory_exposure` 不同，`rubric`、`memory_usage` 和 `memory_windows` 按区间本地 id 键控。评分 note、动作 `anchors`、`analysis`、`evidence`、`step_gaps`、`auto`、`error`、`judge_raw` 和适用时的 `pe_pairing_corrections` 仍保留。评审智能体实际运行的区间——含其自身的 error 结局——带有 `input`：`path: agent`、尝试次数、保留的 `llm_calls` / `tool_calls`、`read_steps`（步骤文件全文及 manifest 中 `result_sources` 指向的结果章节均保留在对话中的全局步骤；目标投递已在开场输入中，无需重复读取——每个被引用的锚点及结构化 evidence 步骤都必须在其中）、实际执行调用数、包括回滚调用的上报用量、上下文回滚、`any_phase_forced`，以及相对于运行文件夹的 transcript 路径；在智能体运行前就被拒的区间（episode 边界不安全或不完整，或父 episode 没有已录制任务）取 error 判定，没有 `input`、没有 evidence 文件夹、也没有 transcript。`gold` 与仅代表最终答案的 `forced` 在适用时出现。轨迹行带有 `id`、`run_id`、`path`、`instance_id`、`model`、`adapter`、`n_steps`、`n_injections`、`n_spans`、其 `episodes` 行以及该轨迹的 `policy`；episode 行记录边界、任务、附件与证据来源，但不产生 episode verdict。

report 的 `tool` 块给出工具名（`name: "injudge"`）与其 `schema`，并带有已解析、无密钥的评审智能体配置（`config`，端点 URL 仅保留源站 `scheme://host[:port]`）以及 `judge_model`。
路径、查询参数、片段与用户信息可能包含秘密，因此不予保留。若已知的评审凭证出现在源站中，也会被打码。

`injections` 数组列出每一次原始投递及其所属区间、该区间的 `episode_execution_id`（若有时）、`role`（`first` 或 `restated`）、`verdict`、`content_chars`，以及 `op_id` / `seq` / `step`。

## 来源记录与解释

Report v23 的 `input.read_steps` 同时证明步骤文件全文及其引用的结果章节已读取。
v22 只证明步骤文件本身；当前读取器不接受或重新解释旧版本。Summary v18 保持不变，
`shared_action_groups` 仍统计记忆组数，不是共享步骤数。

`tool.provenance` 保存 implementation_sha256 和 prompts_sha256；执行过的区间
input 保存 evidence_sha256 和 system_prompt_sha256。它们标识当前代码、提示词及
实际评审输入，不含凭据，也不证明预置记忆库的来源日期或无污染。

influence null 单独计数，不进入已评估影响的分母。selected 的效果为 null，不能
继承后续读取的收益；no_effect 派生为中性的 unproductive，misused 必须为 harmed。
neutral 包括选择、提及、无收益使用和未建立影响；unresolved 包括归因或效果不确定。
符号计数与 token 代理指标不是可相加的系统净收益。对泛化的结论须区分同任务答案
复用与跨任务迁移，不能只凭文本重合或日期猜测来源。


### 投递量、上下文驻留与反向记忆评估

v17 将 `injected_token_share` 改名为 `delivery_token_ratio`。
report 的 `aggregates.overall.policy.residency` 按每个主 lane 输入视图累计有宿主证明的投递文本，输出
`proven_resident_chars`、`approx_resident_tokens`（字符数除以 4）、
`observed_input_views`、`views_with_memory`、`views_with_gaps` 和
`unresolved_placements`。同宿主的重叠片段取并集。每个放置文本必须在原始宿主中
唯一匹配一个区间，证据移除也使用同一匹配规则。其他放置未能确定时，已证明的
区间仍计入驻留量。同一宿主出现
可以跨 episode 边界持续可见；重写或撤回会终止对应出现。
未证明、缺失或无新证明的重录历史可能漏计。这些估算均不是精确计费或相对无记忆
基线的因果开销。policy 总量始终覆盖完整输入轨迹，包括 `--episode` 之外的部分。

未选区间保留连接键和成本的紧凑 `unjudged` 记录，不再重复全文。汇总统计、summary
计数和 dry-run 区间计数均排除这些区间。已评判答案的分析文本不会使其区间
被排除在统计之外。`tool.selection` 说明选择范围。未选 episode 不产生缺失附件
警告，但执行身份歧义检查仍保留。

`memory_usage` 按存储项反查已评估证据。逐项保留 `episodes` 和 `exposures`；
各 exposure 分开记录版本/内容身份、暴露范围、`judged_entries`、区间链接和信号计数。
这些逐记忆记录保留在 report，summary 不重复它们。这些是特定投递条件下的观察，不能从预览推断未见全文，也不能把重复投递
当独立试验，或否定
从未投递的记忆。负面记录只用于人工复核，不自动触发删除。
