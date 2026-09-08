---
description: "LLM 所打分的五信号评分规则，以及 injudge 如何在代码中推导使用标签与区间判定。"
icon: code-branch
---

# 评分规则与判定推导

{% hint style="info" %}
LLM 为每条投递记忆打五个信号，但从不选择区间判定。injudge 在代码中推导使用标签、足迹原因与区间判定（`injudge/judge.py`）。
{% endhint %}

这些标签如何计入比率与汇总，见[产物](artifacts.md#数字如何推导)。

```mermaid
flowchart TD
    llm["LLM：每条投递记忆五个信号"]
    pair["配对校正：positive_effect 仅与 acted-on 的 influence 同时出现"]
    usage["_derive_usage → used / misused / noted / ignored / unclear"]
    reason["_footprint_reason，仅 ignored 记忆"]
    verdict["derive_verdict：任一 misused → negative，否则任一 used → positive，否则 neutral"]
    agg["汇总：比率、评分投影、成本、memory_usage"]
    llm --> pair --> usage
    usage -->|ignored| reason
    usage --> verdict --> agg
```

## 评分信号

每个投递条目在其区间本地 id（`m1`、`m2`……按投递顺序）下独立评分：两个条目可能共用一个显示键——不同的已存记忆，或同次投递的两个版本——绝不折叠为一次评判。note 引用步骤流中的 step 与 seq。录制允许时，每个条目还带**曝光标记**，标明 placement 实际向 agent 展示了其值的哪一部分；评审只评判已曝光的内容，绝不评判未展示的后缀。

| 信号 | 类型 | 问题 |
|---|---|---|
| `relevance` | bool | 注入点上，记忆是否与任务及 agent 当前子目标主题相关？依据区间首步之前的步骤判断。 |
| `accuracy` | enum | `corroborated`（有可观察证据为真）/ `uncontradicted`（无从判断）/ `contradicted`（有可观察证据为假）。 |
| `novelty` | enum | `novel`（任务陈述与注入前步骤中都未出现）/ `confirming`（敲定 agent 明确表达过的疑点）/ `redundant`（早已确立，无疑点）。任务陈述与 agent 自己的注入前回合一律视为已在对话中。 |
| `influence` | enum | `none` / `referenced`（仅被提及）/ `acted_on`（注入后的命令、编辑或测试用到了记忆所携带、且 agent 无法从已有信息推导出的内容，或 agent 明确引用了该记忆）/ `decisive`（该记忆改变了 agent 计划的方向——可论证的路线转变）。足迹是后续行为：从区间首步起算，以该记忆下一次被重投递处为界（重投递即取代；从未被重投递则无上界；其他记忆的重投递只收窄其自身窗口）。回声规则：agent 在区间之前已宣布、推理出或已应用的方向，不能归功于事后再投递的记忆——即使结果动作落在足迹窗口内。另有两条约束：上下文不是记忆内容，由来源的时间与角色判定——任务陈述与注入前步骤是 agent 在投递到达前已有的信息；搜索查询是检索元数据，主 agent 未必见过（它可能源自 memory 或 side lane，仅凭查询不能确立 agent 已知什么）；黄金答案是 agent 从未见过的评测方专有核验材料，而注入后步骤恰是评判足迹的证据所在；`acted_on`/`decisive` 要求一个**被改变的动作**，而非继续执行注入前已确立的计划。`redundant` 的记忆仍可能被 acted on——当投递本身改变了动作（提醒恢复了已放弃的方向，或补足了计划缺少的执行细节）：novelty 与 influence 相互正交。acted-on 或 decisive 断言必须携带**锚点**——足迹出现的 agent 步骤号，且落在该条目的窗口内；塑形动作恰好发生在投递步本身的那次响应时，投递步也算锚点。 |
| `positive_effect` | enum 或 null | `improved` / `no_effect` / `harmed`，以该记忆窗口内可观察的下游事件判断（`improved`：下游事件把收益系于记忆塑形的动作；`no_effect`：该动作没有产生可观察的下游变化；`harmed`：错误位置、错误线索、回退正确工作）。仅当 influence 为 `acted_on` 或 `decisive` 时非空。 |

{% hint style="warning" %}
`positive_effect = no_effect` 对记忆系统不利：上下文与注意力被占用，工作并未改善。仅从某动作缺失推断出的跳过，记为 `influence = none`；缺失本身不是证据。
{% endhint %}

判定规则刻意不对称：被无视的垃圾注入是 **neutral**（无足迹），只有 acted-on 且无效（或有害）才是 **negative**。因此 negative 率为零绝不能单独读作"无害"：报告中它始终与 wasted-token share 一起出现——后者本身是区间级代理（neutral 区间里被引用的记忆计为浪费；positive 区间里被无视的记忆计为有效），每一个报告的份额都附带该警告（见[产物](artifacts.md#判定率)）。运行中完全没有可归类的 agent 回合、因而观察不到任何后续的区间，不经 LLM 直接规则判定，并作为 **unassessable**（观察缺失，而非测得的零效应）单独报告，不进入任何方向性分母。

最终任务结果若可辨认，可以抬高或压低置信度，但不能单独决定任何信号。

## 分工

| 由 LLM 产出 | 由代码推导 |
|---|---|
| 逐记忆的 `relevance`、`accuracy`、`novelty`、`influence`、`positive_effect`（value + note） | 逐记忆的 `usage`（`used` / `misused` / `noted` / `ignored` / `unclear`） |
| 区间级 `analysis`、`evidence[]`、`confidence` | 被忽略记忆的 `reason`（`irrelevant` / `redundant` / `confirming_ignored` / `ignored_novel`） |
| | 区间 `verdict`（`positive` / `negative` / `neutral`） |
| | 投递成本、策略指标，以及全部 `summary.json` 比率 |

`accuracy` 会上报（`inaccurate_keys`），不进入判定路径。`relevance` 与 `novelty` 进入判定路径的唯一方式，是作为被忽略记忆未留足迹的原因。

## 组装

语义检查分两层，都在代码里完成。

**尝试循环内**（`_verdict_semantics`，由 `judge_span` 在 `_strict_parse` 之后对每个最终答案调用）——这些检查是区间相对的（评审无法先验地知道本区间的 id 与窗口），因此走有界的纠正重试；耗尽预算即为 `error` 判定，绝不留下静默不完整的判定：

1. **精确 id 覆盖。** 答案为每个投递条目的区间本地 id 恰好各给一条——缺失条目是未作答的评判，未知条目是噪声，两者都不被接受。
2. **动作锚点。** `acted_on`/`decisive` 的断言至少给出一个锚点步骤，且所有锚点都落在该条目的足迹窗口内（区间首步起，至下一次重投递减一；从未被重投递则至步骤流的最后一步——超出步骤流末尾的步骤是幻觉，不是证据）。区间首步之前的锚点与其他越界锚点一样被拒绝，对任何条目都如此：注入前证据可以支持 novelty 或 accuracy 的 **note**，但永远不能作为锚点——锚点是 influence 证据。`influence` 为 `none` 时 `anchors` 必须为空（无足迹可引）；同一步投递冲突（后一区间在本区间末投递步上再次投递同一记忆）下，共享步骤归先投递方，后一区间的该条目窗口自共享步骤之后才开始——记录不含同一步内的因果顺序，因此该步骤按先 carrier 规则只计一次。
3. **信号正交。** novelty、accuracy、influence 与 effect 各自独立判定：任何跨信号组合都不构成契约违例。提醒效应（`redundant` + `acted_on`）、部分有误却促成了修复的记忆（`contradicted` + `improved`）、真正共享的足迹（两个条目携带相同内容，记入 shared 说明）都是可报告的状态，规则不得压制。

**组装时**（`_assemble_verdict`），作用于已验证的对象：

4. **配对。** `positive_effect` 仅在 `influence` 为 `acted_on` 或 `decisive` 时适用。该情况下 effect 为 null 是评分规则允许的不确定性出口（证据无法判定方向）：usage 推导为 `unclear` 并记录在案，但不算配对纠正。influence 为 `none` 或 `referenced` 时仍给出的 effect 会被强制改为 `null`（记警告并计入配对纠正）。
5. **使用、原因、判定。** `_derive_usage` 为每个条目打标签；`_footprint_reason` 只对 `ignored` 运行；`derive_verdict` 读取使用标签。置信度被夹到 `[0, 1]`。

完全不含投递记忆的区间不是错误：记警告并推导为 neutral，因为没有可打分对象。

仅供展示的字段（`analysis`、证据 claim）会被复制并打码。畸形的证据列表会连同警告被丢弃，不使区间失败。

## 使用标签

`_derive_usage` 在配对校正之后读取 `influence` 与 `positive_effect`：

| `influence` | `positive_effect` | `usage` | 含义 |
|---|---|---|---|
| `acted_on` 或 `decisive` | `improved` | `used` | 记忆塑造了后续行为，并改善了工作。 |
| `acted_on` 或 `decisive` | `no_effect` 或 `harmed` | `misused` | 记忆塑造了后续行为但没有改善工作，或把 agent 带偏。 |
| `acted_on` 或 `decisive` | `null` | `unclear` | 已 acted-on 但缺少 effect 判断。 |
| `referenced` | （忽略；通常为 `null`） | `noted` | 仅被提及。行动并未因此改变。 |
| `none` | （忽略；通常为 `null`） | `ignored` | 没有可观察足迹。 |
| `null` / 未知 | 任意 | `unclear` | influence 本身未知。 |

使用条目上的 evidence 字符串，取 `positive_effect`、`influence`、`novelty` 中第一条非空 note。

## 足迹原因

`_footprint_reason` 只对 `ignored` 记忆运行。严格解析保证此时 `novelty` 是枚举值（覆盖缺口的全空条目会推导为 `unclear`，不会变成 `ignored`）。

| `relevance` | `novelty` | `reason` |
|---|---|---|
| `false` | 任意 | `irrelevant` |
| `true` | `redundant` | `redundant` |
| `true` | `confirming` | `confirming_ignored` — 敲定了未决疑点，但从未被归功 |
| `true` | `novel` | `ignored_novel` |

这些原因填入 `rubric.no_footprint_reasons`。它们不改变区间判定：被忽略的记忆已经不在 `used` / `misused` 之列。

## 区间判定

`derive_verdict` 读取已推导的使用标签。损害优先：

1. 任一记忆 `misused` → **negative**。
2. 否则任一记忆 `used` → **positive**。
3. 否则 → **neutral**。

`noted`、`ignored` 与 `unclear` 单独不会产生 negative。因此仅被提及永远不会得到 negative 区间。

```python
usages = [entry["usage"] for entry in memory_usage.values()]
if "misused" in usages:
    return "negative"
if "used" in usages:
    return "positive"
return "neutral"
```

### 示例

每行是一个区间。未列出的记忆表示不存在。

| 记忆 | 推导出的 usage | 区间判定 | 原因 |
|---|---|---|---|
| A：`acted_on` + `improved`；B：`none` | A `used`，B `ignored` | **positive** | 至少一个 used，没有 misused。 |
| A：`acted_on` + `no_effect` | A `misused` | **negative** | 被采纳但没有改善：上下文被占用却无收益。 |
| A：`acted_on` + `harmed` | A `misused` | **negative** | 主动损害。 |
| A：`decisive` + `improved`；B：`acted_on` + `harmed` | A `used`，B `misused` | **negative** | 损害优先于帮助。 |
| A：`referenced` | A `noted` | **neutral** | 提及但无行动。 |
| A：`acted_on` + `null` | A `unclear` | **neutral** | 有影响但无 effect 判断；不计为 used 或 misused。 |
| A：`none`，`relevance=false` | A `ignored`（`irrelevant`） | **neutral** | 无足迹。 |
| 其后没有 agent 回合（`auto`） | 占位 `unclear` | **neutral** | 规则判定；无 LLM 调用。见下。 |

## 规则判定与失败区间

| 情形 | `verdict` | 评审调用 | 汇总中的评分 / 使用 |
|---|---|---|---|
| 没有可归属的 agent 步骤 | `neutral`，`auto: no_subsequent_turn` | 无 | 占位的空评分与 `unclear` 使用。不计入评分比率、使用汇总与置信度。在汇总中计为 **unassessable**——不进入任何方向性比率，也不计入浪费 token 份额。 |
| `--limit` 截断或 `--dry-run` | `unjudged` | 无 | 只计数。不是比率分母。 |
| 覆盖/锚点违反且纠正重试耗尽、契约失败 | `error` | 已尝试 | 只计数。保留原始文本。不会被改写成 neutral。 |

这些标签如何计入 `summary.json` 的比率、成本与汇总，见[产物](artifacts.md#数字如何推导)。
