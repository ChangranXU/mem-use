---
description: "五个独立信号、证据覆盖与确定性的使用标签和区间结论。"
icon: code-branch
---

# 评分规则与结论推导

评审按区间内条目 id（`m1`、`m2` 等）逐条评分；显示 key 只是标签。
Python 验证答案并推导使用标签和区间结论。任务最终成功与否不能决定记忆结论。

## 五个信号

| 信号 | 值与含义 |
|---|---|
| `relevance` | `true` / `false`：实际暴露内容是否适用于当前任务或子目标。错误建议仍可能相关；名称相同不等于相关。 |
| `accuracy` | `corroborated` / `uncontradicted` / `contradicted`：在该主张所属仓库、版本和情境内，证据确认、未确认也未推翻、或推翻主张。 |
| `novelty` | `novel` / `confirming` / `redundant` / `null`：相对于暴露前的记录。任务给出预期结果不等于已给出解法。多主张条目只有所有实质主张均已成立才是 redundant。更早的投递只有仍在消费输入中时才算基线（`span.md` 会说明）；随同一输入到达的副本不算更早。 |
| `influence` | `none` / `referenced` / `selected` / `acted_on` / `null`，见下文。 |
| `positive_effect` | `improved` / `no_effect` / `harmed` / `null`，仅在 acted_on 时可非空。按锚定行动的可观察后果评分。保留或提交改动本身不算收益。条目塑造的任一行动造成危害即为 `harmed`，即使此前有收益。 |

新颖性、准确性、影响和效果彼此独立。重复提醒可能改变行动，错误主张也可能
促成有效诊断。局部测试通过只证明它检查的行为，局部改善可以与任务失败同时存在。
没有记录结果的执行只能证明尝试，包括提交尝试。参考解只是一个有效解法；有无
参考解使用同一套信号定义。gold 模式只多一条规则：gold 用于判断相关性和效果，
验证、环境修复或避开死路即使不在 gold patch 中也可能有价值，与 patch 一致也不
证明行动由记忆引起。最终结果仅用于佐证。

## 影响类别与范围

- `none`：完整检查窗口后，没有建立引用或可归因行动；不等于证明效果为零。
- `referenced`：提及了记忆，但完整检查窗口后仍未建立行动变化。提及必须单独指向该条目
  （其显示的标识符、独特内容或列表位置）；对整个结果列表的笼统评论不指向任何条目，
  除非只显示了这一条。
- `selected`：暴露的标识符或预览引起更完整记忆内容的检索，且该内容被记录为
  独立的记忆投递，尚无其他内容使用。锚定选择行动而非结果送达步骤；效果为 null，
  返回内容由那次投递评估。检索结果不是已记录投递时，或依照记忆给出的独特路径
  读取仓库代码时，属于 acted_on：没有其他区间评估返回内容，其后果就是本条目的效果。
- `acted_on`：独特暴露内容、明确归因或有证据的方向改变塑造了行动。计划须有
  后续执行支持；常规实践或措辞相似不能单独建立使用。
- `null`：多来源之间归因不清，或调查不完整。指出竞争来源或缺失证据；效果为 null。

v15 移除了冗余的 `decisive` 档位；方向改变写在 acted_on 的说明中。预览即使未被
打开也可能直接影响修改，但必须为该独立行动给出锚点。预览与其选中的、已记录的完整
投递共有的内容归完整投递计功：预览保持 selected，除非它在完整投递到达前或以完整
投递没有的内容塑造了行动；评审完整投递时也不把选中它的预览当作竞争来源。否则两个
区间会互相推让，记忆得不到任何功劳。这只分配先对照任务、代码和既有推理确立的影响，
既不制造影响，也不转移完整读取到达前已形成的行动。同一步可能同时有选择与其他
行动，因此不能只凭后续选择步骤号决定标签。非 acted_on 的影响若给出非空效果，
组装时强制为 null 并计入配对修正（评审合规度）；另外声称的内容使用必须标为
acted_on 并给出自己的锚点。

首次消费步骤已经看到记忆；此前计划必须由更早步骤或暴露前输入证明。逐条目窗口
约束行动锚点，父 episode 内更晚的结果可以证明既有锚点的后果，不必成为新的行动
锚点。不同或更窄的后续暴露可以保留重叠窗口。重暴露截断是归因约定，不是影响消失
的证明。

选择和内容使用是不同阶段，其计数不能相加当成独立收益。`memory_followups` 保留
确定性选择链接；`rubric.shared_anchors` 标记同条目同一步骤的跨区间信用，只是复核
线索，不是完整的因果去重算法。

## 证据与输出验证

答案必须精确覆盖条目 id、使用合法信号值和窗口内锚点。selected/acted_on 必须有
锚点，none 必须为空。所有锚点和 `evidence[].steps` 引用都要求完整步骤文本留在
评审对话中。

读取验证将工具实际返回的编号文本与证据文件比较。分页读取可合并；截断行、仅标题、
搜索命中、失败工具结果或已回滚读取不能证明完整步骤。`baseline.md` 中索引过的完整
历史步骤副本与原步骤文件等价；省略稍后工具结果的暴露前副本不能证明完整步骤。

每份答案必须读完首次消费步骤；none/referenced/selected 以及 improved/no_effect 效果还必须
读完整个条目窗口，因为之后的有害使用会决定效果。harmed 仍须读完首次消费步骤及所引步骤，
但不要求读完剩余窗口。若 acted_on 已成立而效果调查不足，保留 acted_on，并用
positive_effect null；归因尚未评估时用 influence null。不能为了结束预算受限的调查而制造
中性或收益结论。

违规后在相同历史上进行一次无工具修复，只有 `needs_evidence` 明确请求才重新调查。
修复无效或预算耗尽为 error，不会悄悄变为 neutral。措辞检查只是提示，不改写标签：
三个词及以上的引文片段必须逐词出现在某个证据文件中（忽略标点、引号、不可见格式字符和
大小写；省略号和方括号插入表示省略），analysis 超过 60 词也会提示。

## 使用标签与区间结论

| influence | positive_effect | usage |
|---|---|---|
| acted_on | improved | used |
| acted_on | no_effect | unproductive |
| acted_on | harmed | misused |
| acted_on | null | unclear |
| selected | null | selected |
| referenced | null | noted |
| none | null | ignored |
| null | null | unclear |

unproductive 需要行动没有有用后果的肯定证据，不等于伤害。被否定的调查可能通过
消除疑问改善工作，开销单独记录。misused 只表示造成伤害的使用。

保守的区间复核规则：任一 misused → negative；否则任一 used → positive；否则
任一 unclear → unresolved；其余 neutral。neutral 包括选择、提及、无收益使用和
没有建立影响。有益与有害混合的包为 negative，但保留每条标签；这个符号不是净效用。

ignored 的原因仍为 irrelevant、redundant、confirming_ignored、ignored_novel 或
ignored_unknown_novelty，不决定区间符号。

空投递和无法归属消费者的记录无需付费调用，统计为 empty/unassessable。未选中或
受 limit 限制的区间为 unjudged；评审失败为 error。它们不进入方向性比率。influence
null 单独计数，不进入已评估影响的分母。

report v15 与 summary v13 明确改变类别及完整 read_steps 的含义，不会默默重解释
旧产物。参见[产物](artifacts.md)和[评审证据](scoring-input.md)。
