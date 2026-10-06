---
description: "轨迹必须包含什么、记忆身份如何定义，以及损坏输入如何处理。"
icon: file-lines
---

# 输入约定

{% hint style="info" %}
两个工具都读取 **schema-v10 gate 轨迹**。记忆动作完全来自录制器的 `memory_*` 注解事件——绝不解析提示词、角色名或标签约定。录制器的 `bridge_*` 生命周期事件不产生记忆操作。成功的 bridge 删除若缺少来自同一会话、同一生成操作的 `memory_*` 退役记录，会给出警告；bridge 原生版本 ID 不能替代可移植 MemoryRef。其余 bridge 事件被跳过；`run.json` 记录的 `schema_version` 不是 10 时会给出版本警告，不提供兼容解析器。
{% endhint %}

保留归档轨迹及其原始验证日志。版本警告不会使历史验证失效；重新验证应使用匹配的录制器版本。不要将旧产物的版本号改为 schema 10。

`injudge --prepare-only` 是离线 episode 准备路径。Episode 范围来自
`bridge_episode_open` / `bridge_episode_close` 配对，没有记录时则来自明确
属于某 episode 的 `memory_session`；无论哪种来源，生命周期事件都只提供范围
与来源，记忆操作和 MemoryRef 仍然只能来自 `memory_*` 注释。准备产物会记录
未解决的范围边界与归属缺口。

没有注解的轨迹是合法输入：运行得到零个记忆操作，外加一条 validation 说明，而不是报错。

Schema 10 的 `memory_source_set` 保存用于审计的 bridge 来源历史，不是记忆操作或模型可见的交付内容。memgraph 和 injudge 仍使用记忆操作、MemoryRef 和放置证据；来源快照不改变身份、成本、曝光或评分。快照的哈希、作用域与引用由 traj-recorder 的验证工具检查。

## 注解事件

| 事件族 | 记录的动作 |
|---|---|
| `memory_session` / `memory_role_bind` | 会话元数据与泳道绑定 |
| `memory_generate_*` | 记忆创建（produced 与 retired 引用） |
| `memory_change` | 生命周期更新与删除 |
| `memory_search_*` | 检索，每个读取泳道一个操作（`lane_id`） |
| `memory_delivery` | 召回注入：每个读取泳道一条 `searches` 条目，每个渲染槽位一条 `placements` 条目。只有记录为 `status: "placed"` 的投递才产生召回操作；其他状态都会收到警告并被跳过 |

可选的逻辑角色 `embedding` 绑定独立的代理泳道。生成或搜索结束事件可记录该泳道自己的游标、`embedding_call_refs` 和 `embedding_binding_status`；它们不会替代记忆调用引用，也不会产生 agent 步。记录 embedding 游标时，状态和引用列表必须同时存在；`no_call` 也必须显式记录空的 `embedding_call_refs`。损坏的 embedding 绑定会产生 validation 警告。

Placement 保留槽位的记录顺序、名称、种类与全文（全文不可用时保留内容摘要）。槽位安排不同，不会合并为同一评审处理；曝光在各槽位内比较，不跨消息拼接。没有泳道被读取时，空 `searches` 合法。空 `placements` 记录零内容；仅部分槽位有大小时，保留已知成本并警告遗漏的部分。

Generation 的 `retired` 引用补充 delete-change 记录，同一版本只计一次。退役可区分的旧版本不会删除当前条目：原始退役仍保留在操作中，但不添加条目删除边或历史。退役当前版本仍会将条目标记为已删除。状态以最新观察到的候选版本为准；不同版本的退役会明确记入说明，因为不透明的版本 ID 不能推断未观察到的更新顺序；同一生成操作为此前已观察到的退役版本产出可区分的替代版本时属于普通更新，不记说明。即使某个引用没有内容摘要，原生版本 ID 仍可确认这种匹配；操作局部 ID 只能通过已记录的摘要比较。同一生成操作中已记录的连续更新所产生的中间版本也适用此规则。重新产出并退役当前版本仍会删除该条目。

生成操作与每条轨迹的搜索表保留开始和结束事件的 `extensions`；同一命名空间内以结束事件的字段为准。这样原生工具参数可与结果、用量一起提取，无需在每次快照投递中重复搜索元数据。搜索表的 `delivered_by` 引用最终图中的操作 ID。即使召回以空结果完成，已记录的 `error_codes` 也会保留，因此仍能区分原生超时与正常的无匹配结果。

## 记忆身份

记忆以录制器可移植的 MemoryRef 字段区分，绝不以适配器词汇归组。

| 层级 | 键 | 用途 |
|---|---|---|
| 条目 | `(namespace, item_id)` 或稳定显示键 | 图注册表、使用汇总、区间集合差与足迹窗口。跨版本保持稳定；用作显示键的操作本地 ID 不参与比较。 |
| 内容回退 | 隐去本地 ID 后的规范条目身份 | 仅有操作本地显示键时的跨操作归组。相同内容更换本地 ID 后仍归为一组，不同内容保持独立。 |
| 版本 | `namespace` / `identity_scheme` / `version_id` / 内容 sha256，外加记录到的条目身份（`item_id`，否则为显示键回退） | 区间指纹与候选去重。 |

不同 store 铸造的同名显示键，或同一 store 中不同条目持有的同名键，是两条不同的记忆。

投递与搜索的一致性检查还比较每个 MemoryRef 的 `lane_id`，与录制器按泳道区分的身份规则一致。返回的记忆必须属于被搜索的泳道；另一泳道的相同版本不能满足投递的有序子序列检查。

被搜索泳道取自搜索开始事件，且仅在两个泳道 ID 都为字符串时检查是否一致。跨会话搜索链接仍保留展示，但不提供查询或匹配数量。重复的操作链接只贡献一次查询和匹配数，记录到的链接仍保留展示。重复或无法解析的链接会使该投递的子序列检查失效，而不是扩大其搜索结果。

适配器扩展（`extensions.cure`）只贡献**展示**投影：key、`memory_type`、confidence、评审状态、needs_verification、拒绝计数、声明的提取步骤。可移植字段仍是回退，因此非 CURE 适配器也能解析。

{% hint style="info" %}
**区间指纹回退。** 没有可移植版本身份的记忆（`operation_local` 的 version id 配上未摘要的内容）改用投递的 `(namespace, item, value)` 作指纹。被 redact 的内容不记录 value，因此同一 `(namespace, item)` 下不同的 redact 正文会共用一个区间：录制结果无法区分它们。

没有稳定条目键时，图、使用汇总、区间集合差和足迹窗口采用与指纹相同的规范条目身份。因此，更换本地 ID 后采用新处理方式再次投递，在其重新显示早前处理方式的文本时仍会截断该窗口（不同或更窄的已核实文本则不会截断）；稳定条目 ID 则继续跟踪同一条目的不同版本。
{% endhint %}

## Agent 步骤

操作归属于 **agent 步骤**：agent 自己泳道上已完成回合的 1-based 序号。泳道来自 `memory_role_bind`；没有记录到的记忆绑定时，泳道取自操作事件自身的泳道，减去 session 与 delivery 事件打上的泳道，并给出警告。记忆泳道上的回合——记忆系统自身的提取与检索——不是 agent 步骤。

在**任一**会话中承载主流量的泳道不是记忆泳道：它会连同警告从记忆泳道集合中剔除，其回合计入 agent 步骤。

侧泳道（既不是 main 也不是 memory——例如未绑定的 query-rewrite 泳道）既不计 agent 步骤，也不计记忆回合。只有在存在 main 侧证据时才推导并列入 `validation`；没有该证据时，所有泳道都按 agent 流量处理。

归属规则：

- 第一次 agent 回合完成之前观察到的提取记为 step `0`。
- 后续提取和重写标记使用最近一次完成分类的 agent 回合序号，流水线响应不按回合顺序完成时也遵守此规则。
- 无法归属到任何 agent 回合的投递，其 `step = None`，并并入所在位置内容匹配的区间；所有投递都无法归属的区间没有可确证的下游足迹，按规则判为 neutral（见 [injudge](injudge.md#注入区间)）。

投递指定的消费调用若缺少对应回合或该回合尚未分类，就保持无法归属，不回退到更早或声明的步骤。

## 输入路径

每个命令行路径可以是：

- 含 `trajectory.jsonl` 的轨迹目录；
- `trajectory.jsonl` 文件本身；或
- 递归扫描 `trajectory.jsonl` 的根目录。

同一目录的两种写法只加载一次。匹配不到任何内容的路径会明确报用法错误，绝不静默跳过。

## Episode 准备

准备阶段保留解析流中的全局 agent step 序号，在每个主泳道
`turn_start` 重放泳道消息、工具结果消息槽、历史前缀和撤回。一次消息出现的
身份由解析后的事件位置确定，proof 只对绑定 consumer 输入中可见的出现生效，
因此被后续重写复用的槽位无法替代 consumer 当时看到的内容。绑定的 delivery
跟随其 consumer call，即使注释在 episode 关闭后才写入；未绑定 delivery
保留可观察的 episode 范围，但不会伪造 consumer step。

Placement proof 只在绑定 consumer 的输入中解析，不会匹配后续复用同一消息槽
的内容。Placement 可以只占据消息的一部分：已证实的 placement 会从其宿主出现
中投影出来，与槽位名无关（录制器的 `input_prefix` 与 `system_suffix` 槽位也
包括在内）；准备阶段保留其余任务或系统文本，仅将记忆部分关联到 delivery。
`canonical_message` 和 `request_json_pointer` proof 均可离线复核。请求指针需要
绑定调用的 `calls/<call_id>/request.json`、匹配的 JSON 值摘要，以及文本、角色和
已录制工具调用标识相符的唯一可见宿主消息。支持消息/input 内容、工具输出，以及
Anthropic 的 `/system` 和 `/system/N/text`（使用与已录制系统消息块列表相同的
`message_text` 投影，以换行分隔）；不会把请求
数组下标当作录制器消息槽位。
请求缺失或损坏、宿主有歧义、符号链接越出录制目录时，proof 仍保持未解析。
不安全的 placement 归属和 delivery consumer 的上下文缺口会阻止付费评审，并使
dry-run 返回 2；其他视图中的缺口仍是复核提示。有效的 `history_rewritten(kept=k)`
在保留槽位 `0..k-1` 全部已知时清除该 lane 后续视图的活动缺口，包括 `kept=1`
保留已知系统提示的重置，但旧视图与审计诊断保持不变。无效重写或保留槽位缺失
不会被视为已恢复。Episode reset 与已证实的 wrapper 移除分别保留来源事件。

录制器允许的 `not_in_request` 和 `adapter_defined` 声明并不能证明 placement 如何
进入被捕获的智能体输入。它们以及缺少绑定 proof 的情况仍不支持基于输入证据的评审，
不会默认当作已解析或无害。离线准备仍保留审计记录供复核。对新 adapter 或 API 格式
进行付费评审前，先运行 `--prepare-only`；录制格式有效不等于评估器支持。

## 损坏输入

{% hint style="warning" %}
畸形 JSONL 行、损坏的 seq、未知的绑定状态、非列表的载荷字段，以及断裂的
session / operation 链接，都记入 `validation` 与警告，运行继续。
{% endhint %}

无法使用的命令行附件则会快速失败，抛出针对性的 `SystemExit`：无法读取或
schema 不匹配的 `--verdicts` 报告，或空的 `--golden` 文件。

事件流保持文件顺序。缺失或非整数的 seq 会得到一个比当前已见最大值大 1 的合成 seq，使后续的撤回与顺序配对仍附着在目标上。真正非递增的 seq 只记说明，绝不重排。
