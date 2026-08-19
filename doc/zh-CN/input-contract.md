---
description: "轨迹必须包含什么、记忆身份如何定义，以及损坏输入如何处理。"
icon: file-lines
---

# 输入约定

两个工具都读取 **schema-v6 gate 轨迹**。录制器把每个记忆系统动作写成显式的 `memory_*` 注解事件；工具只消费这些事件，绝不解析提示词、匹配角色名或从标签约定推断操作。

没有注解的轨迹是合法输入：运行得到零个记忆操作，外加一条 validation 说明，而不是报错。

## 注解事件

| 事件族 | 记录的动作 |
|---|---|
| `memory_session` / `memory_role_bind` | 会话元数据与泳道绑定 |
| `memory_generate_*` | 记忆创建 |
| `memory_change` | 生命周期更新与删除 |
| `memory_search_*` | 检索 |
| `memory_delivery` | 向 agent 上下文注入召回内容 |

## 记忆身份

记忆以录制器可移植的 MemoryRef 字段区分，绝不以适配器词汇归组。

| 层级 | 键 | 用途 |
|---|---|---|
| 条目 | `(namespace, item_id)` | 图注册表、区间集合差、injudge 使用汇总。在同一条目的各版本间保持稳定；没有 `item_id` 时回退到显示键。 |
| 版本 | `namespace` / `identity_scheme` / `version_id` / 内容 sha256 | 区间指纹——与 gate 自己的 `memory_identity` 比较所用元组相同。 |

不同 store 铸造的同名显示键，或同一 store 中不同条目持有的同名键，是两条不同的记忆。

适配器扩展（`extensions.cure`）只贡献**展示**投影：key、`memory_type`、confidence、评审状态、拒绝计数、声明的提取步骤。可移植字段仍是回退，因此非 CURE 适配器也能解析。

{% hint style="info" %}
**区间指纹回退。** 没有可移植版本身份的记忆（`operation_local` 的 version id 配上未摘要的内容）改用投递的 `(namespace, key, value)` 作指纹。被 redact 的内容不记录 value，因此同一 `(namespace, key)` 下不同的 redact 正文会共用一个区间：录制结果无法区分它们。
{% endhint %}

## Agent 步骤

操作归属于 **agent 步骤**：agent 自己泳道上已完成回合的 1-based 序号。泳道来自 `memory_role_bind`；记忆泳道上的回合——记忆系统自身的提取与检索——不是 agent 步骤。

在**任一**会话中承载主流量的泳道不是记忆泳道：它会连同警告从记忆泳道集合中剔除，其回合计入 agent 步骤。

两个边界情形：

- 第一次 agent 回合完成之前观察到的提取记为 step `0`。
- 无法归属到任何 agent 回合的投递，其 `step = None`。这样的区间没有可评判的下游行为，按规则判为 neutral（见 [injudge](injudge.md#注入区间)）。

## 输入路径

每个命令行路径可以是：

- 含 `trajectory.jsonl` 的轨迹目录；
- `trajectory.jsonl` 文件本身；或
- 递归扫描 `trajectory.jsonl` 的根目录。

同一目录的两种写法只加载一次。匹配不到任何内容的路径会明确报用法错误，绝不静默跳过。

## 损坏输入

损坏数据降级处理，绝不让运行中断。畸形 JSONL 行、损坏的 seq、未知的绑定状态、非列表的载荷字段，以及断裂的 session / operation 链接，都记入 `validation` 与警告，运行继续。手工编辑过的产物同样降级，不抛异常。

事件流保持文件顺序。缺失的 seq 会得到一个比当前已见最大值大 1 的合成 seq，使后续的撤回与顺序配对仍附着在目标上。真正非递增的 seq 只记说明，绝不重排。
