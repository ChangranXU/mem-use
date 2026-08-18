---
description: "轨迹必须包含什么、记忆身份如何定义，以及损坏输入如何处理。"
icon: file-lines
---

# 输入约定

两个工具都读取 **schema-v6 gate 轨迹**。录制器把记忆系统的逻辑动作写成显式的 `memory_*` 注解事件。工具只消费这些事件：不解析提示词、不匹配角色名、不从标签约定推断操作。

没有注解的轨迹是合法输入。运行得到零记忆操作，外加一条 validation 说明。

## 注解事件

| 事件族 | 作用 |
|---|---|
| `memory_session` / `memory_role_bind` | 会话元数据与泳道绑定 |
| `memory_generate_*` | 记忆创建 |
| `memory_change` | 生命周期更新与删除 |
| `memory_search_*` | 检索 |
| `memory_delivery` | 向 agent 上下文注入召回内容 |

## 记忆身份

身份取录制器可移植的 MemoryRef 字段。适配器词汇从不是归组键。

| 层级 | 键 | 用途 |
|---|---|---|
| 条目 | `(namespace, item_id)` | 图注册表、区间集合差、injudge 使用汇总。在同一条目的各版本间保持稳定。没有 `item_id` 时回退到显示键。 |
| 版本 | `namespace` / `identity_scheme` / `version_id` / 内容 sha256 | 区间指纹。与 gate 自己的 `memory_identity` 比较所用元组相同。 |

不同 store 铸造的同名显示键，或同一 store 中不同 item 持有的同名键，是两条记忆。

适配器扩展（`extensions.cure`）只作*展示*投影（key、`memory_type`、confidence、评审状态、拒绝计数、声明的提取步骤）。可移植字段是回退，因此非 CURE 适配器也能解析。

{% hint style="info" %}
若区间内的记忆没有可移植的版本身份（`operation_local` 的 version id 配上未摘要的内容），指纹改用投递的 `(namespace, key, value)`。被 redact 的内容不记录 value，因此同一 `(namespace, key)` 下不同的 redact 正文会共用一个区间：录制结果无法区分它们。
{% endhint %}

## Agent 步骤

两个工具都将操作归属到 **agent 步骤**：已完成的非记忆泳道回合的 1-based 序号。泳道来自 `memory_role_bind`。在任一会话中属于主流量的泳道会从记忆集合中剔除，其回合计入 agent 步骤。

第一次 agent 回合完成之前观察到的提取记为 step 0。无法归属到任何 agent 回合的投递，其 `step = None`。

## 输入路径

每个命令行路径可以是：

- 含 `trajectory.jsonl` 的轨迹目录
- `trajectory.jsonl` 文件本身
- 递归扫描 `trajectory.jsonl` 的根目录

同一目录的两种写法只加载一次。匹配不到任何内容的路径会明确报用法错误。

## 损坏输入

畸形 JSONL 行、损坏的 seq、未知的绑定状态、非列表的载荷字段，以及断裂的 session / operation 链接，都写入 `validation` / 警告。运行继续。手工编辑过的产物降级处理，不抛异常。

事件流保持文件顺序。缺失的 seq 会得到一个比当前已见最大值大 1 的合成 seq，使后续配对仍附着在目标上。非递增的 seq 只记说明，不重排。
