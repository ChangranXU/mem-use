# 输入约定

工具读取 **schema-v6 gate 轨迹**。录制器把记忆系统的逻辑动作写成显式的
`memory_*` 注解事件：

- `memory_session` / `memory_role_bind` — 会话与泳道绑定
- `memory_generate_*` — 记忆创建
- `memory_change` — 生命周期变更
- `memory_search_*` — 检索
- `memory_delivery` — 向 agent 上下文注入召回内容

工具只消费这些事件：不认标签约定、不解析提示词、不猜测载荷。没有注解的
轨迹得到零记忆操作加一条 validation 说明——这是合法输入，不是错误。

## 记忆身份

身份取录制器可移植的 MemoryRef 字段，绝不依赖适配器词汇：

- **条目级归组**——图注册表、区间集合差、injudge 的使用汇总——取
  (`namespace`, `item_id`) 对，在同一条目的各版本间保持稳定；没有记录
  `item_id` 时回退到显示键。不同 store 铸造的同名键、或同一 store 中不同
  item 持有的同名键，是两条记忆，绝不误并。
- **区间指纹**比较的是 gate 自己的 `memory_identity` 所用的版本级元组：
  `namespace` / `identity_scheme` / `version_id` / 内容 sha256。

适配器扩展（`extensions.cure`）只作*展示*投影读取（key、memory_type、
confidence、评审状态、拒绝计数、声明的提取步骤），可移植字段是回退，因此
非 CURE 适配器也能解析。任何行为都不以某个适配器的词汇为键。

## 损坏输入

损坏输入被容忍，绝不致命：畸形 JSONL 行、损坏的 seq、未知的绑定状态、
非列表的载荷字段、断裂的 session/operation 链接，都落入
`validation`/警告，运行继续。手工编辑过的产物降级处理，绝不抛异常。

## 输入路径

运行时输入可以是轨迹目录（含 `trajectory.jsonl`）、`trajectory.jsonl`
文件本身，或递归扫描的根目录。匹配不到任何内容的路径会明确报用法错误，
绝不悄悄跳过。
