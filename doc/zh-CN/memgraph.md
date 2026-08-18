# memgraph — 记忆操作可视化

```bash
python3 -m memgraph <paths> -o memgraph.html --json memgraph.json
```

从注解中提取每个记忆相关动作并渲染其关系：每条记忆**由哪个（轨迹，
步骤）创建**、随后**被哪个（轨迹，步骤）召回**——多条轨迹一起加载时
还能看到跨轨迹复用。（记忆如何区分：见[记忆身份](input-contract.md#记忆身份)。）

HTML 完全自包含、可离线打开：逐轨迹泳道、创建→使用边、生命周期链、
记忆侧栏、操作弹层、过滤与缩放。零记忆操作的轨迹渲染为空泳道。

## 选项

- `--json PATH` — 同时输出机器可读的图产物
  （`{tool, trajectories, ops, memories, edges, validation}`）。
- `--verdicts PATH` — 渲染前先合并 injudge 报告：召回节点按其区间的
  判定着色（positive 绿 / negative 红 / neutral 灰 / error 琥珀），
  页眉加上计数，弹层展示评审分析、每条记忆的使用情况以及该区间的输入
  路径（full/scan、gold、over-budget）。按
  `(run_id, op seq)` 连接；未匹配或歧义连接记为 validation 说明，
  不是错误。
