---
description: "将记忆操作可视化为自包含的 HTML 图，并可按 injudge 判定着色。"
icon: share-nodes
---

# memgraph

```bash
python3 -m memgraph <paths> -o memgraph.html --json memgraph.json
```

memgraph 从注解中提取每个记忆操作，渲染记忆如何被创建、更新、删除与召回。对每条记忆回答两个问题：由哪个 `(轨迹, 步骤)` 创建，随后被哪个 `(轨迹, 步骤)` 召回。多条运行一起加载时可以看到跨轨迹复用。

记忆如何区分，见[记忆身份](input-contract.md#记忆身份)。

HTML 是单个自包含文件，可离线打开。零记忆操作的轨迹渲染为空泳道——同样是合法输入。

## 图模型

JSON 产物——嵌在 HTML 里的是同一个 dict——的形状是：

```text
{tool, trajectories, ops, memories, edges, validation}
```

| 集合 | 内容 |
|---|---|
| `ops` | 记忆操作。`kind` 为 `extract`（生成）或 `recall`（投递）；每个操作锚定在 `(traj, step, seq)`。 |
| `memories` | 每个 `(namespace, item_id)` 身份一条记录：来源、创建位置、后续更新、召回、删除。 |
| `edges` | 从操作指向记忆的关系：`creates`、`supersedes`、`deletes`、`recalls`。 |

两条时序说明——都属于合法输入而非错误：

- 已加载集合中没有观察到创建的召回记忆，以 `origin: "not_observed"` 注册并画幽灵创建标记。store 条目只是早于已加载轨迹。
- 时间戳（或在同一轨迹内按 seq）早于首次观察到的创建的召回，会标记 `recalled_before_creation`。边仍会画出；警告记入 `validation`。

## HTML

页面提供：

- 每条轨迹一条泳道（有 `run.json` 的 `started_at` 时按其排序）；
- 创建 → 召回边，以及生命周期链（更新与删除）；
- 记忆侧栏与逐操作弹层；
- 过滤与缩放。

## 选项

| 参数 | 作用 |
|---|---|
| `-o PATH` | HTML 输出路径（默认 `memory-graph.html`） |
| `--json PATH` | 同时写出机器可读的图产物 |
| `--verdicts PATH` | 渲染前合并一份 `injudge.report/v5` 文件 |

## 判定叠加

`--verdicts` 按 `(run_id, op seq)` 把报告连接到召回操作。轨迹编号（`T1`、`T2`……）按每次调用分配，两个工具在不同路径集合上运行时编号可能不同；`run_id` 才是稳定的连接键。

每条召回操作就地标注：

| 字段 | 含义 |
|---|---|
| `verdict` | 区间首次注入上的区间判定；后续相同投递为 `restated`（unjudged 区间保持 `unjudged`） |
| `span_verdict` | 该区间在每一次注入上的判定——着色依据 |
| `verdict_span` | injudge 区间 id |
| `verdict_detail` | 置信度、分析、证据、逐记忆使用、评分、投递成本、输入路径，以及 `auto` / `error` 标记 |

召回节点按 `span_verdict` 着色：positive 绿、negative 红、neutral 灰、error 琥珀。页眉加上判定计数；弹层展示评审分析、每条记忆的使用情况，以及该区间的输入路径（`full` / `scan`、gold、over-budget）。

未匹配或歧义的连接记为 validation 说明，绝不报错。歧义键——多个已加载轨迹共享同一 `run_id`，或同一 `(run_id, seq)` 被多条报告注入或多条召回操作占用——会被跳过，而不是交叉标注。图仍会渲染。
