---
description: "将记忆操作可视化为自包含的 HTML 图，并可按 injudge 判定着色。"
icon: share-nodes
---

# memgraph

{% hint style="info" %}
memgraph 从注解中提取每个记忆操作，渲染记忆如何被创建、更新、删除与召回——生成单个自包含的 HTML 文件，可离线打开。
{% endhint %}

```bash
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json
```

对每条记忆回答两个问题：由哪个 `(轨迹, 步骤)` 创建，随后被哪个 `(轨迹, 步骤)` 召回。多条运行一起加载时可以看到跨轨迹复用。

记忆如何区分，见[记忆身份](input-contract.md#记忆身份)。零记忆操作的轨迹渲染为空泳道——同样是合法输入。

## 图模型

JSON 产物——嵌在 HTML 里的是同一个 dict——的形状是：

```text
{tool, trajectories, ops, memories, edges, validation}
```

| 集合 | 内容 |
|---|---|
| `ops` | 记忆操作。`kind` 为 `extract`（生成）或 `recall`（投递）；每个操作锚定在 `(traj, step, seq)`。召回操作带 `searches`（投递的按泳道 `{lane_id, operation_id}` 链接）和 `queries`（每条已链接搜索的解析查询）。候选、投递与删除引用带 `grouping_id`，用于与注册表联结。 |
| `memories` | 每个稳定条目身份一条记录：通常是 `(namespace, item_id)`，否则采用输入约定中的显示键或内容摘要回退。每条记录包含来源、创建位置、后续更新、召回与删除历史——`deletions` 列出每一次观察到的删除（重新创建会标记 `revived`，绝不抹除历史），`deleted` 是终态。 |
| `edges` | 从操作指向记忆的关系：`creates`、`supersedes`、`deletes`、`recalls`。 |

两条时序说明——都属于合法输入而非错误：

- 已加载集合中没有观察到创建的召回记忆，以 `origin: "not_observed"` 注册并画幽灵创建标记。store 条目只是早于已加载轨迹。
- 时间戳（或在同一轨迹内按 seq）早于首次观察到的创建的召回，会标记 `recalled_before_creation`。边仍会画出；警告记入 `validation`。

## HTML 功能

页面提供：

- 每条轨迹一条泳道（横轴为 agent 步骤；有 `run.json` 的 `started_at` 时按其排序）
- 提取（▲）与召回（▼）操作；相邻步骤中投递同一记忆集合的召回折叠为 ×n 显示区间
- 创建 → 召回边，以及虚线生命周期链（更新与删除）
- 来源不在已加载集合中的记忆的幽灵创建标记
- 历史重写标记；连续步骤上的重写会折叠成带（记忆泳道重写使用不同颜色）
- 记忆侧栏、逐操作详情、页内图例、过滤与缩放
- 无法归属步骤的操作停在末步右侧的槽位

## 选项

| 参数 | 作用 |
|---|---|
| `-o PATH` | HTML 输出路径（默认 `memory-graph.html`） |
| `--json PATH` | 同时写出机器可读的图产物 |
| `--verdicts PATH` | 渲染前合并一份 `injudge.report/v13` 文件 |

## 判定叠加

`--verdicts` 按 `(run_id, op seq)` 把报告连接到召回操作。轨迹编号（`T1`、`T2`……）按每次调用分配，两个工具在不同路径集合上运行时编号可能不同；`run_id` 才是稳定的连接键。

每条召回操作就地标注：

| 字段 | 含义 |
|---|---|
| `verdict` | 区间首次注入上的区间判定；后续相同投递为 `restated`（unjudged 区间保持 `unjudged`） |
| `span_verdict` | 该区间在每一次注入上的判定——着色依据 |
| `verdict_span` | injudge 区间 id |
| `verdict_detail` | 置信度、分析、证据、逐记忆使用、评分（两者均以区间本地条目 id 键控，`memory_labels` 把 id 映射回显示键）、投递成本、输入路径，以及 `auto` / `error` 标记 |

召回节点按 `span_verdict` 着色：

| 颜色 | 判定 |
|---|---|
| 绿色 | positive |
| 红色 | negative |
| 灰色 | neutral |
| 琥珀色 | error |

页眉加上判定计数；弹层展示评审分析、每条记忆的使用情况，以及该区间的输入记录（`agent` 路径，含 gold / forced 标记与尝试次数）。

{% hint style="info" %}
未匹配或歧义的连接记为 validation 说明，绝不报错。歧义键——多个已加载轨迹共享同一 `run_id`，或同一 `(run_id, seq)` 被多条报告注入或多条召回操作占用——会被跳过，而不是交叉标注。图仍会渲染。
{% endhint %}
