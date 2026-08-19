---
description: "mem-eval 从已录制的 agent 轨迹评估记忆系统质量。"
icon: book-open
---

# 简介

**mem-eval** 度量 agent 记忆系统的实际效果。它的输入是 traj-recorder 录制的 schema-v6 `trajectory.jsonl` 轨迹——每个记忆动作（创建、检索、投递）都以显式的 `memory_*` 注解事件记录在案；输出是全部记忆操作的可视化图，以及每次召回注入的 LLM 评审判定。

记忆动作绝不从提示词、角色名或标签约定中推断。没有注解的轨迹是合法输入，运行得到零个记忆操作。

两个只读工具共享同一套解析核心：

- **memgraph** 把每次记忆操作及其关系渲染成自包含的 HTML 图（可选 JSON 产物）。
- **injudge** 按五信号评分规则为每次召回注入打分，写出 `injudge.report/v5` 产物；用 `--verdicts` 可将其叠加回图上。

贯穿两个工具的一条原则：召回注入以其对 agent 的**后续影响**来评判。最终任务结果可以佐证判断，但绝不单独决定判断。

## 快速开始

```bash
# 首次，在 checkout 内执行（dev 组为测试套件提供 pytest）
uv sync

# 1 —— 可视化全部记忆操作
uv run -m memgraph runs/ -o memgraph.html --json memgraph.json

# 2 —— 评审召回注入（先把 .env.example 复制为 .env 并填写）
uv run -m injudge runs/ -o output/

# 3 —— 把判定叠加回图
uv run -m memgraph runs/ -o memgraph-verdicts.html --verdicts output/injudge-report.json
```

路径可以是轨迹目录、`trajectory.jsonl` 文件，或递归扫描的根目录——见[输入约定](input-contract.md#输入路径)。

## 工作流

{% stepper %}
{% step %}
## 可视化

memgraph 展示每条记忆由哪个 `(轨迹, 步骤)` 创建、随后被哪个 `(轨迹, 步骤)` 召回。多条运行一起加载时可以看到跨轨迹复用。
{% endstep %}

{% step %}
## 评审

injudge 把内容相同的重复注入归并为**区间（span）**，由 LLM 为每条投递记忆打分，再由代码推导区间判定。损害优先：任一误用即为 negative；否则任一采纳即为 positive；否则为 neutral。
{% endstep %}

{% step %}
## 叠加

判定报告回灌进图，每个召回节点按其区间的判定着色。
{% endstep %}
{% endstepper %}

## 方法论

LLM 从不直接给出区间判定。它为每条投递记忆打五个信号——`relevance`、`accuracy`、`novelty`、`influence`、`positive_effect`——injudge 再在代码中由这些信号推导使用标签与区间判定。信号定义、损害优先规则与映射关系见[评分规则与判定推导](verdict-derivation.md)；标签如何计入统计见[产物](artifacts.md#数字如何推导)。

{% hint style="info" %}
最终补丁正确，仍可能包含 agent 碰巧扛过去的有害注入；任务失败，仍可能包含 agent 未能利用的有用注入。injudge 评的是注入的足迹，不是 agent 的编码能力。
{% endhint %}

## 目录

<table data-view="cards">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th data-hidden data-card-target data-type="content-ref"></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>输入约定</strong></td>
      <td>注解事件、记忆身份、agent 步骤与损坏输入处理。</td>
      <td><a href="input-contract.md">input-contract</a></td>
    </tr>
    <tr>
      <td><strong>memgraph</strong></td>
      <td>图模型、HTML 可视化，以及判定叠加。</td>
      <td><a href="memgraph.md">memgraph</a></td>
    </tr>
    <tr>
      <td><strong>injudge</strong></td>
      <td>区间、产物、三阶段评审流程与 golden answer。</td>
      <td><a href="injudge.md">injudge</a></td>
    </tr>
    <tr>
      <td><strong>评分输入</strong></td>
      <td>预算门：逐字全文，或分块证据扫描。</td>
      <td><a href="scoring-input.md">scoring-input</a></td>
    </tr>
    <tr>
      <td><strong>评分规则与判定推导</strong></td>
      <td>LLM 所打的五个信号；代码推导的使用标签与判定。</td>
      <td><a href="verdict-derivation.md">verdict-derivation</a></td>
    </tr>
    <tr>
      <td><strong>评审客户端</strong></td>
      <td>合规层级、重试、失败处理与凭证。</td>
      <td><a href="judge-client.md">judge-client</a></td>
    </tr>
    <tr>
      <td><strong>配置</strong></td>
      <td>全部 <code>.env</code> 键与命令行参数。</td>
      <td><a href="configuration.md">configuration</a></td>
    </tr>
    <tr>
      <td><strong>产物</strong></td>
      <td>report 与 <code>summary.json</code> 字段说明，含计数规则。</td>
      <td><a href="artifacts.md">artifacts</a></td>
    </tr>
    <tr>
      <td><strong>开发</strong></td>
      <td>测试、目录结构与贡献约定。</td>
      <td><a href="development.md">development</a></td>
    </tr>
  </tbody>
</table>
