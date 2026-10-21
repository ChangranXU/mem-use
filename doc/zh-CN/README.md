---
description: "mem-eval 从已录制的 agent 轨迹评估记忆系统质量。"
icon: book-open
---

# 简介

{% hint style="info" %}
**mem-eval 是什么？** 一套只读分析工具，度量 agent 记忆系统的实际效果——可视化全部记忆操作，并以 LLM 评分规则为每次召回注入打分。
{% endhint %}

**mem-eval** 以 traj-recorder 录制的 schema-v10 `trajectory.jsonl` 轨迹为输入——其中每个记忆动作都作为显式的 `memory_*` 注解事件记录在案——将其转化为可操作的质量指标。

记忆动作绝不从提示词、角色名或标签约定中推断。没有注解的轨迹是合法输入，运行得到零个记忆操作。

## 工具

两个只读工具共享同一套解析核心：

| 工具 | 用途 |
|---|---|
| **memgraph** | 把每次记忆操作及其关系渲染成自包含的 HTML 图（可选 JSON 产物）。 |
| **injudge** | 按五信号评分规则为 episode 内的召回区间打分，写出 `injudge.report/v26` 产物；用 `--verdicts` 可将其叠加回图上。 |

贯穿两个工具的一条原则：召回注入以其对 agent 的**后续影响**来评判。最终任务结果可以佐证判断，但绝不决定判断。

## 快速开始

```bash
# 安装依赖（首次，在 checkout 内执行）
uv sync
npm --prefix agent install   # TS 评审智能体的依赖包（node ≥ 22.19）

# 1 —— 可视化全部记忆操作
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json

# 2 —— 评审召回注入（先把 .env.example 复制为 .env 并填写）
uv run -m injudge <paths> -o output/

# 3 —— 把判定叠加回图
uv run -m memgraph <paths> -o memgraph-verdicts.html --verdicts output/<run-stamp>/injudge-report.json
```

{% hint style="tip" %}
路径可以是轨迹目录、`trajectory.jsonl` 文件，或递归扫描的根目录——见[输入约定](input-contract.md#输入路径)。`--dry-run` 可预览每个原本会送入评审的区间，不发起调用、不写文件。
{% endhint %}

## 工作流

{% stepper %}
{% step %}
### 可视化

memgraph 展示每条记忆由哪个 `(轨迹, 步骤)` 创建、随后被哪个 `(轨迹, 步骤)` 召回。多条运行一起加载时可以看到跨轨迹复用。
{% endstep %}

{% step %}
### 评审

injudge 把处理方式相同的重复注入归并为**区间（span）**，由 LLM 为每条投递记忆打分。没有可归属 agent 步骤的区间不经 LLM，按规则判定并计为 **unassessable**。区间判定再由代码推导。损害优先：任一有害使用即为 negative；否则任一采纳即为 positive；否则任一归因或效果未定即为 unresolved；否则为 neutral。
{% endstep %}

{% step %}
### 叠加

判定报告回灌进图，每个召回节点按其区间的判定着色。
{% endstep %}
{% endstepper %}

## 方法论

LLM 从不直接给出区间判定。它为每条投递记忆打五个信号——`relevance`、`accuracy`、`novelty`、`influence`、`positive_effect`——injudge 再在代码中由这些信号推导使用标签与区间判定。

信号定义、损害优先规则与映射关系见[评分规则与判定推导](verdict-derivation.md)；标签如何计入统计见[产物](artifacts.md#数字如何推导)。

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
      <td><strong>评审证据</strong></td>
      <td>每个 LLM 评审区间一个证据文件夹，通过只读工具提供给智能体。</td>
      <td><a href="scoring-input.md">scoring-input</a></td>
    </tr>
    <tr>
      <td><strong>评分规则与判定推导</strong></td>
      <td>LLM 所打的五个信号；代码推导的使用标签与判定。</td>
      <td><a href="verdict-derivation.md">verdict-derivation</a></td>
    </tr>
    <tr>
      <td><strong>评审智能体</strong></td>
      <td>预算与强制作答、纠正重试、失败处理与凭证。</td>
      <td><a href="judge-client.md">judge-client</a></td>
    </tr>
    <tr>
      <td><strong>配置</strong></td>
      <td>全部 <code>.env</code> 键与命令行参数。</td>
      <td><a href="configuration.md">configuration</a></td>
    </tr>
    <tr>
      <td><strong>产物</strong></td>
      <td>report 与 <code>summary.json</code> 字段说明、计数规则及可读摘要 <code>summary.md</code>。</td>
      <td><a href="artifacts.md">artifacts</a></td>
    </tr>
    <tr>
      <td><strong>开发</strong></td>
      <td>测试、目录结构与贡献约定。</td>
      <td><a href="development.md">development</a></td>
    </tr>
  </tbody>
</table>
