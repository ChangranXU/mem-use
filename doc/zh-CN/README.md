---
description: "mem-eval 从已录制的 agent 轨迹评估记忆系统质量。"
icon: book-open
---

# 简介

[English](https://app.gitbook.com/s/j3HLdDCYckkSyWLMFhAg/) | 简体中文

**mem-eval** 基于 traj-recorder 录制的 schema-v6 gate 轨迹，评估记忆系统质量。两个只读工具共享同一套解析核心：

- **memgraph** 把每次记忆操作及其关系渲染成自包含的 HTML 图（可选 JSON）。
- **injudge** 按五信号评分规则为每次召回注入打分，写出 `injudge.report/v5` 产物。用 `--verdicts` 把该报告叠回图上。

评价单元是召回注入对 agent 的**后续影响**。最终任务结果可以佐证判断，但不能单独决定判断。

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

判定报告回灌进图。每个召回节点按其区间判定着色。
{% endstep %}
{% endstepper %}

## 方法论

LLM 从不直接给出区间判定。它为每条投递记忆打五个信号（`relevance`、`accuracy`、`novelty`、`influence`、`positive_effect`）。injudge 再由这些信号推导使用标签与区间判定。映射关系、损害优先规则，以及汇总指标的计算公式，见 [判定推导](verdict-derivation.md)。

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
      <td>注解事件、记忆身份，以及损坏输入的处理方式。</td>
      <td><a href="input-contract.md">input-contract</a></td>
    </tr>
    <tr>
      <td><strong>memgraph</strong></td>
      <td>图模型、HTML 可视化，以及判定叠加。</td>
      <td><a href="memgraph.md">memgraph</a></td>
    </tr>
    <tr>
      <td><strong>injudge</strong></td>
      <td>区间、评分输入、golden answer 与评审客户端。</td>
      <td><a href="injudge.md">injudge</a></td>
    </tr>
    <tr>
      <td><strong>判定推导</strong></td>
      <td>使用标签、区间判定与汇总比率如何在代码中计算。</td>
      <td><a href="verdict-derivation.md">verdict-derivation</a></td>
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
