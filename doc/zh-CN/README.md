# 简介

[English](../en/README.md) | 简体中文

**mem-eval** 分析 schema-v6 gate 轨迹（extension/traj-recorder）以评估记忆
系统质量。两个只读工具共享同一套解析核心：

- **memgraph** 把每次记忆操作及其关系渲染成自包含的 HTML/JSON 图。
- **injudge** 用 LLM 评分规则判定每次召回注入的影响，产出
  `injudge.report/v5` 产物，可经 `--verdicts` 回灌给 memgraph。

方法论一句话：注入由它对 agent 的**后续影响**来评判——绝不看最终任务
结果。

## 工作流

1. **可视化** — memgraph 展示每条记忆**由哪个（轨迹，步骤）创建**、随后
   **被哪个（轨迹，步骤）召回**——多条轨迹一起加载时还能看到跨轨迹复用。
2. **评审** — injudge 把内容相同的重复注入归并为区间，对每条投递记忆打五
   个评分信号，再由代码推导区间判定：任一误用 ⇒ negative，否则任一采纳
   ⇒ positive，否则 neutral。
3. **叠加** — 判定报告回灌进图，召回节点按区间判定着色。

## 本书目录

- [输入约定](input-contract.md) — 轨迹必须包含什么，记忆身份如何判定。
- [memgraph](memgraph.md) — 可视化工具：选项与判定合并。
- [injudge](injudge.md) — 注入区间、评分信号、golden answer、评审稳健性。
- [配置](configuration.md) — 全部 `.env` 键与命令行参数。
- [产物](artifacts.md) — report 与 summary 文件，逐字段说明。
- [开发](development.md) — 测试与项目结构。
