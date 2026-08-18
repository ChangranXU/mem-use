# mem-eval

[English](README.md) | 简体中文

从录制的 agent 轨迹评估记忆系统质量。两个只读工具消费 schema-v6 gate 轨迹
（extension/traj-recorder）：

- **memgraph** — 把每次记忆操作及其关系渲染成自包含、可离线打开的 HTML 图。
- **injudge** — 用 LLM 评分规则判定每次召回注入，产出可叠加回图上的报告。

> 注入由它对 agent 的**后续影响**评判——绝不看最终任务结果。

## 快速开始

需要 Python 3.11+。memgraph 只用标准库；injudge 另需 `tiktoken` 与 `tqdm`。

```bash
pip install tiktoken tqdm

# 1. 可视化记忆操作——离线，不调用评审
python3 -m memgraph <paths> -o memgraph.html --json memgraph.json

# 2. 评审召回注入——会产生 LLM 调用，先配置评审端点
cp .env.example .env                       # 填写评审端点凭据
python3 -m injudge <paths> --dry-run --limit 3   # 先预览评审输入
python3 -m injudge <paths>                 # 输出 output/injudge-report.json 与 summary.json

# 3. 把判定结果叠加到图上（或用 injudge --html 一并产出）
python3 -m memgraph <paths> --verdicts output/injudge-report.json -o memgraph-verdicts.html
```

每个 `<path>` 可以是轨迹目录（含 `trajectory.jsonl`）、`trajectory.jsonl`
文件本身，或递归扫描的根目录。

## 工作原理

injudge 把内容相同的重复注入归并为**区间（span）**，让评审模型对每条投递
记忆打五个信号（relevance、accuracy、novelty、influence、positive
effect），再由代码推导区间判定：任一记忆被误用 ⇒ **negative**，否则任一
被采纳 ⇒ **positive**，否则 **neutral**。映射、示例与汇总公式见
[判定推导](doc/zh-CN/verdict-derivation.md)。长轨迹走分块证据扫描；输入不
为凑预算而截断。

## 输入约定

工具只读录制器写下的显式 `memory_*` 注解事件——不解析提示词、不认标签
约定。没有注解的轨迹是合法输入：零记忆操作，外加一条 validation 说明，
不是错误。详见[输入约定](doc/zh-CN/input-contract.md)。

## 文档

完整文档在 [`doc/`](doc/)（English + 简体中文，GitBook 布局），从
[中文文档](doc/zh-CN/README.md)或 [English book](doc/en/README.md) 进入：

- [输入约定](doc/zh-CN/input-contract.md)——轨迹需要包含什么
- [memgraph 指南](doc/zh-CN/memgraph.md) · [injudge 指南](doc/zh-CN/injudge.md)
- [判定推导](doc/zh-CN/verdict-derivation.md)——使用标签、区间判定与比率如何计算
- [配置参考](doc/zh-CN/configuration.md)——全部 `.env` 键与 CLI 参数
- [产物](doc/zh-CN/artifacts.md)——report 与 `summary.json` 格式
- [开发](doc/zh-CN/development.md)——目录结构、测试、风格规则

## 测试

```bash
uv venv .venv && uv pip install pytest tiktoken tqdm   # 首次，在 checkout 内执行
.venv/bin/python -m pytest tests -q
```

测试不走网络——评审与提取客户端都打桩。
