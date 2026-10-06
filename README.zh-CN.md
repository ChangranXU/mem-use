# mem-eval

[English](README.md) | 简体中文

从录制的 agent 轨迹评估记忆系统质量。两个只读工具消费 schema-v10 gate 轨迹
（extension/traj-recorder）：

- **memgraph** — 把每次记忆操作及其关系渲染成自包含、可离线打开的 HTML 图。
- **injudge** — 用智能体式 LLM 评分规则判定每次召回注入对后续行为的影响，
  产出可叠加回图上的报告。

> 注入由它对 agent 的**后续影响**评判——绝不看最终任务结果。

## 环境

- Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 22.19（injudge 智能体运行与 TypeScript 测试需要）

Python 依赖声明在 `pyproject.toml`（memgraph 只用标准库，injudge 另需
`tqdm`）。评审是 `agent/` 下的 TypeScript 智能体，构建于
[pi](https://github.com/earendil-works/pi)，用 npm 安装。

## 快速开始

```bash
uv sync                        # 首次——创建 .venv
npm --prefix agent install     # 首次——pi 智能体包（需要 node >= 22.19）

# 1. 可视化记忆操作——离线，不调用评审
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json

# 2. 评审召回注入——先配置评审端点
cp .env.example .env
uv run -m injudge <paths> --dry-run --limit 3   # 预览证据；不发请求、不写文件
uv run -m injudge <paths>                       # 单独运行目录；已评审区间另有 evidence/ 与 agent-runs/

# 3. 把判定结果叠加到图上（或给 injudge 加 --html）
uv run -m memgraph <paths> --verdicts output/<run-stamp>/injudge-report.json -o memgraph-verdicts.html
```

每个 `<path>` 可以是轨迹目录（含 `trajectory.jsonl`）、`trajectory.jsonl`
文件本身，或递归扫描的根目录。

## 工作原理

injudge 在每个 episode 内把处理方式相同的重复注入归并为**区间（span）**，
再为每个送入评审的区间建立独立的证据文件夹——目标区间及其所属 episode 的
任务、上下文、投递与原始步骤——由一个 pi 智能体用只读工具探索。智能体对
每条投递记忆打五个信号；再由代码推导区间判定：任一记忆使用造成伤害 →
**negative**，否则任一被采纳 → **positive**，否则任一归因或效果未定 →
**unresolved**，否则 **neutral**。仅选择后续读取与无收益使用单独记录，不等于伤害。没有可归属
agent 步骤的区间不送评审，计为 **unassessable**。

被评判的证据不超出录制 placement 实际曝光的内容，acted-on 的影响力断言
必须携带落在该记忆足迹窗口内的动作步骤锚点。每条评审对话只受两个预算
约束——最大步数与上下文窗口——且都以强制作答收尾，而非中途失败。

完整流程见[中文文档](doc/zh-CN/README.md)；信号、使用标签与汇总公式见
[判定推导](doc/zh-CN/verdict-derivation.md)。

## 输入约定

工具只读录制器写下的显式 `memory_*` 注解事件——不解析提示词、不认标签
约定。没有注解的轨迹是合法输入：零记忆操作，外加一条 validation 说明，
不是错误。详见[输入约定](doc/zh-CN/input-contract.md)。

## 文档

完整文档在 [`doc/`](doc/)（English + 简体中文，GitBook 布局），从
[中文文档](doc/zh-CN/README.md)或 [English book](doc/en/README.md) 进入：

- [输入约定](doc/zh-CN/input-contract.md)——轨迹需要包含什么
- [memgraph 指南](doc/zh-CN/memgraph.md) · [injudge 指南](doc/zh-CN/injudge.md)
- [判定推导](doc/zh-CN/verdict-derivation.md)——使用标签、区间判定与比率
- [配置参考](doc/zh-CN/configuration.md)——`.env` 键与 CLI 参数
- [产物](doc/zh-CN/artifacts.md)——report 与 `summary.json` 格式
- [开发](doc/zh-CN/development.md)——目录结构、测试、风格规则

## 测试

```bash
uv run pytest tests -q
```

测试不走网络：Python 测试用预制结果替换付费评审调用。TypeScript 运行器与
证据工具经 `node --test` 由 `tests/test_agent_tools.py` 驱动，缺 node 或
尚未安装 agent 包时跳过。
