# mem-eval

[English](README.md) | 简体中文

从录制的 agent 轨迹评估记忆系统质量。两个只读工具消费 schema-v6 gate 轨迹
（extension/traj-recorder）：

- **memgraph** — 把每次记忆操作及其关系渲染成自包含、可离线打开的 HTML 图。
- **injudge** — 用智能体式 LLM 评分规则判定每次召回注入，产出可叠加回图上的报告。

> 注入由它对 agent 的**后续影响**评判——绝不看最终任务结果。

## 快速开始

需要 Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)，评审智能体另需
node ≥ 22.19。Python 依赖声明在 `pyproject.toml`（memgraph 只用标准库，
injudge 另需 `tqdm`）；评审本身是 `agent/` 下的 TypeScript 智能体，构建于
[pi](https://github.com/earendil-works/pi) 包，用 npm 安装。

```bash
uv sync                     # 首次，在 checkout 内执行——创建 .venv
cd agent && npm install     # 首次——pi 智能体包（需要 node >= 22.19）

# 1. 可视化记忆操作——离线，不调用评审
uv run -m memgraph <paths> -o memgraph.html --json memgraph.json

# 2. 评审召回注入——会运行 LLM 智能体，先配置评审端点
cp .env.example .env                       # 填写评审端点凭据
uv run -m injudge <paths> --dry-run --limit 3   # 先预览每个区间的证据文件夹
uv run -m injudge <paths>                  # 输出 output/injudge-report.json 与 summary.json

# 3. 把判定结果叠加到图上（或用 injudge --html 一并产出）
uv run -m memgraph <paths> --verdicts output/injudge-report.json -o memgraph-verdicts.html
```

每个 `<path>` 可以是轨迹目录（含 `trajectory.jsonl`）、`trajectory.jsonl`
文件本身，或递归扫描的根目录。

## 工作原理

injudge 把内容相同的重复注入归并为**区间（span）**，把每个区间的证据
（任务、区间文档、gold 附件、每个已录制 agent 步骤一个文件）物化进一个
文件夹，让一个工具只读且限定在该文件夹内的 pi 智能体对投递的记忆条目
打五个信号（relevance、accuracy、novelty、influence、positive effect，
按稳定的区间本地 id 逐条评分），再由代码推导区间判定：任一记忆被误用 ⇒
**negative**，否则任一被采纳 ⇒ **positive**，否则 **neutral**。被评判的
证据不超出录制 placement 实际曝光的内容（被截断的值只按已曝光前缀评判；
无法确定的曝光标记为未验证），acted-on 的影响力断言必须携带落在该记忆
足迹窗口内的动作步骤锚点。映射、示例与汇总公式见
[判定推导](doc/zh-CN/verdict-derivation.md)。每条对话只受两个简单预算约束
——最大步数与上下文窗口——且都以强制作答收尾而非中途失败；长轨迹只是
意味着智能体按需选读。

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
uv sync               # 首次，在 checkout 内执行
uv run pytest tests -q
```

测试不走网络——评审智能体的运行器打了桩，TypeScript 证据工具经 `node --test` 由 `tests/test_agent_tools.py` 驱动。
