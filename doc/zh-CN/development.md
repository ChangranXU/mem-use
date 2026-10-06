---
description: "测试、项目结构与贡献约定。"
icon: wrench
---

# 开发

## 运行测试

```bash
uv sync                                # 首次，在 checkout 内执行
npm --prefix agent install             # 首次：pi 智能体包（需要 node >= 22.19）
uv run pytest tests -q                 # 全部测试（同时运行 TS 测试）
uv run pytest tests/test_injudge.py -q # 单个文件
```

{% hint style="info" %}
测试套件不走网络。Python 测试用预制结果替换付费评审调用。TypeScript 运行器与证据工具在 `node --test` 下运行（缺 node 或尚未安装 agent 包时跳过）。夹具由 `tests/annotated_fixtures.py` 合成 schema-v9 轨迹。
{% endhint %}

## 项目结构

| 路径 | 作用 |
|---|---|
| `pyproject.toml` | uv 环境：`package = false`（工具以 `uv run -m …` 运行）、`tqdm` 运行时依赖，以及 `pytest` 开发依赖组 |
| `util.py` | 共享工具（原子写产物、安全打印） |
| `memgraph/` | 注解索引、解析、链接、HTML 渲染、判定合并、CLI（`python3 -m memgraph`） |
| `injudge/` | 步骤流、注入区间、判定契约与评审提示词（`judge.py`）、证据文件夹（`evidence.py`）、智能体运行器客户端与纠正循环（`agent.py`）、gold 边车（`gold.py`）、聚合、报告、CLI（`python3 -m injudge`） |
| `agent/` | TypeScript 评审智能体：带预算与强制作答的 pi 运行器（`runner.ts`）、只读证据工具（`evidence-tools.ts`）、事件流记录（`trajectory.ts`）、作业/结果协议（`protocol.ts`） |
| `prompts/` | 所有 LLM 提示词文件，按工具与用途分目录（经 `injudge/prompts.py` 加载） |
| `calibration/` | 受控的评审校准用例与报告打分（`python3 -m calibration`） |
| `tests/` | pytest 测试套件与共享的 schema-v9 合成轨迹构造器 |
| `doc/` | 本书（GitBook 布局：`LANGS.md` + `en/` + `zh-CN/`） |

## 评审校准

预制答案测试的是契约，而非判断质量。`calibration/` 包含受控的开发回归用例，覆盖显式
与未引用的使用、通用做法、来源与新颖性的区别、未决疑点、计划执行与逆转、有害建议、
无成效的行动、读取候选条目后采用或搁置其未被记录为投递的全文、已记录的预览/完整读取
配对，以及仅措辞不同的基线。每个用例只固定其构造能够确立的标签。

```bash
uv run -m calibration build calibration-cases/
uv run -m injudge calibration-cases/ -o output/calibration   # 付费；重复运行以测稳定性
uv run -m calibration score output/calibration/*/injudge-report.json
```

`score` 报告每处分歧、重复运行的逐标签稳定性，以及匹配用例的新颖性一致性。相同设置
衡量可重复性，改变设置衡量敏感性。除标签外还须审阅来源断言：neutral 判定仍可能错误
描述证据。`AGENTS.md` 要求付费校准取得用户授权。

所有现有用例都属于回归集。留出集必须重新编写，并在首次付费运行前冻结用例、预期标签
和评审提示词。一旦其结果被用于调优，它就成为回归数据；留出验证需要另一套新用例。

已记录后续读取的校准案例包含预览和单独绑定的完整读取，分别评分 case 与
case/read；缺失或失败的目标保留为不可评估。控制案例覆盖有用读取、未使用
读取，以及完整读取前预览已塑造行动的情形。信用归属只是确立影响后的记账
约定，不能覆盖更早行动或仅凭文字一致制造影响。

## 贡献约定

工作约定——代码风格、commit message 格式、产物契约——见仓库根目录的 `AGENTS.md`。开 PR 前先跑一遍测试套件。
