---
description: "测试、项目结构与贡献约定。"
icon: wrench
---

# 开发

## 运行测试

```bash
uv sync                                # 首次，在 checkout 内执行
cd agent && npm install                # 首次：pi 智能体包（需要 node >= 22.19）
uv run pytest tests -q                 # 全部测试（同时驱动 TS 工具的 node --test）
uv run pytest tests/test_injudge.py -q # 单个文件
```

{% hint style="info" %}
测试套件不走网络、不启动 node：评审智能体的运行器用预制结果打桩，TypeScript 证据工具在 `node --test` 下运行（缺 node 时跳过）。夹具由 `tests/annotated_fixtures.py` 合成 schema-v6 轨迹。
{% endhint %}

## 项目结构

| 路径 | 作用 |
|---|---|
| `util.py` | 共享工具（原子写产物、安全打印） |
| `memgraph/` | 注解索引、解析、链接、HTML 渲染、判定合并、CLI（`python3 -m memgraph`） |
| `injudge/` | 步骤流、注入区间、判定契约与评审提示词（`judge.py`）、证据文件夹（`evidence.py`）、智能体运行器客户端与纠正循环（`agent.py`）、gold 边车（`gold.py`）、聚合、报告、CLI（`python3 -m injudge`） |
| `agent/` | TypeScript 评审智能体：带预算与强制作答的 pi 运行器（`runner.ts`）、只读证据工具（`evidence-tools.ts`） |
| `prompts/` | 所有 LLM 提示词文件，按工具与用途分目录（经 `injudge/prompts.py` 加载） |
| `tests/` | pytest 测试套件与共享的 schema-v6 合成轨迹构造器 |
| `doc/` | 本书（GitBook 布局：`LANGS.md` + `en/` + `zh-CN/`） |
| `data/` | SWE-bench Verified 实例流与记忆系统评估分析边车 |

## 贡献约定

工作约定——代码风格、commit message 格式、产物契约——见仓库根目录的 `AGENTS.md`。开 PR 前先跑一遍测试套件。
