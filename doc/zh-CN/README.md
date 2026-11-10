# MemUse 文档

[English](../en/README.md) | [仓库首页](../../README.zh-CN.md)

MemUse 用录制证据评估记忆对 agent 后续行为的影响。命令为 `memgraph` 和 `injudge`。GitHub 仓库名为 `mem-use`，示例中的本地目录名为 `mem-eval`。

## 开始使用

先配置 [memory-bridge](https://github.com/ChangranXU/memory-bridge) 和 [traj-recorder](https://github.com/ChangranXU/traj-recorder)，生成包含显式 `memory_*` 注解的 schema-v10 轨迹。外部项目的安装与录制步骤以各自文档为准。

1. 按[仓库首页](../../README.zh-CN.md#快速开始)安装本地依赖。
2. 用 [memgraph](memgraph.md) 画图，检查记忆操作；用 `injudge --prepare-only` 检查 episode 和消费输入。
3. 按[配置](configuration.md)填写评审端点，用 [injudge](injudge.md) 评审。
4. 从 `summary.md` 查看结果，再按区间打开报告、证据和叠加图。

## 阅读结果

评审按 episode 内的召回区间逐条打五个信号，由代码推导使用标签和区间结论。六个条目占比保留相关性、证实程度、信息增益、行动影响、收益和伤害；未知项和未评估部分单列。结果衡量可观察的记忆影响，不是任务成功率，也不是系统净收益。

- [评分规则与结论推导](verdict-derivation.md)：五个信号、证据要求、使用标签和区间结论。
- [产物](artifacts.md)：报告、摘要、条目占比、覆盖率和成本口径。

## 使用与技术参考

| 页面 | 内容 |
|---|---|
| [输入约定](input-contract.md) | 轨迹路径、注解、记忆身份、episode、步骤归属与损坏输入。 |
| [memgraph](memgraph.md) | 图模型、页面操作和报告叠加。 |
| [injudge](injudge.md) | 区间、运行输出、退出状态和可选参考答案。 |
| [配置](configuration.md) | `.env`、命令行参数和离线准备。 |
| [评审证据](scoring-input.md) | 输入可见性、动作窗口、证据文件和只读工具。 |
| [评审智能体](judge-client.md) | 预算、跟进、对话记录、失败和凭证保护。 |
| [开发](development.md) | 测试、真实评审校准和系统比较。 |

贡献约定见 [AGENTS.md](../../AGENTS.md)，实现契约索引见[贡献者参考](../contributor-reference.md)。
