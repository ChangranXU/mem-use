---
description: "按召回注入对 agent 的后续影响打分，并写出判定报告。"
icon: gavel
---

# injudge

{% hint style="info" %}
injudge 根据召回区间在后续 agent 回合中的足迹，将可评判区间判为 **positive**、**negative** 或 **neutral**。没有可归属 agent 步骤的区间单独报告为 **unassessable**。
{% endhint %}

```bash
uv run -m injudge <paths>    # 写出 output/<run-stamp>/injudge-report.json 与 summary.json

# 评审的 TS 半边（一个 pi 智能体）需要先安装一次依赖：
npm --prefix agent install

# 预览模式：打印每个原本会送入智能体的区间的证据文档、文件索引与开场消息，
# 全程不访问网络、不启动 node、不写文件：
uv run -m injudge <paths> --dry-run --limit 3
```

进度经 tqdm 输出到 stderr：终端上是轨迹条加当前轨迹的区间条；管道输出时每完成一条轨迹打一行检查点。

## 产物

每次运行都在 `-o DIR`（默认 `output/`）下写入自己的 `<run-stamp>/` 文件夹（`YYYYMMDD-HHMMSS-<hex>`），连续或并发运行互不覆盖、互不混淆：

| 文件 | 内容 |
|---|---|
| `injudge-report.json` | 运行时评审记录（`injudge.report/v13`），每条轨迹后重写，崩溃也保留已完成部分 |
| `summary.json` | 最终统计（`injudge.summary/v11`）：计数、比率、逐记忆使用 |
| `memgraph-verdicts.html` | 仅在 `--html` 时产出：召回节点按判定着色的 memgraph |
| `evidence/` | 每个送入评审智能体的区间一个文件夹（含 error 结局）：向智能体提供的文件（区间文档、manifest、逐步骤文件、gold） |
| `agent-runs/` | 每个送入评审智能体的区间一条轨迹 JSONL：判定如何得出的 pi 事件流记录 |

投递成本与投递策略指标由录制数据按确定性规则计算，不经评审——见[产物](artifacts.md#交付成本与策略)。完整字段说明也在该页。

任一区间以 `error` 结束时，进程以退出码 `2` 退出。

## 注入区间

相同内容常常被重复注入，因此评审单元是**注入区间（span）**：采用同一处理方式的召回操作最大段——相同条目按相同顺序、相同 placement 呈现、且每条目的曝光相同。

先划分 episode，再在各 episode 内归并区间；跨 episode 的相同投递不会合并。步骤保留录制时的全局编号，窗口至迟在所属 episode 的最后一个可用步骤结束。

- 步骤间隔不切段；记入 `step_gaps` 并展示给评审。
- 内容或处理方式的变化都会切段——重新排序的包、重新框定的块、或单条目曝光的变化都会开启新区间，即使投递内容完全相同。每个投递条目的足迹评判窗口各自从区间首步到该条目下一次被重投递处（重投递即取代——同一条记忆在两个区间投递，不会被重复归功）；从未被重投递的条目窗口直到步骤流末尾，中间内容无关的区间不构成上界，同区间其他条目的窗口结束也不收窄本条目的窗口。窗口按投递条目（rubric 键用的区间本地 id）计算：共用一个显示键的多个身份各自保有自己的边界，每个条目——无论有无键——都参与评分。
- 重复投递归并为 `restated`：计入投递成本与 `n_injections`，但绝不进入判定率。
- 没有可归属 agent 步骤的区间（`step_first` 为 `None`）没有下游足迹。按规则判 **neutral** 并标记 `auto: no_subsequent_turn`，不发起评审调用。汇总将其计为 `unassessable`，不进入任何方向性比率。

"每区间判定一次"的归并预设了较早注入仍驻留在 agent 的上下文中。当录制显示相反的情形——至少一半的投递在下一次投递之前被一次丢弃上下文的历史重写跟随（尾随最后一次投递的丢弃也算：它在运行结束前驱逐了该块），即只有最新的记忆块驻留、重复投递才是可见性机制——且重述投递占主导时，报告会在 validation 备注中声明这一驻留情形，让读者知道 restated 指标假设的是哪种 harness 语义。

区间如何构造（含未摘要内容的指纹回退）见[记忆身份](input-contract.md#记忆身份)。

## 区间如何评审

每个区间经过三个阶段，各自单独成页：

1. **物化证据。** 已录制的任务陈述（若有）、区间文档、gold 附件（若有），以及每个已录制 agent 步骤一个文件，写入一个由智能体用其两个只读工具读取的文件夹。→ [评审证据](scoring-input.md)
2. **按评分规则打分。** 一条智能体对话检索证据，为每条投递记忆返回五个信号——`relevance`、`accuracy`、`novelty`、`influence`、`positive_effect`——绝不返回判定。→ [评分规则与判定推导](verdict-derivation.md)
3. **代码推导判定。** 先得出使用标签，再得出区间判定：任一误用即为 negative；否则任一采纳即为 positive；否则为 neutral。→ [评分规则与判定推导](verdict-derivation.md)

预算、强制作答与对话的重试/失败行为见[评审智能体](judge-client.md)。

## Golden Answer 与 Correctness

评分调用可携带两类可选附件。在 `--limit` 生效前按 episode 连接；`--golden` 和 `--correctness` 只允许恰好一个已发现的 episode。按 benchmark 身份连接的元数据可用于多个 episode；未带执行身份的轨迹边车文件仅在该轨迹只有一个 episode 时适用。优先级为：命令行值 → 元数据连接 → 可用的边车文件。

| 附件 | 来源 | 作用 |
|---|---|---|
| Golden answer 正文 | `golden.txt`（与可用的 `golden.md` 同时存在时优先）、`--gold-metadata PATH`（verified-metadata.json：按 episode 记录的 benchmark 身份连接 gold `patch`），或 `--golden PATH` | 将该 episode 的区间调用切换到 **gold 评分规则**（`GOLD_SYSTEM_PROMPT`），并提供 `gold.md`；gold 本身不是判定。 |
| 已记录的 correctness | `correctness.json`、`--harness-report PATH`（按 episode 的 benchmark 身份连接 SWE-bench 风格结果），或 `--correctness resolved\|unresolved` | 通过 `outcome.json` 提供佐证；不能单独决定信号，也不会选用 gold 规则。 |

`correctness.json` 接受简单的 `{"resolved": true|false, "score": …}` 记录，或 SWE-bench 风格的评测报告（`resolved_ids` / `unresolved_ids`）。harness 中的错误、未完成、缺失与冲突结果保留为不同状态，边车文件不能覆盖 harness 连接。仅按 benchmark 标识的 harness 报告无法区分重复执行或不同实验组；这种连接会在付费调用前拒绝。不可用的 linkage 文件也会提前失败。

两者都没有时，评分使用朴素规则。gold 体系记录在该操作的 `input` 上（`"gold": true`），并计入 `aggregates.overall.gold_scored`。
