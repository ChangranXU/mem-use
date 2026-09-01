---
description: "按召回注入对 agent 的后续影响打分，并写出判定报告。"
icon: gavel
---

# injudge

{% hint style="info" %}
injudge 根据召回注入在后续 agent 回合中的足迹，将其判为 **positive**、**negative** 或 **neutral**。
{% endhint %}

```bash
uv run -m injudge <paths>    # 写出 output/injudge-report.json 与 output/summary.json

# 预览模式：按区间打印评审输入（含 gold 块），全程不访问网络。
# 只读 .env 中的预算/分块键——不读凭证、不发起调用、不写文件：
uv run -m injudge <paths> --dry-run --limit 3
```

进度经 tqdm 输出到 stderr：终端上是轨迹条加当前轨迹的区间条；管道输出时每完成一条轨迹打一行检查点。

## 产物

每次运行写入 `-o DIR`（默认 `output/`）：

| 文件 | 内容 |
|---|---|
| `injudge-report.json` | 运行时评审记录（`injudge.report/v5`），每条轨迹后重写，崩溃也保留已完成部分 |
| `summary.json` | 最终统计（`injudge.summary/v2`）：计数、比率、逐记忆使用 |
| `memgraph-verdicts.html` | 仅在 `--html` 时产出：召回节点按判定着色的 memgraph |

投递成本与投递策略指标由录制数据按确定性规则计算，不经评审——见[产物](artifacts.md#交付成本与策略)。完整字段说明也在该页。

## 注入区间

相同内容常常被重复注入，因此评审单元是**注入区间（span）**：投递集合具有相同可移植身份的连续召回操作的最大段。

- 步骤间隔不切段；记入 `step_gaps` 并展示给评审。
- 内容变化会切段。每条投递记忆的足迹评判窗口各自从区间首步到该记忆下一次被重投递处（重投递即取代——同一条记忆在两个区间投递，不会被重复归功）；从未被重投递的记忆窗口直到步骤流末尾，中间内容无关的区间不构成上界，同区间其他记忆的窗口结束也不收窄本记忆的窗口。
- 重复投递归并为 `restated`：计入投递成本与 `n_injections`，但绝不进入判定率。
- 没有可归属 agent 步骤的区间（`step_first` 为 `None`）没有下游足迹。按规则判 **neutral** 并标记 `auto: no_subsequent_turn`，不发起评审调用。

"每区间判定一次"的归并预设了较早注入仍驻留在 agent 的上下文中。当录制显示相反的情形——多数投递在下一次投递之前被一次丢弃上下文的历史重写跟随，即只有最新的记忆块驻留、重复投递才是可见性机制——且重述投递占主导时，报告会在 validation 备注中声明这一驻留情形，让读者知道 restated 指标假设的是哪种 harness 语义。

区间如何构造（含未摘要内容的指纹回退）见[记忆身份](input-contract.md#记忆身份)。

## 区间如何评审

每个区间经过三个阶段，各自单独成页：

1. **组装评分输入。** 已录制的任务陈述（若有）、该区间，以及 agent 步骤流。装得下 `JUDGE_INPUT_TOKENS` 的输入逐字进入；超预算的输入走证据扫描而非截断。→ [评分输入](scoring-input.md)
2. **按评分规则打分。** 一次 LLM 调用为每条投递记忆返回五个信号——`relevance`、`accuracy`、`novelty`、`influence`、`positive_effect`——绝不返回判定。→ [评分规则与判定推导](verdict-derivation.md)
3. **代码推导判定。** 先得出使用标签，再得出区间判定：任一误用即为 negative；否则任一采纳即为 positive；否则为 neutral。→ [评分规则与判定推导](verdict-derivation.md)

调用本身的重试、层级与失败行为见[评审客户端](judge-client.md)。

## Golden Answer 与 Correctness

评分调用可携带两类可选附件。各自来自 `trajectory.jsonl` 旁的边车文件或命令行；命令行值覆盖同名边车字段。对多条已加载轨迹共用一个 CLI gold 会给出警告。

| 附件 | 来源 | 作用 |
|---|---|---|
| Golden answer 正文 | `golden.txt`（与可用的 `golden.md` 同时存在时优先），或 `--golden PATH` | 把该次调用切换到 **gold 评分规则**（`GOLD_SYSTEM_PROMPT`）。同样的五个信号、同样的 schema、同样的推导链。`relevance` 与 `positive_effect` 的方向相对 gold 文本评判；gold 不评判 `accuracy`、`novelty`、`influence`。 |
| 已记录的 correctness | `correctness.json`，或 `--correctness resolved\|unresolved` | 以 `TASK CORRECTNESS` 块附加在该次调用所用提示词之下。仅作佐证：不能决定任何信号，也不会选用 gold 规则。 |

`correctness.json` 接受简单的 `{"resolved": true|false, "score": …}` 记录，或 SWE-bench 风格的评测报告（`resolved_ids` / `unresolved_ids`）。评测报告按轨迹记录的 `instance_id` 连接；error / incomplete 实例属于 harness 故障而非结果，不附加 correctness。

两者都没有时，评分使用朴素规则。gold 体系记录在每个经过预算门的操作的 `input` 上（`"gold": true`），并计入 `aggregates.overall.gold_scored`。
