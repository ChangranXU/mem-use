---
description: "按召回注入对 agent 的后续影响打分，并写出判定报告。"
icon: gavel
---

# injudge

{% hint style="info" %}
injudge 根据召回区间在后续 agent 回合中的足迹，将可评判区间判为 **positive**、**negative**、**unresolved**（归因或效果未定）或 **neutral**。没有可归属 agent 步骤的区间单独报告为 **unassessable**。
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
| `injudge-report.json` | 运行时评审记录（`injudge.report/v21`），付费调用前写入并将待评区间标为 `unjudged`；每个区间完成后保存，包括并行完成的结果 |
| `summary.json` | 最终统计（`injudge.summary/v18`）：计数、比率、投递估算、评审运行总量 |
| `memgraph-verdicts.html` | 仅在 `--html` 时产出：召回节点按判定着色的 memgraph |
| `evidence/` | 每个送入评审智能体的区间一个文件夹（含运行失败产生的 error 判定）：只包含目标 `span.md`，加上其父 episode 已准备的文件与可选附件 |
| `agent-runs/` | 每个送入评审智能体的区间一条轨迹 JSONL：判定如何得出的 pi 事件流记录 |

投递成本与投递策略指标由录制数据按确定性规则计算，不经评审——见[产物](artifacts.md#交付成本与策略)。完整字段说明也在该页。

任一区间以 `error` 结束、或任一 validation 说明的级别为 `error` 时，进程以退出码 `2` 退出；`--dry-run` 仅凭 error 级说明即以 `2` 退出。

区间在评审前被拒绝的情形：其父 episode 的边界或归属不是 `prepared`，或一个带投递且已归属步骤的区间在其父 episode 中没有已录制的任务。此时该区间得到 `error` 判定，且没有智能体运行——没有证据文件夹、没有轨迹记录、也没有 `input` 记录。

可评区间还会因 episode 中未解析的 placement 归属、无法分离的 placement 与宿主文本，或 delivery consumer 上下文的可见性缺口而被拒绝。Dry-run 使用同一检查并返回 2；单纯的信息性诊断不会阻止评审，空区间和 consumer 不可观察的区间仍按规则跳过。`summary.json` 仅在运行完成后写入；中断时的 report 保留已完成结果和显式待评区间。中断会取消尚在队列中的并行任务，已经运行的任务可能继续完成。完成结果立即保存，validation 诊断仍按源区间顺序排列。

## 离线准备

`--prepare-only` 写出离线 episode 包，而不是一次评审运行：`<run-stamp>/episode-inputs.json`（`injudge.episode-inputs/v4` 索引），外加每个物化 episode 一个 `<run-stamp>/episodes/<episode-dir>/` 文件夹。它不加载 `.env`、不需要 Node，也不发起任何评审调用；若边界或归属需要复核，包会先写出，再以非零状态退出。参见[配置](configuration.md#离线准备)。

## 注入区间

相同内容常常被重复注入，因此评审单元是**注入区间（span）**：采用同一处理方式的召回操作最大段——相同条目按相同顺序、相同 placement 呈现、且每条目的曝光相同。

先划分 episode，再归并区间。跨 episode 的相同处理方式绝不合并；每个 episode 各自保留全局步骤编号、自己的任务/上下文，以及自己的 gold/outcome 连接。空投递与无法归属的区间按机械规则计数，不消耗付费调用。

- 步骤间隔不切段；记入 `step_gaps` 并展示给评审。
- 内容或处理方式的变化都会切段——重新排序的包、重新框定的块、或单条目曝光的变化都会开启新区间，即使投递内容完全相同。每个投递条目按各自的足迹窗口评判；窗口边界与逐条目曝光的规格见[评审证据](scoring-input.md)。
- 重复投递归并为 `restated`：计入投递成本与 `n_injections`，但绝不进入判定率。
- 没有可归属 agent 步骤的区间（`step_first` 为 `None`）没有下游足迹。按规则判 **neutral** 并标记 `auto: no_subsequent_turn`，不发起评审调用。汇总将其计为 `unassessable`，不进入任何方向性比率。

区间如何构造（含未摘要内容的指纹回退）见[记忆身份](input-contract.md#记忆身份)。

未观测到消费者的投递单独计为不可评估区间，即使内容与已观测投递相同，也不继承其判定或成本归类。重复处理的归并是记账约定，不证明持续驻留或独立的因果效应；可见性由录制的输入视图确定。

## 区间如何评审

每个区间经过三个阶段，各自单独成页：

1. **物化证据。** 目标区间及其父 episode 已录制的任务、可见上下文、其他投递、原始全局步骤与可选附件，写入同一个由工具访问范围限定的文件夹。→ [评审证据](scoring-input.md)
2. **按评分规则打分。** 一条智能体对话检索证据，为每条投递记忆返回五个信号——`relevance`、`accuracy`、`novelty`、`influence`、`positive_effect`——绝不返回判定。→ [评分规则与判定推导](verdict-derivation.md)
3. **代码推导判定。** 先得出使用标签，再得出区间判定：任一有害使用即为 negative；否则任一采纳即为 positive；否则任一归因或效果未定即为 unresolved；否则为 neutral。→ [评分规则与判定推导](verdict-derivation.md)

预算、强制作答与对话的重试/失败行为见[评审智能体](judge-client.md)。

## Golden Answer 与 Correctness

评分调用可携带两类可选附件，二者在 `--limit` 筛选付费区间**之前**就连接到 episode。`--golden` 和 `--correctness` 要求恰好一个已发现的 episode。按 benchmark 身份带键的元数据可附着到多个 episode；未带键的轨迹边车仅在该轨迹恰好只有一个 episode 时适用。优先级为：命令行值 → 带键连接 → 可用的边车。

| 附件 | 来源 | 作用 |
|---|---|---|
| Golden answer 正文 | `golden.txt`（与可用的 `golden.md` 同时存在时优先）、`--gold-metadata PATH`（verified-metadata.json：按 episode 记录的 benchmark 身份连接 gold `patch`），或 `--golden PATH` | 将该 episode 的区间调用切换到 **gold 评分规则**（`GOLD_SYSTEM_PROMPT`），并提供 `gold.md`。同样五个信号、同样的 schema 与推导；参考文本不是判定。 |
| 已记录的 correctness | `correctness.json`、`--harness-report PATH`（按 episode 的 benchmark 身份连接 SWE-bench 风格结果），或 `--correctness resolved\|unresolved` | 提供 `outcome.json`，仅作佐证；它不能决定信号，也不会选用 gold 规则。 |

`correctness.json` 接受简单的 `{"resolved": true|false, "score": …}` 记录，或 SWE-bench 风格的评测报告（`resolved_ids` / `unresolved_ids`）。harness 中的错误、未完成、缺失与冲突结果都保留为显式状态，不会被猜成 resolved/unresolved；边车绝不覆盖 harness 连接；而仅按 benchmark 标识的报告——它无法识别重复执行或不同实验组——会在评审前失败，不可用的 linkage 文件也同样如此；请改用执行级的结果。

两者都没有时，评分使用朴素规则。gold 体系记录在该操作的 `input` 上（`"gold": true`），并计入 `aggregates.overall.gold_scored`。
