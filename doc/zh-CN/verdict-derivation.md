---
description: "injudge 如何在代码中从 LLM 评分信号推导使用标签、区间判定与汇总比率。"
icon: code-branch
---

# 判定推导

LLM 为每条投递记忆打五个信号，不选择区间判定。injudge 在代码中推导使用标签、足迹原因、区间判定，以及依赖它们的全部汇总指标（`injudge/judge.py`、`injudge/aggregate.py`）。

本页是映射说明。信号定义见 [injudge](injudge.md#评分信号)。

```mermaid
flowchart TD
    llm["LLM：每条投递记忆五个信号"]
    pair["配对校正：positive_effect 仅与 acted-on 的 influence 同时出现"]
    usage["_derive_usage → used / misused / noted / ignored / unclear"]
    reason["_footprint_reason，仅 ignored 记忆"]
    verdict["derive_verdict：任一 misused → negative，否则任一 used → positive，否则 neutral"]
    agg["汇总：比率、评分投影、成本、memory_usage"]
    llm --> pair --> usage
    usage -->|ignored| reason
    usage --> verdict --> agg
```

## 分工

| 由 LLM 产出 | 由代码推导 |
|---|---|
| 逐记忆的 `relevance`、`accuracy`、`novelty`、`influence`、`positive_effect`（value + note） | 逐记忆的 `usage`（`used` / `misused` / `noted` / `ignored` / `unclear`） |
| 区间级 `analysis`、`evidence[]`、`confidence` | 被忽略记忆的 `reason`（`irrelevant` / `redundant` / `confirming_ignored` / `ignored_novel`） |
| | 区间 `verdict`（`positive` / `negative` / `neutral`） |
| | 投递成本、策略指标，以及全部 `summary.json` 比率 |

`accuracy` 会上报（`inaccurate_keys`），不进入判定路径。`relevance` 与 `novelty` 进入判定路径的唯一方式，是作为被忽略记忆未留足迹的原因。

## 组装

`_assemble_verdict` 把形状合法的评审对象变成内部记录。JSON schema 表达不了的语义检查在这里完成：

1. **每个 key 一条。** 同一次投递中同一 key 的多个版本折叠成一行评分（记警告）。没有 key 的记忆不进评分与使用记录；其投递成本仍计入。
2. **覆盖。** 评审数组中缺失的区间 key 会得到全空信号并记警告。区间内带 key 的记忆一条都未被覆盖时，立即记 `error` 判定——这是模型质量信号，不是推导出的 neutral。评审数组中的未知 key 会被丢弃。
3. **配对。** `positive_effect` 仅在 `influence` 为 `acted_on` 或 `decisive` 时适用。该情况下缺少 effect 会使 usage 为 `unclear`（记警告）。influence 为 `none` 或 `referenced` 时仍给出的 effect 会被强制改为 `null`（记警告）。
4. **使用、原因、判定。** `_derive_usage` 为每条记忆打标签；`_footprint_reason` 只对 `ignored` 运行；`derive_verdict` 读取使用标签。置信度被夹到 `[0, 1]`。

仅供展示的字段（`analysis`、证据 claim）会被复制并打码。畸形的证据列表会连同警告被丢弃，不使区间失败。

## 使用标签

`_derive_usage` 在配对校正之后读取 `influence` 与 `positive_effect`：

| `influence` | `positive_effect` | `usage` | 含义 |
|---|---|---|---|
| `acted_on` 或 `decisive` | `improved` | `used` | 记忆塑造了后续行为，并改善了工作。 |
| `acted_on` 或 `decisive` | `no_effect` 或 `harmed` | `misused` | 记忆塑造了后续行为但没有改善工作，或把 agent 带偏。 |
| `acted_on` 或 `decisive` | `null` | `unclear` | 已 acted-on 但缺少 effect 判断。 |
| `referenced` | （忽略；通常为 `null`） | `noted` | 仅被提及。行动并未因此改变。 |
| `none` | （忽略；通常为 `null`） | `ignored` | 没有可观察足迹。 |
| `null` / 未知 | 任意 | `unclear` | influence 本身未知。 |

使用条目上的 evidence 字符串，取 `positive_effect`、`influence`、`novelty` 中第一条非空 note。

## 足迹原因

`_footprint_reason` 只对 `ignored` 记忆运行。严格解析保证此时 `novelty` 是枚举值（覆盖缺口的全空条目会推导为 `unclear`，不会变成 `ignored`）。

| `relevance` | `novelty` | `reason` |
|---|---|---|
| `false` | 任意 | `irrelevant` |
| `true` | `redundant` | `redundant` |
| `true` | `confirming` | `confirming_ignored` — 敲定了未决疑点，但从未被归功 |
| `true` | `novel` | `ignored_novel` |

这些原因填入 `rubric.no_footprint_reasons`。它们不改变区间判定：被忽略的记忆已经不在 `used` / `misused` 之列。

## 区间判定

`derive_verdict` 读取已推导的使用标签。损害优先：

1. 任一记忆 `misused` → **negative**。
2. 否则任一记忆 `used` → **positive**。
3. 否则 → **neutral**。

`noted`、`ignored` 与 `unclear` 单独不会产生 negative。因此仅被提及永远不会得到 negative 区间。

```python
usages = [entry["usage"] for entry in memory_usage.values()]
if "misused" in usages:
    return "negative"
if "used" in usages:
    return "positive"
return "neutral"
```

### 示例

每行是一个区间。未列出的记忆表示不存在。

| 记忆 | 推导出的 usage | 区间判定 | 原因 |
|---|---|---|---|
| A：`acted_on` + `improved`；B：`none` | A `used`，B `ignored` | **positive** | 至少一个 used，没有 misused。 |
| A：`acted_on` + `no_effect` | A `misused` | **negative** | 被采纳但没有改善：上下文被占用却无收益。 |
| A：`acted_on` + `harmed` | A `misused` | **negative** | 主动损害。 |
| A：`decisive` + `improved`；B：`acted_on` + `harmed` | A `used`，B `misused` | **negative** | 损害优先于帮助。 |
| A：`referenced` | A `noted` | **neutral** | 提及但无行动。 |
| A：`acted_on` + `null` | A `unclear` | **neutral** | 有影响但无 effect 判断；不计为 used 或 misused。 |
| A：`none`，`relevance=false` | A `ignored`（`irrelevant`） | **neutral** | 无足迹。 |
| 其后没有 agent 回合（`auto`） | 占位 `unclear` | **neutral** | 规则判定；无 LLM 调用。见下。 |

## 规则判定与失败区间

| 情形 | `verdict` | 评审调用 | 汇总中的评分 / 使用 |
|---|---|---|---|
| 没有可归属的 agent 步骤 | `neutral`，`auto: no_subsequent_turn` | 无 | 占位的空评分与 `unclear` 使用。不计入评分比率、使用汇总与置信度。在判定率中计为已评判的 **neutral**。 |
| `--limit` 截断或 `--dry-run` | `unjudged` | 无 | 只计数。不是比率分母。 |
| 覆盖缺口、契约失败或尝试耗尽 | `error` | 已尝试 | 只计数。保留原始文本。不会被改写成 neutral。 |

## 从标签到汇总

`injudge/aggregate.py` 把操作记录滚成按轨迹与整体两套块。`injudge-report.json` 与 `summary.json` 共用这些函数。

**判定率。** 分母是已评判区间：`positive + negative + neutral`。规则判定的 auto-neutral 区间在该分母中。`error` 与 `unjudged` 只计数。重述的重复注入不出现在比率中；它们已经并入所属区间。

$$
\mathrm{rates}[v] = \frac{\mathrm{count}[v]}{\mathrm{positive} + \mathrm{negative} + \mathrm{neutral}}
$$

**评分比率。** 总体是 LLM 已评判区间上的已打分投递（`区间 × key`），不含 auto-neutral。全空的覆盖缺口行会被跳过。为 null 的信号值不进入该比率的分母。

| 比率 | 分子 | 分母 |
|---|---|---|
| `relevance` | `true` | `true` + `false` |
| `accuracy` | `corroborated` + `uncontradicted` | 上述 + `contradicted` |
| `novelty` | `novel` + `confirming` | 上述 + `redundant` |
| `influence` | `referenced` + `acted_on` + `decisive` | 上述 + `none` |
| `positive_effect` | `improved` | `improved` + `no_effect` + `harmed` |

`positive_effect` 只在有 effect 判断的投递上计算。这是契约：除非 influence 为 acted-on，该信号为 null。

**排查清单。** `inaccurate_keys` 收集 `accuracy` 为 `contradicted` 的身份；`irrelevant_keys` 收集 `relevance` 为 `false` 的身份。每条带上产生它的区间 id。同一次操作中一个 key 对应多个 `(namespace, item)` 身份时，每个身份都会入账。

**逐记忆使用。** 只在 LLM 已评判区间上按 `(namespace, item)` 合并。对多 item 投递的联合判断会在每个身份下计数。当首尾两个*已录制* value 都存在且不同时，`value_drift` 为 true；redact（未录制）的末端不是 drift。

**成本。** 每个区间的 `approx_tokens_total`（`content_chars_total // 4`，含重述投递）计入其判定类别。headline 的 `wasted_token_share` 为 `(negative + neutral 的 token) / 已评判类别的 token`。error 与 unjudged 的 token 计入 `cost`，但不进该分母。

**置信度。** 均值与分桶（`[0, 0.5)`、`[0.5, 0.8)`、`[0.8, 1.0]`）只覆盖 LLM 已评判区间。置信度从不加权计数。置信度 `< 0.5` 的区间列入 `low_confidence`。

**输入路径**（仅 `overall`）统计带 `input` 记录的预算门区间：`input_paths.full` / `scan`、`gold_scored`、`over_budget`。auto-neutral 与 unjudged 区间没有 `input`，因此不出现在这些计数中。

写出产物的字段级说明见 [产物](artifacts.md)。
