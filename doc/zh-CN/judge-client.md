---
description: "injudge LLM 调用的合规层级、重试、失败处理与凭证规则。"
icon: plug
---

# 评审客户端

{% hint style="info" %}
客户端经 urllib 与配置的端点通信，不使用 SDK。以下行为同时适用于评分调用与提取调用；提取运行在自己的配置副本上（见[评分输入](scoring-input.md#装不下)）。
{% endhint %}

端点设置见[配置](configuration.md)。

## 合规层级

客户端按合规层级链行走：**structured**（受限解码）→ **json**（JSON 模式）→ **prompt**。anthropic 没有 json 层级。

- 格式字段被拒绝或被证明忽略时，降级免费：层级发现不消耗尝试次数。
- 首个可用层级会记忆到本次运行结束。
- 长度/上下文窗口拒绝（413，或带签名的 400/422 响应体）绝不走层级链：评分调用在分支入口即以"过长"失败，该区间回退到分块扫描路径一次，并标记 `length_fallback`（见[评分输入](scoring-input.md#保障)）。
- 用 `OUTPUT_FORMAT` 可固定层级。固定后没有链条，400/422 会使该区间失败——记录在案，运行继续。

区间相关的检查位于层级链之上：评分契约在**尝试循环内**验证精确 id 覆盖、动作锚点窗口，以及评分规则的跨信号依赖（`redundant` 的 novelty 绝不与 acted-on 的 influence 共存；`contradicted` 的 accuracy 绝不与 `improved` 的 effect 共存）。一个形状合法但未通过这些检查的回答在任何层级都属模型错误（约束解码无法知道本区间的 id 与窗口）——走纠正重试，耗尽预算即为可见的 `error`，绝不留下静默不完整的判定。

## 重试与失败处理

瞬时故障——429/5xx、传输错误、空内容——与 `MAX_ATTEMPTS` 共享预算并退避，尊重服务端声明的 `Retry-After`。

- 输出 token 预算在 JSON 中途切断的回答记为指明截断的 `error` 判定：不免费降级，不以同一预算重试。
- 预算耗尽时，该区间为 `error`，并保留评审原始文本。
- 失败不会被改写成 neutral。
- 只要有区间出错，进程退出码为 2。

## 凭证

{% hint style="warning" %}
端点凭证只发往评审端点，绝不出现在产物、报告、日志或错误信息中。回显了请求的端点输出会在任何文本被保留前先打码。
{% endhint %}

重定向被直接拒绝：301/302/303 本会让标准库的重定向处理器把 `Authorization` / `x-api-key` 头转发到任意 `Location`——跨主机、或 HTTPS 降级为 HTTP——因此 3xx 以请求失败的形式记录在该区间上（不走层级链、不重试、无旋钮），凭证的目的地始终是配置的端点。报告 `tool.config` 清单记录已解析的配置以便复现，净化为 `scheme://host[:port][/path]`——绝不含密钥，也绝不含 URL 的 userinfo 或查询串。
