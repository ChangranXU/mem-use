---
description: "How injudge assembles each span's scoring input: verbatim when it fits JUDGE_INPUT_TOKENS, evidence-scanned when it overflows."
icon: align-left
---

# Scoring Input

{% hint style="info" %}
One scoring call per span. The request carries the recorded task statement (when present), the span, and the agent step stream — every completed agent turn with its reasoning, content, and tool results.
{% endhint %}

The span header tells the judge the span's position among the trajectory's spans, the search query that produced the injection, and how the injection was presented when recorded (the placement kind plus a capped excerpt of the delivered block's opening; a redacted placement records no content). Each delivered memory's judged footprint window runs from the span's first step to that memory's next re-delivery — a re-delivery supersedes, so a memory delivered in two spans is never credited in both; the header lists each memory's window separately, and a memory never re-delivered is unbounded.

Routing is decided by `JUDGE_INPUT_TOKENS`, measured with tiktoken `o200k_base` over the scoring system prompt plus the scoring user:

```mermaid
flowchart TD
    size{"Does the verbatim scoring input fit JUDGE_INPUT_TOKENS?"}
    full["Full step stream, verbatim"]
    split["Split the stream at the injection step into pre / post"]
    extract["Each chunk extracts evidence only, concurrently up to SCAN_PARALLEL"]
    merge["Code assembles evidence blocks by step range plus a verbatim seam neighborhood"]
    judge["One scoring call per span"]
    derive["derive_verdict in code"]
    size -->|yes| full
    size -->|no| split
    split --> extract
    extract --> merge
    full --> judge
    merge --> judge
    judge --> derive
```

## Fits the Budget

The complete step stream goes in verbatim. Short trajectories are a single full-text scoring call.

## Overflows the Budget

The stream is never truncated to fit. Instead:

1. **Partition.** The stream is split at the span's first injection step into pre- and post-injection sides; every step lands in exactly one chunk, up to the footprint windows — the post side stops at the largest per-memory window end when every delivered memory is re-delivered later (later steps belong to the re-delivery spans, and the chunk header states the per-memory bounds to extraction). Chunk size is bounded by `SCAN_CHUNK_STEPS` and `SCAN_CHUNK_TOKENS`.
2. **Extract.** Each chunk runs an **extraction** call that returns evidence items only — never rubric values, never a verdict. Extraction is seeded with `[anchor]` items: one per step where a delivered memory's key or a distinctive value word appears verbatim. Anchors are capped in count and length, counted toward the chunk budget, and seed extraction rather than replace it.
3. **Assemble.** Code assembles the scoring user from evidence blocks grouped by step range, plus a verbatim **seam neighborhood** of `SCAN_NEIGHBORHOOD` steps on each side of the injection (`0` keeps only the injection step).

### Guardrails

- A full-text send the endpoint rejects as too long (413, or a signed length/context-window 400/422 — the tiktoken estimate can undercount for the real judge tokenizer) falls back to this scan path once instead of erroring the span, marked `length_fallback` on the operation's `input` record and noted in warnings.
- Over-cap evidence and an over-budget assembled user each get one LLM summary pass; a still-over result keeps the original text. An evidence overrun is noted in warnings; an over-budget assembly is sent as-is and marked `over_budget` on the operation's `input` record.
- A chunk with no usable record fails that span with an `error` verdict naming the chunk. There is no silent coverage gap.
- Extraction runs on a copy of the scoring config with the `SCAN_*` overrides, so a tier degradation learned during extraction never leaks onto scoring.

{% hint style="info" %}
Extraction is per span: each span's split and prompt differ, so *N* overflowing spans pay *N* full extraction passes. Golden-answer and correctness attachments (see [injudge](injudge.md#golden-answer-and-correctness)) are added to the scoring call only; they are never sent to extraction.
{% endhint %}

The chunk knobs (`SCAN_*`) are documented in [Configuration](configuration.md#scan-overflow-path). What the LLM returns for this input is defined in [Rubric and verdict derivation](verdict-derivation.md#rubric-signals).
