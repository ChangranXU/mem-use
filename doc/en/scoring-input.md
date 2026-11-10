# Inspect judge evidence

[Guide index](README.md) · [简体中文](../zh-CN/scoring-input.md)

MemUse gives each eligible span one evidence folder and one judge conversation.
The folder contains the target span and evidence from its parent episode only.
The judge can read and search it with two tools.

Run `injudge --prepare-only` to inspect episode evidence, or `--dry-run` to
print the span document and opening without a judge call.

## The target span

`span.md` identifies the target, its parent execution, and the delivered entries
in order. Entries use local IDs: `m1`, `m2`, and so on. These IDs are the rubric
keys. Display labels are shown only when the recorded placement exposed them.
Hidden annotation keys stay in audit records.

For each entry, the file shows:

- exposed text and exposure state;
- delivery and consumer source records;
- eligible action steps and window bounds;
- earlier same-item deliveries in the episode;
- whether earlier text remains in the first consuming input;
- proven follow-up retrievals, with separate action and result-delivery steps.

Other deliveries provide competing-source evidence. They are not additional
rubric targets. Memory type labels and confidence are withheld from judge
evidence.

## Exposure limits

The judge can credit only text the agent received.

| State | Evidence available to the judge |
|---|---|
| `full` | The full recorded value. |
| `full_normalized` | The full value after safe display normalization. This can include lossless JSON-string decoding and contiguous numbered tool-result lines. Code whitespace must still match. |
| `clipped` | Only the exposed prefix. |
| `unknown` | The canonical value is withheld because exposure is unverified. |

A preview cannot gain a mechanism from a later full read, a query, or a golden
answer. A search hit locates evidence; it does not certify the step or prove
that the agent acted on it.

## Action windows and repeated copies

An entry's action window ends at the next same-episode delivery that takes over
its credit, or at the episode's last eligible global step. Re-exposure must show
all the earlier entry's verified text. Different or narrower text can leave
windows overlapping. Each entry is judged against its own text and window.

An equal or shorter copy adds no action credit while an earlier copy remains
provably visible in the consuming input. This includes a search preview of a
still-visible full read. The earlier owner keeps its window; the later entry
has no eligible action steps. `memory_window_owners` identifies the owner.
Ownership resets at episode boundaries.

This rule requires continuous visibility. If visibility cannot be proved, a
later delivery uses the ordinary handoff rules. A break in visibility starts
a separate span with its own baseline. Wider or different text, a newly shown
selector, and a proven preview-to-full-read selection keep their own rules.
Ownership is an accounting rule, not proof that the first copy caused an action.

Deliveries consumed in one turn are all eligible at that step. A same-step copy
that adds neither verified text nor a shown selector instead starts after the
shared step. Each entry has its own window; one entry's handoff does not move
another entry's bounds.

Visibility is checked per entry and per placement. A surviving slot cannot keep
text from a retracted slot visible. If only a clipped copy survives, only its
prefix remains visible. Unknown exposure cannot be treated as visible once any
slot from its delivery is gone.

Action anchors must stay within the entry's window and parent episode. Later
results in the same episode can support an earlier eligible action. They cannot
create a new action anchor outside its window.

## The folder

```text
<output>/<run-stamp>/evidence/<span-dir>/
  span.md
  task.md
  baseline.md
  messages.md          # when needed
  deliveries.md
  searches.json
  manifest.json
  gold.md              # optional
  outcome.json         # optional
  steps/
    step_000257.md
    step_000258.md
```

| File | Contents |
|---|---|
| `span.md` | Target entries, exposure, provenance, and windows. Included in the opening. |
| `task.md` | Recorded user task and updates, with source positions. |
| `baseline.md` | First consuming input in message order. Each `[role occurrence]` block points to its text. |
| `messages.md` | Input text with no task or step file, including system updates and replayed history. Records source positions and first visible turn/step. |
| `deliveries.md` | Episode deliveries. The target points to `span.md`; other deliveries have `D<n>` references. |
| `searches.json` | Search query, counts, status, and source references. Opaque extensions remain in the audit records, outside judge evidence. |
| `manifest.json` | Episode identity, file index, required reads, baseline metadata, and result-source references. |
| `steps/` | Full step text, reasoning, tools, results, and result availability. Global step numbers are preserved. |
| `gold.md` | Optional reference answer. Selects the gold rubric. |
| `outcome.json` | Optional recorded result. Supporting evidence only. |

Each text has one home. Proven memory placements are removed from their host
messages and linked to their delivery. The surrounding task or system text
stays in the host. An emptied host still names the memory file; it is not
presented as missing content.

Historical steps remain in `steps/`, but a rewrite does not put them back into
the current input. Full per-turn visibility records remain in the offline
`episode-inputs.json` audit package.

`baseline.md` reports the visible-message count and reconstruction gaps. A
missing slot inside the observed or retained prefix blocks judging before calls.
Only a retracted tail is a known removal. A gap-free reconstruction describes
recorded input; it does not establish every possible source of agent knowledge.

Step files label timing as before exposure, first consuming step, or after
exposure. This is the timing of the action or received result. It is not the
creation date of displayed source code, and it does not prove continued memory
visibility.

## Manifest and required reads

`manifest.json` records `target_baseline`: the consumer, visible occurrence IDs,
IDs that still contain non-memory text, and visibility gaps.

Its `deliveries.md` entry contains section references and inclusive line bounds.
The renderer records these bounds. Headings inside memory text cannot define
them. Coverage checks use the bounds to require complete sections for visible
baseline deliveries or referenced tool results. Target deliveries are already
in the opening and need no duplicate read.

`required_reads.always` lists the task, baseline, optional messages and gold,
and the first consuming step plus its preceding episode step when present.
`required_reads.prior_sources` lists other baseline step files and visible
`deliveries.md#D<n>` sections. The opening repeats these requirements.

The opening contains the target document and navigation, not the full step
bodies. The runner checks estimated required input against its budget before
any paid call. See [required-read validation](judge-client.md#required-reads).

## The tools

| Tool | Behavior |
|---|---|
| `read_evidence` | Reads one or more `paths` with 1-based line numbers. `offset`, `limit`, and `column` select a range. Returns one text block per file, up to 200,000 characters per call. |
| `search_evidence` | Searches complete lines with a JavaScript regular expression. Use `ignore_case`, not inline `(?i)`. Returns path, line, and an excerpt of at most 300 characters around each match. |

A default search covers `steps/` only. To search another file or subfolder,
set `path`. Search returns at most 200 matching lines and reports truncation.
A search that runs longer than five seconds fails. Narrow the pattern or path
when necessary.

Reads report `details.files` with path, returned line range, truncation, or
error. Files beyond the character cap are unread. Partial reads give a
continuation offset and column. Batch step files to reduce model calls;
each model call resends the retained conversation.

Absolute paths, `..`, and symlinks outside the evidence folder are refused.
The judge has no shell or general filesystem/network tool.

Search helps locate identifiers. It does not replace reading an action window.
A keyword miss cannot prove no influence. An execution without a recorded
result proves only an attempt. Missing evidence requires uncertainty under the
[rubric rules](verdict-derivation.md#rubric-signals).
