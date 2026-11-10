# Inspect the judge agent

[Guide index](README.md) · [简体中文](../zh-CN/judge-client.md)

The MemUse judge uses the pi agent runtime through `agent/runner.ts`.
Python starts one runner process for each conversation phase. The first phase
investigates one span. A later phase continues the same history after a rejected
answer. Python validates the answer and derives the verdict.

See [configuration](configuration.md) for installation, endpoint settings,
budgets, and parallelism.

## The conversation

Each span gets one conversation and one [evidence folder](scoring-input.md),
limited to its parent episode. The opening contains the target span, file
navigation, and required reads. The full manifest and episode files are
available through two read-only tools: `read_evidence` and `search_evidence`.

The runner receives a job file without the key or endpoint. It returns answer
text, history, retained call counts, executed call counts, reported usage, and
context rollbacks. The operation's `input` record sums executed work across
phases. It keeps retained counts from the last phase. Missing or invalid counts
stay `null`; valid counts are nonnegative integers.

## Required reads

Python checks both answer structure and evidence coverage. A file access alone
does not prove a complete read. The check compares retained numbered tool
output with the evidence files.

Every answer needs the manifest's `required_reads.always` files:

- `task.md` and `baseline.md`;
- `messages.md` and `gold.md`, when present;
- the first consuming step and its preceding episode step, when present.

A non-null novelty or an `acted_on` influence also needs all
`required_reads.prior_sources`. These include baseline step files and complete
sections for other deliveries visible in that input. The target delivery is
already in the opening.

`none`, `referenced`, and `selected` need the entry's complete action window.
A non-null acted-on effect needs every step from the window start through the
episode end. Later consequences can confirm or overturn an apparent effect.
Use `null` when the required coverage is incomplete.

Every action anchor and structured `evidence[].steps` citation must name a
completely read step. A complete step read includes its manifest
`result_sources` sections. `input.read_steps` records the steps that passed
this check.

Paged reads can combine. Header-only reads, search hits, truncated lines,
failed reads, and rolled-back reads do not establish full coverage. Each read
has a 200,000-character cap. A partial read supplies the next `offset` and
`column`, so even a single long line can be read in full. Columns count Unicode
characters from 1. Continued lines use `line@column: text`.

Coverage checks establish what the judge read. They do not prove its judgment
is correct.

## Budgets and forced answers

Before the first paid call, the runner estimates the opening, system prompt,
tool schemas, and `required_reads.always` text. If these exceed the input budget,
the span fails without a call. The estimate excludes blank lines, read-format
text, and optional evidence. Passing does not guarantee the whole investigation
will fit.

Two limits can force a final answer:

| Limit | Behavior |
|---|---|
| Call budget (`MAX_STEPS`) | Reserves a final round. The forced message asks for an answer from evidence already read. Tool calls in that round are blocked. |
| Context budget (`CONTEXT_WINDOW`) | Uses reported input usage plus trailing messages, or estimates message content when usage is unavailable. It reserves pi's context allowance, output space, and the forced message. An overflowing round is rolled back before the forced answer. |

A follow-up gets at most four model calls, including its final answer, or
`MAX_STEPS` if lower. It does not get a second full investigation budget.
The existing context limit can force an earlier answer.

During a follow-up, the runner recalculates output room after each tool round.
It keeps evidence that fits and can reduce the output cap. The cap must still
fit the estimated shortest valid answer. If the opening cannot leave that
room, the phase fails before a paid call. Evidence that exceeds the budget
still rolls back. The estimate does not guarantee a complete answer.

A forced answer can pass after sufficient retained reads. Without a successful
retained step read, valid JSON alone cannot pass. `input.forced` describes the
final answer. `input.any_phase_forced` records whether any phase was forced.

## Follow-ups

Python validates strict JSON, signal values, exact entry-ID coverage, action
windows, and required reads. `selected` and `acted_on` need anchors. `none`
requires an empty anchor list. A valid answer ends the conversation.

Every rejected answer takes the same follow-up path while phases remain.
The runner restores `job.history`, keeps the evidence tools, and receives a
message listing the rejection reasons. This applies to malformed JSON,
invalid values, empty answers, and missing reads. A response cut at the output
limit also takes this path.

`MAX_ATTEMPTS` limits all phases, including the first investigation. When the
limit is reached, the span has an `error` verdict and retains masked answer
text. The parser does not strip formatting to rescue an invalid answer.

## Trajectory recording

Each span's judge conversation is saved to
`agent-runs/<span-dir>.jsonl`. Phases append to the same file. Sequence numbers
are local to each attempt.

| Record type | Contents |
|---|---|
| `attempt` | Starts a runner phase. Attempt 1 is the investigation. |
| `message` | A completed user, assistant, or tool message, including usage. Seeded history does not produce duplicate records. |
| `tombstone` | `removedSeqs` and a reason: `context_budget_rollback` or `forced_tool_tail`. These records identify messages removed from the live conversation. |
| `attempt_end` | Answer, stop reason, forced flag, error, retained counts, executed counts, usage, and rollbacks. |

The runner records forced messages explicitly because a context rewrite does
not emit the normal event. If a forced answer also contains blocked tool calls,
it records the retained answer text after removing the tool tail.

`input.transcript` gives the path relative to the run folder. Executed counts
include discarded rounds. Retained counts describe the final conversation.

## Failure handling

Missing Node, Node older than 22.19, a missing runner, or missing packages cause
a setup error before input preparation. The installed pi library supplies the
context reserve used to check configuration.

A runner that cannot start, exits with an error, or writes no readable result
produces a span `error` and a validation note. Other spans continue. The same
runner failure is not retried. Agent-level errors, such as failed authentication
or persistent transport faults, also produce a span error.

`TIMEOUT` limits idle waits for headers or the next body chunk. An active stream
has no total elapsed deadline. Before streaming starts, pi can retry a transient
request failure up to twice. Each retry gets a new idle wait. A failure after
streaming starts is not retried. A requested retry delay above 60 seconds fails.

## Credentials

The key and endpoint reach the runner only through `JUDGE_API_KEY` and
`JUDGE_BASE_URL`. Neither appears as a configuration value in job or result
files. An endpoint path can contain a key, so the URL needs the same protection.
The child does not inherit `OPENAI_*` or `ANTHROPIC_*` variables that could
change provider headers or the endpoint.

Endpoint output can repeat the key in errors or successful responses. Masking
covers its raw text, percent-encoded forms in either hex case, and literal
JSON-serialized spelling before it reaches verdicts, `judge_raw`, or notes.
Keys shorter than eight characters are treated as absent by masking.

When credentials are loaded, Python masks evidence before writing it.
Openings, hashes, and coverage checks use those masked files. Jobs, reports,
checkpoints, summaries, and HTML also mask the key before saving.
Offline preparation and dry-run do not load judge credentials.

The runner masks transcript and result strings before serialization. This does
not change the live conversation. Python checks the saved transcript after each
phase. It decodes the JSONL envelope and masks string fields, without parsing
JSON-looking text inside those fields. It drops incomplete or malformed records
and keeps a final newline for safe append. If it cannot save the masked
transcript, it removes the file and records a span error with input diagnostics.
