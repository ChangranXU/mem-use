"""injudge — verdicts for recall-injection operations in gate trajectories.

Judges each recall injection's observable footprint in the agent's SUBSEQUENT
behavior: the verdict comes from downstream influence, never from the
final task result alone; "no footprint" ⇒ neutral, undetermined attribution or effect ⇒ unresolved. Same-treatment
re-injections — identical ordered entries, placement presentation, and
per-entry exposure — are grouped into spans and judged once per span. An
LLM agent (built on pi, agent/runner.ts) searches the span's evidence folder
and scores every delivered memory on a five-signal rubric; usage labels and
the span verdict derive from that rubric in code.

Reuses memgraph for trajectory discovery, op extraction, and step
attribution; this package adds the step stream, span grouping, evidence
materialization, the agent runner client, aggregation, and reporting on
top.
"""

from __future__ import annotations

# The report artifact's schema string, shared by the writer
# (injudge.report.py) and the reader (memgraph/verdicts.py) so a bump can
# never land on one side only and surface at read time. Tests pin the
# literal deliberately — a bump must update them (see AGENTS.md).
# v21 drops the runner's duplicate retained_evidence count and the tool-free
# repair phase (every rejected answer takes one follow-up path); evidence
# folders replace context.json with manifest required_reads and messages.md
# for input text without a task/step home. Labels match v20.
# v20 changes influence attribution (memory-specific content, null only for
# unread windows or open-possibility ties): influence, effect, and verdict
# counts are not comparable with v19. The artifact shape is unchanged.
# v19 removes the aggregate headline block and the unused span set-diff
# fields (new/removed/previous/changed keys). Verdict counts, rates, costs
# by verdict, and judge labels are unchanged. Summary keeps schema v17.
# v18 dropped the episode reset/wrapper-rewrite heuristics from boundary
# provenance and the per-trajectory cost caveat, and rated novelty and
# accuracy over relevant entries (rubric.relevant_counts). Judge labels are
# unchanged from v17, which separated memory misdirection from execution
# slips. v16 harm judgments are not directly comparable.
REPORT_SCHEMA = "injudge.report/v21"

# v18 renames judge.spans_with_repairs to spans_with_followups. v17 rates
# novelty and accuracy over relevant entries (relevant_counts); v16 rated them
# over all entries. Per-memory evidence stays in the report.
SUMMARY_SCHEMA = "injudge.summary/v18"

# v4 replaces each episode folder's context.json with messages.md for input
# text without a task/step home; full replay stays in the package's audit
# records. v3 dropped the duplicate delivery proof list,
# raw placement counts, step sources, and the reset/wrapper-rewrite heuristics.
EPISODE_INPUT_SCHEMA = "injudge.episode-inputs/v4"
