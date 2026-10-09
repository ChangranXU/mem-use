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
# v19 removes the aggregate headline block and the unused span set-diff
# fields (new/removed/previous/changed keys). Verdict counts, rates, costs
# by verdict, and judge labels are unchanged. Summary keeps schema v17.
# v18 dropped the episode reset/wrapper-rewrite heuristics from boundary
# provenance and the per-trajectory cost caveat, and rated novelty and
# accuracy over relevant entries (rubric.relevant_counts). Judge labels are
# unchanged from v17, which separated memory misdirection from execution
# slips. v16 harm judgments are not directly comparable.
REPORT_SCHEMA = "injudge.report/v19"

# v17 rates novelty and accuracy over relevant entries (relevant_counts);
# v16 rated them over all entries. Per-memory evidence stays in the report.
SUMMARY_SCHEMA = "injudge.summary/v17"

# v3 drops the duplicate delivery proof list, raw placement counts, step
# sources, and the reset/wrapper-rewrite heuristics.
EPISODE_INPUT_SCHEMA = "injudge.episode-inputs/v3"
