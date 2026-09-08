"""injudge — verdicts for recall-injection operations in gate trajectories.

Judges each recall injection's causal footprint in the agent's SUBSEQUENT
behavior: the verdict comes from downstream influence, never from the
final task result alone; "no footprint" ⇒ neutral. Injections with
identical delivered content are grouped into spans and judged once per
span: an LLM judge scores every delivered memory on a five-signal rubric,
and the usage labels and span verdict derive from the rubric in code.

Reuses memgraph for trajectory discovery, op extraction, and step
attribution; this package adds the step stream, span grouping, judge input
rendering, judging, aggregation, and reporting on top.
"""

from __future__ import annotations

# The report artifact's schema string, shared by the writer
# (injudge/report.py) and the reader (memgraph/verdicts.py) so a bump can
# never land on one side only and surface at read time. Tests pin the
# literal deliberately — a bump must update them (see AGENTS.md).
# injudge.report/v8: rubric/memory_usage key on span-local entry ids
# (m1, m2, …) stamped on each memories entry; rule-judged auto-neutral
# (no-subsequent-turn) spans tally as `unassessable` in the aggregate
# counts/cost buckets — out of the directional denominators and the
# wasted-token share (observation missingness is not a measured null
# effect) — and the headline names their span/token totals when non-zero.
REPORT_SCHEMA = "injudge.report/v8"

# injudge.summary/v5: the unassessable tally joins the verdict counts,
# and the headline carries unassessable_spans/unassessable_tokens when
# non-zero (the wasted-token caveat rides every reported share).
SUMMARY_SCHEMA = "injudge.summary/v5"

