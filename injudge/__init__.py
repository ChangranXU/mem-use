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
# v7: the rubric/memory_usage keys are span-local memory ids (m1, m2, …)
# instead of display keys; every delivered entry is judged independently;
# rubric entries carry action anchors; memories carry their id and a
# per-entry memory_exposure record; new_keys/removed_keys gained
# previous_keys/changed_keys; the tool block carries the resolved,
# secret-free run config.
REPORT_SCHEMA = "injudge.report/v7"

# The summary artifact's schema string (report.build_summary writes it).
# v4 follows the v7 report: id-keyed span usage labels, headline
# numerator/denominator fields, and the judge identity in the tool block.
SUMMARY_SCHEMA = "injudge.summary/v4"

