"""injudge — verdicts for recall-injection operations in gate trajectories.

Judges each recall injection's observable footprint in the agent's SUBSEQUENT
behavior: the verdict comes from downstream influence, never from the
final task result alone; "no footprint" ⇒ neutral. Same-treatment
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
# injudge.report/v13: empty deliveries have a separate count/cost bucket;
# episode diagnostics and counts remain explicit alongside span verdicts.
REPORT_SCHEMA = "injudge.report/v13"

# injudge.summary/v11: empty-span accounting and compact episode indexes.
SUMMARY_SCHEMA = "injudge.summary/v11"

EPISODE_INPUT_SCHEMA = "injudge.episode-inputs/v1"
