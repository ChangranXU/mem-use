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
# v15 separates selection from content use, removes the redundant decisive
# tier, allows unknown influence, and distinguishes unproductive from harmful
# use. read_steps certifies complete retained text, including baseline copies.
REPORT_SCHEMA = "injudge.report/v15"

# v13 carries the revised influence/usage populations and unknown influence.
SUMMARY_SCHEMA = "injudge.summary/v13"

EPISODE_INPUT_SCHEMA = "injudge.episode-inputs/v1"
