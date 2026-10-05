"""injudge — verdicts for recall-injection operations in gate trajectories.

Judges each recall injection's observable footprint in the agent's SUBSEQUENT
behavior: the verdict comes from downstream influence, never from the
final task result alone; "no footprint" ⇒ neutral, an acted-on footprint
of undetermined effect ⇒ unresolved. Same-treatment
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
# injudge.report/v14: the `unresolved` span verdict (acted on, effect
# undetermined), the headline renamed to tokens/share without demonstrated
# benefit, rubric.shared_anchors, and per-entry memory_variant_steps /
# memory_followups (action/delivery steps and tool/search provenance) beside
# the re-exposure-bounded footprint windows; shared anchors include trajectory.
REPORT_SCHEMA = "injudge.report/v14"

# injudge.summary/v12: the unresolved verdict class, the renamed benefit
# headline, and the shared-anchor triage list.
SUMMARY_SCHEMA = "injudge.summary/v12"

EPISODE_INPUT_SCHEMA = "injudge.episode-inputs/v1"
