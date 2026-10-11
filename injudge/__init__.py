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

# Writers and readers share these schema names. Update their tests together.
# See doc/en/artifacts.md for the current contract and comparison limits.
REPORT_SCHEMA = "injudge.report/v22"
SUMMARY_SCHEMA = "injudge.summary/v18"
EPISODE_INPUT_SCHEMA = "injudge.episode-inputs/v4"
CHECKPOINT_SCHEMA = "injudge.checkpoint/v1"
