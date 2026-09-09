"""injudge — verdicts for recall-injection operations in gate trajectories.

Judges each recall injection's observable footprint in the agent's SUBSEQUENT
behavior: the verdict comes from downstream influence, never from the
final task result alone; "no footprint" ⇒ neutral. Injections with
identical delivered content are grouped into spans and judged once per
span: an LLM agent (built on pi, agent/runner.ts) searches the span's
evidence folder and scores every delivered memory on a five-signal rubric,
and the usage labels and span verdict derive from the rubric in code.

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
# injudge.report/v11: novelty can be null when its recorded baseline cannot
# establish a tier. Other signals and the influence/effect verdict rule stay
# unchanged. Ignored relevant entries with null novelty carry the reason
# ignored_unknown_novelty, never ignored_novel.
REPORT_SCHEMA = "injudge.report/v11"

# injudge.summary/v8: results-first layout, grouped counts/verdicts/delivery/
# judge statistics, compact trajectory rows, and an id-keyed details section.
# Source report schema is retained; scoring and null-novelty semantics do not
# change (null counts as unknown and stays out of novelty-rate denominators).
SUMMARY_SCHEMA = "injudge.summary/v8"
