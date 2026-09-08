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
# injudge.report/v10: the agentic judge. Each judged operation's `input`
# record becomes the agent-run record ({"path": "agent", "attempts",
# "llm_calls", "tool_calls", "transcript", "forced"?, "gold"?} —
# `transcript` points at the conversation's pi event-sourced trajectory
# JSONL under agent-runs/); tool.config becomes the resolved agent
# configuration (max_steps / max_output_tokens / context_window /
# thinking_level, no scan block); aggregates replace input_paths'
# full/scan regimes with the agent regime and over_budget with
# forced_answers. The rubric, entry ids, verdicts and unassessable
# accounting are unchanged.
REPORT_SCHEMA = "injudge.report/v10"

# injudge.summary/v6: input_paths carries the agent regime and forced
# answers replaces over_budget (same rationale as the report bump).
SUMMARY_SCHEMA = "injudge.summary/v6"
