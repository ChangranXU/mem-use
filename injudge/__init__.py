"""injudge — verdicts for recall-injection operations in gate trajectories.

Judges each recall injection's causal footprint in the agent's SUBSEQUENT
behavior (eval/TARGET.md: the verdict comes from downstream influence, never
from the final task result alone; "no footprint" ⇒ neutral). Injections with
identical delivered content are grouped into spans and judged once per span
by a hybrid of deterministic footprint signals and an LLM judge.

Reuses eval.memgraph for trajectory discovery, op extraction, and step
attribution; this package adds the step stream, span grouping, judge input
rendering, judging, aggregation, and reporting on top.
"""

from eval.injudge.model import AgentStep, InjectionSpan, ToolExec

__all__ = [
    "AgentStep",
    "InjectionSpan",
    "ToolExec",
]

__version__ = "0.1.0"
