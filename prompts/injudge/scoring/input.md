Your input is the exact TASK STATEMENT the agent worked on (present only
when the run recorded one), the agent step stream of the trajectory — every
completed agent turn with its assistant reasoning/content and tool
executions — plus the span to evaluate, its position among the trajectory's
spans, the search query that produced the injection, and how the injection
was presented to the agent when the recording carries those. Each delivered
memory in the span header carries a stable id ([m1], [m2], ...) — key your
answer's entries by those ids — and, where the recording shows it, an
exposure marker stating what of its value the agent actually saw: judge the
exposure, never an unseen suffix.
The step stream comes either COMPLETE and verbatim, or — when it overflowed
the input budget — as a clearly labeled evidence pack assembled from a
complete chunked scan of the whole stream, with the verbatim steps around
the injection point attached.
Steps before the span's first step are the pre-injection context (what the
agent had done and planned when the injection arrived); steps from the
span's first step onward are where the span's footprint, if any, appears.
Each delivered memory's footprint window — stated in the span header and
defined under signal 4 (influence) — bounds where that memory's influence
may be claimed.

The recorded step stream is not a complete reconstruction of every system
or user message the agent saw. Missing initial context, earlier memory
contents, or extracted evidence is not proof that the agent lacked a fact.
When the record cannot establish a signal, use null with a specific note,
not a forced novelty or influence label. Search queries and item keys are
metadata, not proof of main-agent knowledge or delivered content.
