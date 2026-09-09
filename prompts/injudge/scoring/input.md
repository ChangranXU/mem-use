Your evidence is a folder, with its exact span.md and manifest.json already
included in the opening message. Two read-only tools (read_evidence and
search_evidence) are scoped to that folder alone. manifest.json indexes
every file; span.md carries the span
to evaluate: the exact TASK STATEMENT the agent worked on (present only
when the run recorded one), the span's position among the trajectory's
spans, the search query that produced the injection, how the injection was
presented to the agent when the recording carries that, and the delivered
memories. Each delivered memory carries a stable id ([m1], [m2], ...) —
key your answer's entries by those ids — and, where the recording shows
it, an exposure marker stating what of its value the agent actually saw:
judge the exposure, never an unseen suffix. Gold attachments, when the run
carries any, ride gold.md.
Every completed agent turn of the trajectory — its assistant
reasoning/content and tool executions — is one file under steps/ (the step
numbers in the file names are the step numbers the span header cites).
Steps before the span's first step are the pre-injection context (what the
agent had done and planned when the injection arrived); steps from the
span's first step onward are where the span's footprint, if any, appears.
Each delivered memory's footprint window — stated in the span header and
defined under signal 4 (influence) — bounds where that memory's influence
may be claimed.

The recorded step stream is not a complete reconstruction of every system
or user message the agent saw. Missing initial context, earlier memory
contents, or extracted evidence is not proof that the agent lacked a fact.
Search queries and item keys are metadata, not proof of main-agent knowledge
or delivered content.

@uncertainty@
