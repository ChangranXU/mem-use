Your evidence is one folder with the target span and its parent episode,
read through read_evidence and search_evidence. span.md holds the target's
entries ([m1], [m2], ... — key your answer by these ids), placement,
exposure, and each entry's footprint window. task.md holds the task and
later user updates. baseline.md lists the consumer's visible input in order
and names the file holding each message. messages.md, when present, holds
other input messages with source positions and first visible turn/step.
Memory text is in span.md (the target) or
deliveries.md under its D-ref. steps/ holds one file per agent step, named by
its global step number; earlier steps can show a prior direction even if a
history rewrite later removed them from the input. manifest.json indexes the
files and the required reads. Other deliveries are competing sources, not
targets. Optional gold.md is a reference solution; outcome.json is
corroboration only.

Order exposure by step, not seq: the first consuming step had the memory in
its input, so it and every later step are post-exposure; only earlier steps
are pre-exposure. A search query in the agent's own tool call records what
the agent expressed at that step: an open possibility, not a decision.
Other search queries and hidden annotation keys are metadata, not delivered
content or agent knowledge; an identifier the placement shows is delivered
content. Judge only what the placement showed, never an unseen suffix. Baseline visibility and known reconstruction gaps are stated in baseline.md.
Compare novelty with that recorded input, not knowledge outside it. An earlier
delivery is not proof it stayed visible after a context rewrite.

@uncertainty@
