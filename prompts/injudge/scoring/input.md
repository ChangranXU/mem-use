Your evidence is one folder with the target span and its parent episode,
read through read_evidence and search_evidence. span.md holds the target's
entries ([m1], [m2], ... — key your answer by these ids), placement,
exposure, and each entry's footprint window. task.md holds the task and
later user updates. baseline.md lists the consumer's visible input in order
and names the task or step file holding each message; text with no other
home is written there. Memory text is in span.md (the target) or
deliveries.md under its D-ref. steps/ holds one file per agent step, named by
its global step number; earlier steps can show a prior direction even if a
history rewrite later removed them from the input. context.json is the
detailed visibility index. Other deliveries are competing sources, not
targets. Optional gold.md is a reference solution; outcome.json is
corroboration only.

Order exposure by step, not seq: the first consuming step had the memory in
its input, so it and every later step are post-exposure; only earlier steps
are pre-exposure. A search query in the agent's own tool call records what
the agent expressed at that step: an open possibility, not a decision.
Other search queries and hidden annotation keys are metadata, not delivered
content or agent knowledge; an identifier the placement shows is delivered
content. Judge only what the placement showed, never an unseen suffix. The
recording is incomplete: a missing fact is not proof the agent lacked it,
and an earlier delivery is not proof it stayed visible after a context
rewrite.

@uncertainty@
