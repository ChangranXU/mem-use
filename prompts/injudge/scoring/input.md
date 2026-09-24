Your evidence is one folder with the target span and its parent episode,
read through read_evidence and search_evidence. span.md holds the target's
entries ([m1], [m2], ... — key your answer by these ids), placement,
exposure, and each entry's footprint window. task.md holds the task and later
user updates; context.json target_baseline holds the input visible before
the delivery; steps/ holds one file per agent step, named by its global
step number. Other deliveries are competing sources, not targets.
Optional gold.md is a reference solution; outcome.json is corroboration only.

Order exposure by step, not seq: the first consuming step had the memory in
its input, so everything in it is post-exposure; only earlier steps are
pre-delivery. Judge only what the placement showed, never an unseen suffix.
The recording is incomplete: a missing fact is not proof the agent lacked it,
and an earlier delivery is not proof it stayed visible after a context
rewrite. Queries and keys are metadata, not delivered content or agent
knowledge.

@uncertainty@
