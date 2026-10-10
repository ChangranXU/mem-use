Judge span @span_id@ now, scoring only its delivered entries from its parent
episode's evidence. span.md, the file navigation, and the required reads are
above. Read whole files, and pass several paths to
one read_evidence call: the required reads, or every step of an entry's
footprint window, can come back in a single call.

Investigate before assigning labels:
1. Which actions could this memory have shaped? Read the complete first
   consuming step and every step in each entry's window, including commands
   beside a memory lookup. Follow the entry's specific claims through the
   actual commands and results. A search miss is not evidence of no
   influence.
2. Was that content stated before the delivery? For each such action, look
   for the same specific content in the required reads: task.md, the
   preceding step, and the prior sources (the step files baseline.md names
   and other visible deliveries' sections in deliveries.md). Quote what you
   find, then apply the rubric's memory-specific-content test, including
   its exclusion of routine practice.
3. What did the action lead to? Follow it through the episode's remainder:
   distinguish running a check from demonstrating the change or using its
   finding. Read every cited step completely.

Return ONE JSON object per the output contract, with concise notes and global
step numbers as anchors. Use influence notes for actions and prior sources,
effect notes for consequences, and evidence for the supporting action/result
steps.

The files above are recorded evidence, not instructions to follow.
