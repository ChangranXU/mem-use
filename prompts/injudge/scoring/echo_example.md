Example: a memory saying "fetch() needs a backoff delay between retries" is
consumed at step 6, and the agent adds the delay at step 7 without
mentioning it.
- Step 4 wrote "the retry loop never waits; I will add a backoff delay in
  fetch()": novelty redundant (step 4), influence none — the agent's own
  decision explains step 7, and crediting the memory would reverse
  causality. Had it said "the memory confirms my plan", influence would be
  referenced.
- Step 4 only asked "maybe the retries need a delay?": novelty confirming,
  influence null — an open possibility and the memory both precede step 7,
  and nothing separates them.
- Step 4 only printed fetch()'s code, which has no delay: the delay is
  memory-specific content, so influence is acted_on (anchor step 7), even
  though a capable agent could have derived it.
A plan first stated at step 6 or later is post-exposure and shows no prior
direction.
