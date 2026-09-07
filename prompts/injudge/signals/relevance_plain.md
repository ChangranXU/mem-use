1. relevance (true|false) — is the memory topically related to the task
   (the recorded statement above; when absent, reconstruct the task from
   the pre-injection steps) and to the agent's current subgoal at the
   injection point? Use the steps before the span's first step.
   relevance=false says the memory was off-topic — it never by itself
   decides influence or positive_effect (an off-topic memory can still be
   acted on, and a topical one can be ignored).
