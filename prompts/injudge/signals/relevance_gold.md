1. relevance (true|false) — the GOLDEN ANSWER below is a reference for
   what the task actually required, not the only valid solution or the only
   permitted subgoal: is the memory topically related to what the task
   needed (repairing the environment, verifying work, or avoiding a
   demonstrated dead end can matter even when the gold patch does not
   mention them), and to the agent's current subgoal at the injection
   point? Judge against the gold text and the steps before the span's
   first step. relevance=false says the memory was off-topic — it never by
   itself decides influence or positive_effect (an off-topic memory can
   still be acted on, and a topical one can be ignored).
