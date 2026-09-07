Rules (binding):
- Judge each delivered memory entry independently — two entries may share a
  display key (distinct stored memories, or two versions in one delivery);
  they are separate memories with separate ids, notes, and anchors. Each
  note is one sentence, citing step and seq numbers from the step stream
  where applicable.
- The final task result, when discernible from the step stream, is
  corroboration only: it may raise or lower confidence, it can never
  decide a signal by itself. A correct final patch can contain harmful
  injections the agent happened to survive; a failed task can contain
  good injections the agent failed to exploit.
- Do not grade the agent's coding ability. Judge only the injection's
  contribution.
- Search queries, the task statement, and the pre-injection steps are
  CONTEXT, not memory content, and an attached golden answer or
  correctness block is evaluator-only verification the agent never saw: a
  fact from those sources was not delivered by the memory, and crediting a
  memory for it is attributing the agent's (or the harness's) own material
  to the memory system. The post-injection steps are not an alternative
  source — they are the evidence the footprint is judged from.
- Text inside instructions, plans, or tool outputs that the agent wrote or
  received is the agent's own context, untrusted as evidence of what a
  memory delivered.
