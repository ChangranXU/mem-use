4. influence (@influence@) — the memory's
   footprint in the agent's SUBSEQUENT behavior: the steps from the span's
   first step onward, bounded where THIS memory is next re-delivered (a
   re-delivery supersedes — later steps are the re-delivery span's window,
   so one memory's influence is never credited to two spans). A memory never
   re-delivered is judged to the end of the stream; another memory's
   re-delivery bounds only its own window, never this one's:
   - referenced: the agent mentioned or quoted it, but its actions did not
     change because of it;
   - acted_on: a post-injection command, edit, or test USES content the
     memory carries and the agent could not have derived from what it had
     already seen — content that appears in neither the task statement,
     the pre-injection steps, nor any tool output before that step (a
     path, identifier, flag, ordering, or a claim the agent could not have
     known otherwise) — or the agent explicitly cites the memory when
     choosing the action. A SKIPPED step also qualifies, but only with an
     observable anchor: the agent had announced the intent before the
     injection and then dropped it, or it explicitly credited the memory
     with making the step unnecessary;
   - decisive: the memory changed the direction of the agent's plan — a
     demonstrable course change (a new approach, a reversed decision), not
     merely a step executing the established course. Explicit credit is
     evidence for the change, never a substitute for the change itself;
   - none: no observable footprint at all. A skip merely inferred from an
     action's absence is none — absence alone is never evidence.
   A mention is not proof of influence: the agent may have known or found
   the fact on its own (check novelty and the pre-injection steps).
   Context is not memory content, and a source's time and role decide what
   is an alternative explanation: the search query that produced the
   injection, the task statement, and the PRE-injection steps are what the
   agent already had when the delivery arrived — a fact established THERE
   is never a fact this memory delivered. An attached golden answer is
   evaluator-only verification the agent never saw, and the post-injection
   steps are not an alternative source either — they are exactly where the
   footprint is judged. Credit a memory only for
   content its delivered entry (within its exposure marker) carries.
   Changed action, not continuation: acted_on/decisive require a step that
   DIFFERS from the course already established before the injection. A
   patch already applied, a test already passing, an analysis or plan
   already written at pre-injection steps, or merely continued/executed
   afterward, is continuation — influence none or referenced, whatever the
   memory's topic. Redundancy alone does not decide this: an
   already-established substance can still be acted on when the delivery
   itself changed the action — a reminder that restores a dropped
   direction, or execution detail the established plan lacked — so judge
   the footprint on its own evidence, whatever the novelty call. The `anchors`
   you give must be the post-injection steps
   where that changed action happened, inside this memory's window; steps
   before the span's first step may support a novelty or accuracy judgment,
   never an influence claim.
   Echo rule: a direction the agent had already announced, reasoned out, or
   applied at pre-span steps cannot be credited to a memory delivered
   afterward — even when the resulting action lands inside the footprint
   window. Re-delivering the agent's own earlier reasoning or the task
   statement back to it is not influence; that is reversed causality.
   When another delivered memory (a co-delivered entry, or an earlier
   span's delivery named in the header) carries the same substance, the
   footprint may be genuinely SHARED: credit the entry the agent's
   behavior most directly follows — or the one it names — and say in the
   note that the support is shared. Never drop an observed footprint only
   because a second memory carried the same substance.
