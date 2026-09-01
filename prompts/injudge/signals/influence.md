4. influence (@influence@) — the memory's
   footprint in the agent's SUBSEQUENT behavior: the steps from the span's
   first step onward, bounded where THIS memory is next re-delivered (a
   re-delivery supersedes — later steps are the re-delivery span's window,
   so one memory's influence is never credited to two spans). A memory never
   re-delivered is judged to the end of the stream; another memory's
   re-delivery bounds only its own window, never this one's:
   - referenced: the agent mentioned or quoted it, but its actions did not
     change because of it;
   - acted_on: a later command, edit, or test was shaped by the memory
     (the action would likely have differed without it). A SKIPPED step
     also qualifies, but only with an observable anchor: the agent had
     announced the intent before the injection and then dropped it, or it
     explicitly credited the memory with making the step unnecessary;
   - decisive: the agent explicitly credited the memory with changing its
     course, or the course change is unmistakable from the step stream;
   - none: no observable footprint at all. A skip merely inferred from an
     action's absence is none — absence alone is never evidence.
   A mention is not proof of influence: the agent may have known or found
   the fact on its own (check novelty and the pre-injection steps).
   Echo rule: a direction the agent had already announced, reasoned out, or
   applied at pre-span steps cannot be credited to a memory delivered
   afterward — even when the resulting action lands inside the footprint
   window. Re-delivering the agent's own earlier reasoning or the task
   statement back to it is not influence; that is reversed causality.
