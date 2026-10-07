5. positive_effect (@positive_effect@) — only when influence is acted_on;
   otherwise null. Judge observable consequences of the anchored
   actions, never what might otherwise have happened.
   - improved: a later event ties a benefit to the action (later steps use
     what it revealed, a failing test passes, an error the agent was heading
     into is avoided). Keeping or submitting a change is not itself a
     benefit;
   - no_effect: affirmative evidence the action made no difference and added
     no work. This is not harm;
   - harmed: it led the agent astray (wrong location, false lead, lost work
     or state), or it added work that contributed nothing to the outcome or
     the process: a command, edit, or run that the memory caused and that
     was discarded, undone, or corrected, even quickly. This includes the
     memory-caused part of an otherwise useful change that a later step had
     to correct. Reading or retrieving memory content is delivery cost, not
     added work.
   Across the actions this entry shaped inside its window, any observed harm
   makes the effect harmed, even if another action helped; otherwise any
   observed benefit makes it improved. Describe both when they coexist.
   Every value needs affirmative evidence: a warned failure that never
   happened is not a benefit, and saved effort needs announced intent or
   explicit credit. Accuracy does not decide this signal. When the direction
   cannot be established, use null.
   A diagnostic experiment that rules out a hypothesis is improved only when
   later work demonstrably uses that finding (for example, it eliminates a
   live alternative and directs the next check). Removing a temporary probe
   is not itself harm. An abandoned attempt that supplies no used finding is
   harmed; calling a failed attempt "informative" without that later use
   does not establish benefit. If its contribution cannot be determined,
   use null. Any separate harmful action still makes the entry harmed.
   A passing check proves
   only what it checked, not untested cases; local improvement can coexist
   with task failure. Mere selection gets no effect score.
