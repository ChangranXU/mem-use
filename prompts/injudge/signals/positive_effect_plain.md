5. positive_effect (@positive_effect@) — only when influence is acted_on or
   decisive; otherwise null. Judge observable consequences of the anchored
   actions, never what might otherwise have happened.
   - improved: a later event ties a benefit to the action (later steps build
     on it, a failing test passes, an error the agent was heading into is
     avoided);
   - no_effect: no observable downstream change (output unused, edit
     reverted or superseded, result unchanged);
   - harmed: it led the agent astray (wrong location, false lead, lost work
     or state).
   Every value needs affirmative evidence: a warned failure that never
   happened is not a benefit, and saved effort needs announced intent or
   explicit credit. Accuracy does not decide this signal. When the direction
   cannot be established, use null.
