5. positive_effect (@positive_effect@) — only when influence is acted_on or
   decisive; otherwise null. Did the anchored actions move the agent toward
   what the task required (gold.md shows one valid solution) or away from it?
   Judge observable consequences, never what might otherwise have happened.
   - improved: a later event ties a benefit to the action (later steps build
     on it, a failing test passes, an error the agent was heading into is
     avoided); verification, environment repair, or an avoided dead end can
     improve work without appearing in the gold patch;
   - no_effect: no observable downstream change (output unused, edit
     reverted or superseded, result unchanged);
   - harmed: it led the agent astray of what the task required (wrong
     location, false lead, lost work or state).
   Every value needs affirmative evidence: matching the patch does not prove
   the memory caused the action, a warned failure that never happened is not
   a benefit, and saved effort needs announced intent or explicit credit.
   Accuracy does not decide this signal. When the direction cannot be
   established, use null.
