The final answer failed the output contract: judge output @error@.

This is one repair call, not a new investigation. Evidence tools are disabled;
the full conversation and the rejected answer are already in your context.
Repair only what the validation error requires. Preserve supported signal
values, claims and anchors; do not invent evidence or change the assessment
just to make the object valid. Mere prose or Markdown around an otherwise
valid object needs only formatting repair.

Return the corrected verdict as ONE JSON object and nothing else. Start with
{ and end with }; no introduction, Markdown fences, or closing commentary.

Only if repair genuinely requires evidence not in this conversation, return
instead exactly {"needs_evidence": "<specific missing fact and the step/file to inspect>"}.
This requests renewed investigation; it is not a verdict. Formatting errors
alone do not require more evidence. Use null for a signal the recording itself
cannot establish, rather than requesting information the recording cannot contain.
