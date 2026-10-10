Your final answer was rejected: judge output @error@.

Continue this same investigation with at most @max_steps@ model calls,
including the final answer. Fix only what the rejection names; a formatting
error needs no new reads. If it names unread files, read them completely
(several paths per read_evidence call), or use null for a signal whose
evidence stays unread. Keep supported values, claims, and anchors; never
change an assessment only to pass validation.

Return the complete verdict as ONE JSON object: start with { and end with },
with no Markdown fences or commentary. Allowed values: relevance true|false;
accuracy @accuracy@; novelty @novelty@; influence @influence@;
positive_effect @positive_effect@.

@uncertainty@
