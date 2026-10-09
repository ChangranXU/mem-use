Your entire final response must be ONE JSON object: start with { and end
with }. No introduction, Markdown fences, or text outside the object.
Use exactly this shape:
{
  "memories": [
    {
      "id": "<the memory's id from the delivered-memories list, e.g. m1>",
      "relevance":       {"value": true|false, "note": "<one sentence>"},
      "accuracy":        {"value": @accuracy@,
                          "note": "<one sentence>"},
      "novelty":         {"value": @novelty@,
                          "note": "<baseline citation, or specific gap for null>"},
      "influence":       {"value": @influence@,
                          "note": "<action step; memory-specific content it uses, or the quoted prior source>"},
      "positive_effect": {"value": @positive_effect@,
                          "note": "<result step and demonstrated consequence, or what remains unknown>"},
      "anchors":         [<int>...]
    }
  ],
  "analysis": "<At most 60 words: observed actions and memory attribution>",
  "evidence": [{"claim": "<what happened>", "steps": [<int>...],
                "seqs": [<int>...]}...],
  "confidence": <number 0.0-1.0>
}
Keep causal-source arguments in entry notes: an influence note names the
memory-specific content an action uses, or the quoted prior source or
routine practice that explains it. Keep analysis and evidence claims
consistent with those notes: when attribution is null, do not summarize
either memory or its competing source as the established cause.
memories has exactly one entry per delivered id. anchors lists in-window
steps where the shaped action or decision happened, including the consuming
step when applicable. A decision can anchor credit if later execution
substantiates it; an unexecuted plan earns no action credit. Delivery or
mention alone is not an action anchor. Anchors are required for acted_on or
selected, optional for referenced or null, empty for none; out-of-window
anchors are rejected. positive_effect must be null unless influence is
acted_on. Read the complete text of every step cited in anchors or evidence;
paged reads of a step file combine.
