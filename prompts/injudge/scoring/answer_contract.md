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
                          "note": "<one sentence citing the task statement or the pre-injection step>"},
      "influence":       {"value": @influence@,
                          "note": "<one sentence>"},
      "positive_effect": {"value": @positive_effect@,
                          "note": "<one sentence>"},
      "anchors":         [<int>...]
    }
  ],
  "analysis": "<2-4 sentences synthesizing the span's footprint>",
  "evidence": [{"claim": "<what happened>", "steps": [<int>...],
                "seqs": [<int>...]}...],
  "confidence": <number 0.0-1.0>
}
memories must contain exactly one entry per delivered memory id, no more and
no less. `anchors` lists the agent-step numbers where this memory's
footprint appears, inside its footprint window: REQUIRED and non-empty
whenever influence is acted_on or decisive (the steps where the shaped
action happened — the span's delivery step counts when the shaped action
happened in that very response; a step is not an anchor merely because the
memory was delivered or mentioned there, and the plan behind an action
executed later is not one either); optional for referenced;
empty for none. Anchors outside the memory's window are rejected: the
window states where this memory's influence may be claimed.
