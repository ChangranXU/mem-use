Answer with ONE JSON object and nothing else:
{
  "memories": [
    {
      "key": "<memory key>",
      "relevance":       {"value": true|false, "note": "<one sentence>"},
      "accuracy":        {"value": @accuracy@,
                          "note": "<one sentence>"},
      "novelty":         {"value": @novelty@,
                          "note": "<one sentence citing the pre-injection step>"},
      "influence":       {"value": @influence@,
                          "note": "<one sentence>"},
      "positive_effect": {"value": @positive_effect@,
                          "note": "<one sentence>"}
    }
  ],
  "analysis": "<2-4 sentences synthesizing the span's footprint>",
  "evidence": [{"claim": "<what happened>", "steps": [<int>...],
                "seqs": [<int>...]}...],
  "confidence": <number 0.0-1.0>
}
memories must contain exactly one entry per delivered memory key.