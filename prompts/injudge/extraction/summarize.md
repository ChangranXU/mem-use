You compress evidence items collected for one injection span's audit: the
items below exceed their token budget. Answer with ONE JSON object and
nothing else:
@shape@

Rules (binding):
- The compressed items' combined size must stay within @max_tokens@ tokens.
- Keep every distinct fact the judge cannot recover otherwise; merge
  duplicates; keep each item's step/seq citation.
- Never invent facts, never emit a step range for the set as a whole —
  the assembler stamps it.