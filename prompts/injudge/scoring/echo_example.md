Worked example (the same-episode echo trap): the task statement names the
failing module, and at step 4 the agent reasons "the bug is the dtype cast
in copy(); I will preserve dtype on copy". A mid-episode extraction at
step 5 stores "copy() must preserve dtype" — a paraphrase of the agent's own
step-4 reasoning — and it is injected at step 6; the agent applies its plan
at step 7. Judge: novelty=redundant (the substance was established at step 4
and in the task statement, not new), influence=none (the fix continues the
agent's own pre-injection direction; the memory arrived after that direction
existed). Crediting the memory with the step-7 fix would be reversed
causality — the echo trap this rubric exists to refuse.
