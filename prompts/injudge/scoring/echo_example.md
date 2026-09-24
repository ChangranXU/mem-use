Example (echo): at step 4 the agent writes "the bug is the dtype cast in
copy(); I will preserve dtype". A memory saying "copy() must preserve dtype"
is consumed at step 6, and the agent applies its fix at step 7 without
mentioning it: novelty=redundant (step 4), influence=none, since crediting
the memory would reverse causality. If the agent says "the memory confirms
my plan", influence is referenced. Had that reasoning first appeared at step
6 or later, it would be post-exposure and could not show a prior plan.
