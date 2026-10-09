4. influence (@influence@) — what this entry demonstrably shaped inside its
   stated footprint window. Decide in this order:

   a. Prior sources are the records before the first consuming step: task.md,
      baseline.md's visible input, earlier steps (reasoning, commands,
      queries, tool results), and other deliveries already in that input.
      A prior source states content only if you can quote the same specific
      fact or instruction from it: the same identifier, value, location,
      command, check, or remedy, applied to the same purpose. A pattern used
      elsewhere, content a capable agent could derive, outside knowledge, and
      anything first recorded at or after the consuming step are not prior
      sources.
   b. Memory-specific content is a specific mechanism, location,
      identifier, value, command, check, or remedy this entry exposed that
      no prior source states. Routine practice (reproducing the failure,
      reading the code at hand, running tests, keeping changes small, in a
      usual order) is not memory-specific unless the action uses a detail
      only this entry named.
   c. Find every in-window action that uses this entry's content: commands,
      edits, tests, searches, and decisions that execution later carries
      out. Check commands and results, not only reasoning.

   Then label:
   - acted_on: an action uses memory-specific content, or the agent
     explicitly credits this entry for an action, or the agent turns from a
     recorded prior direction to the one this entry names right after its
     delivery. Anchor those actions. For an action that mixes sources,
     credit only its memory-specific part;
   - selected: its shown identifier or preview caused retrieval of fuller
     memory content that the record shows as its own memory delivery (the
     step's result reads [memory delivery …]; span.md lists proven
     follow-ups), with no other use of this entry's content. Anchor the
     retrieval action. Selection proves retrieval, not benefit:
     positive_effect is null, and the fuller delivery's effect is judged on
     that delivery. A retrieval whose result is not a recorded delivery, or
     opening repository code or task data at a path the memory supplied, is
     acted_on;
   - referenced: the agent mentions or quotes this entry (its shown
     identifier, distinctive content, or list position; a remark about the
     whole result list references no single entry unless this entry was the
     only one shown) with no action established;
   - null: an action uses content that, before the delivery, was stated only
     as an open possibility (the agent's unverified hypothesis, question, or
     search query) or only in a prior source the agent had not acted on,
     and the record shows nothing that separates the two; or the window was
     not inspected completely. Name the action and the quoted prior source,
     or the unread steps;
   - none: after reading the complete window, no reference, selection, or
     attributable action was established, and no action remains ambiguous
     under the null rule above. An action that only carries out what the
     task instructs, or what the agent decided or did before the delivery,
     is explained by that source. An open possibility or a prior source the
     agent had not acted on is not a settled decision: apply null first.

   Deliveries of other items are prior sources once they are in the agent's
   input. When a preview's shown identifier selected a recorded fuller
   delivery of the same item, content both carry is the fuller delivery's to
   credit; the preview is acted_on only for an action taken before that
   delivery arrived or with content only the preview showed. Exact
   same-item, same-text copies consumed together are one source. Judge each
   entry by its own window and exposed text, never hidden keys or an unseen
   suffix. Anchors stay in the entry's window; later results can show an
   anchored action's consequence without becoming anchors.
