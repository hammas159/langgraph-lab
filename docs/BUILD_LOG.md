# Build log

How this repo was actually built, in the order it happened. Full per-project accounts are in
each project's README under "Problems hit while building this" — this file pulls out the ones
that changed how later projects were built, or that generalise past a single project.

---

## Project 01 · Does the revision loop converge?

Built first, with no surprises in the graph itself — `generate -> critique -> revise -> critique`
compiled and ran correctly on the first attempt. The finding came entirely from reading the
score trajectory rather than trusting the final number: `t3`'s score sat at 0.00 through all six
forced revisions, which looked like a scorer bug until the actual draft text was read. It was a
real finding — fluent, technically empty prose that neither the model's own critique nor three
additional revisions ever flagged as missing anything.

## Project 02 · Does the router know when it's guessing?

`confidence_gate` scoring only 33% on **clean, unambiguous** tickets looked like a bug in the
gating logic. It was not: the router was classifying every clean ticket correctly and reporting
only MEDIUM confidence while doing it, so the gate correctly escalated four tickets that never
needed escalating. The premise the defence rests on — that confidence tracks correctness — was
the thing that was wrong, not the code.

## Project 03 · The synthesis that reads as complete

The most consequential check of the whole repo happened before any code was written for this
project: the plan assumed LangGraph silently picks a winner when two parallel branches write
the same state key. Tested directly against the installed version, it does not — it raises
`InvalidUpdateError` and refuses to run. The project was rebuilt around the failure mode that
actually exists (silent branch failure reaching a synthesis step) rather than the one that had
been assumed.

Two further surprises followed from that pivot:

- **Returning `{**state, ...}` from a parallel node fails even when every branch writes a
  different key**, because the spread also resubmits every *unchanged* field as an update —
  four branches all writing the same unchanged value to `model` still counts as four
  conflicting concurrent writes. Parallel nodes must return only their own delta.
- **A dict-valued key needs its own reducer too.** Each analyst returning `{"findings": {aspect:
  value}}` still produces four different dict values for one channel; `Annotated[dict,
  merge_dicts]` is what lets them combine instead of raising the same error one level down.

A signature-phrase scoring bug also showed up here first and shaped every project after it: a
single fixed phrase ("CTO left") scored a correct paraphrase ("departure of the CTO") as an
omission. Every subsequent project's constraint checks were built to avoid single-string
matching from the start rather than discover the same bug again.

## Project 04 · Does resuming re-run work that already happened?

**A live `Ledger` cannot live inside checkpointed graph state.** The first version put it
there, matching every other project's pattern. LangGraph's checkpointer serialises state between
invocations, so every resume operated on a *deserialised copy* of the ledger, disconnected from
the instance the benchmark was reading. Real model calls kept happening — the draft text
genuinely differed between rounds, proving the graph logic was correct — and the external call
count stayed at 1 regardless of how many resumes occurred. Fixed with a plain module-level
registry keyed by a checkpoint-safe string, never inside the graph's own state.

A second bug was arithmetic rather than architectural: the naive-strategy benchmark loop
conflated "get the initial draft" with "the final approval call" whenever a scenario had zero
revision rounds, understating the naive strategy's overhead on the simplest case. Restructuring
the loop into three explicit phases — initial draft, one call per revision round, one final
approval call — is what the reported "flat +1 regardless of revision count" result depends on.

## Project 05 · Does the constraint survive the handoff?

Applying the lesson from project 03, every constraint check here was reviewed against real
model output before being trusted, and three genuine false positives were still caught:

- The bare word **"peanut"** flagged a response correctly saying *"considering your peanut
  allergy"* and *"ensuring no peanuts are included"* — both safe, both flagged. Every forbidden
  list now names specific dishes and brands, never the allergen category word itself.
- A banned substring **`"$1"`** also matches **`"$100"`**, comfortably under an $800 ceiling.
  The budget check now extracts every dollar figure with a regex and compares the numeric value.
- A bare substring check on **"egg"** also matches **"eggplant"**, a vegan vegetable. That one
  term is checked with a word boundary; nothing else in the corpus needed it.

A fourth issue was conceptual rather than a bug: a response that loses the topic entirely
mentions no forbidden term by having said nothing relevant, which would otherwise score
identically to a genuinely safe answer. `on_topic()` is checked as a separate, required
condition — `safe_and_relevant` needs both — specifically so `last_message_only` could not look
artificially better than it is on the case where it goes silent rather than getting it wrong.

One flagged case (`summary_handoff` / Asana) was left in the results deliberately unresolved:
the response named the rejected product only to argue against it, not to recommend it, which is
a genuine judgement call rather than a clean failure. It is reported as such in the project
README rather than smoothed into a number that implies more confidence than the case supports.
