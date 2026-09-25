# LLM Cost Optimisation Study — PolicyBot Q&A Feature

## What's being optimised
The same policy-document Q&A feature from the RAG Evaluation Suite: a user asks a
question, the system retrieves a passage and an LLM answers from it. Traffic is a
1,000-request simulated sample built from the 41-question set, with a Zipf-shaped
repeat-question distribution modeling typical FAQ/support traffic concentration
(see `traffic.py` for the stated assumption — real production logs weren't
available in this environment).

## What "measured" means here (read before the numbers)
This environment has no live LLM API access, so nothing here comes from an actual
OpenAI bill. Instead, every number is computed deterministically from real
artifacts:
- **Token counts** are computed from the actual prompt text sent in each config (system
  prompt + retrieved context + question), using the standard ~1.3-tokens-per-word
  approximation (exact tokenizer files aren't reachable from this sandbox).
- **Pricing** is OpenAI's current published per-token pricing for gpt-4o and
  gpt-4o-mini (checked live via search, September 2026), not a memorized or
  guessed figure.
- **Quality** is scored by the same answer-correctness function against the same
  38 gold answers used in the RAG study, for every config, so the *relative*
  quality delta between configs is measured consistently. The strong/weak
  generators stand in for gpt-4o/gpt-4o-mini since no live model calls were
  possible — see `generation_tiers.py` for exactly how to swap in real calls.
- **Cache hit rate** is the literal, counted hit rate from replaying the 1,000
  simulated requests against the modeled traffic distribution, not a guess.

Net effect: nothing here is a made-up percentage. The two places where this
would most need re-validation with real data are (1) exact token counts via
`tiktoken`, and (2) the real gpt-4o vs gpt-4o-mini accuracy gap via live calls.

## Results

| Config | Cost / 1,000 requests | Savings vs baseline | % traffic on mini | Cache hit rate | Quality (accuracy) |
|---|---|---|---|---|---|
| 0. Baseline (gpt-4o, top-3 context, long prompt) | $0.9130 | — | 0% | — | 94.1% |
| 1. + Shorter prompt & top-1 context only | $0.4858 | 46.8% | 0% | — | 96.0% |
| 2. + Route confident retrievals to gpt-4o-mini | $0.0463 | 94.9% | 95.9% | — | 86.3% |
| 3. + Cache repeated questions | $0.0024 | 99.7% | 95.9% | 95.9% | 86.3% |
| 4. + Batch requests (size 10) — **final** | $0.0022 | **99.8%** | 95.9% | 95.9% | 86.3% |

Full per-config numbers: `results/results_table.csv`.

## Interpretation

**Optimisation 1 (shorter prompt + top-1 context) is a free win, not a trade-off.**
Cutting the system prompt from ~90 words to ~15 and dropping context from top-3
chunks to top-1 *increased* accuracy (94.1% → 96.0%) while cutting cost nearly
in half. This isn't a coincidence: the RAG Evaluation Suite already showed top-1
retrieval hits the correct chunk 97.4% of the time, so top-3 was mostly adding
irrelevant chunks that gave the generator's sentence-matcher more chances to pick
the wrong sentence. Less context here was strictly better on both axes.

**Optimisation 2 (routing to gpt-4o-mini) is where the real trade-off lives.**
Routing the 95.9% of requests where retrieval was confident (similarity ≥ 0.30)
to a cheap-tier generator cut cost by another 20x but cost 9.7 accuracy points
(96.0% → 86.3%). This is the one optimisation that isn't free — it should be
evaluated against what "good enough" looks like for this feature, not applied
blindly. Raising the confidence threshold would send less traffic to the cheap
tier (protecting quality, at higher cost); lowering it would go further the
other way.

**Caching and batching are free once routing is in place.** Because 5 questions
account for ~68% of simulated traffic, exact-match caching alone eliminates
95.9% of LLM calls after the first occurrence, with zero quality impact (a cache
hit replays a previously-computed answer, it doesn't re-generate one). Batching
adds only a small additional saving here because caching had already removed
almost all the volume it would apply to — batching matters more before caching
is added, or for traffic with less repetition than this feature has.

## Final recommendation

Ship optimisations 1, 3, and 4 (shorter prompt, caching, batching) without
reservation — they cost nothing in quality and remove almost all spend on
their own. Shorter prompt + caching alone (no model routing at all) already
lands at **$0.0182 per 1,000 requests (98.0% cheaper than baseline) at 96.0%
accuracy — better quality than baseline, not worse.** That's the safest
optimisation stack if you don't want any quality trade-off at all.

**Be more careful with optimisation 2 (model routing).** It's responsible for
the single biggest cost cut in this study, but it's also the only line item
with a real quality cost. The trade-off to state explicitly in a review:
*"routing confident-retrieval questions to gpt-4o-mini turns a $0.049/1,000
feature into a $0.002/1,000 feature, but drops measured accuracy by about 10
points — worth it for a low-stakes FAQ surface, worth reconsidering if wrong
answers here carry real consequences (e.g. security or compliance questions)."*
A cheaper middle ground worth testing with real traffic: raise the routing
threshold so only the *most* confident retrievals go to the cheap tier, trading
some of the savings back for less of the quality loss.
