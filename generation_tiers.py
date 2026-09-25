"""
Two answer-quality tiers, standing in for gpt-4o (strong) vs gpt-4o-mini
(weak) since this environment has no live API access.

QUALITY CAVEAT (read this before trusting the accuracy numbers):
These are two different deterministic heuristics, not two different real
LLM calls -- there is no way to get an actual GPT-4o vs GPT-4o-mini quality
delta without a live API key. What IS real: both tiers are scored against
the same 38 gold answers with the same scoring function, so the *relative*
quality gap between tiers, and the effect of routing on that gap, is
measured consistently -- just not calibrated to the real GPT-4o/mini gap.
To get the real gap, swap `strong_generate`/`weak_generate` for calls to
`generation.llm_generate()` (from the RAG Evaluation Suite repo) with
model="gpt-4o" and model="gpt-4o-mini" respectively, on a machine with API
access.
"""
import re

ABSTAIN_MSG = "I don't have enough information in the provided context to answer that."
STOPWORDS = {
    "the", "a", "an", "is", "are", "does", "do", "for", "of", "to", "in",
    "on", "and", "or", "what", "how", "many", "much", "under", "at", "must",
    "can", "with", "before", "after", "within",
}


def _keywords(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def strong_generate(question, context_text):
    """Stand-in for gpt-4o: picks the best-matching sentence by keyword overlap."""
    body = re.sub(r"^##\s+.+\n", "", context_text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", body)
    q_kw = _keywords(question)
    best_sentence, best_overlap = None, -1
    for s in sentences:
        overlap = len(q_kw & _keywords(s))
        if overlap > best_overlap:
            best_overlap, best_sentence = overlap, s
    if best_sentence is None or best_overlap == 0:
        return ABSTAIN_MSG
    return best_sentence.strip()


def weak_generate(question, context_text):
    """Stand-in for gpt-4o-mini: no matching logic, just returns the first sentence."""
    body = re.sub(r"^##\s+.+\n", "", context_text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", body)
    return sentences[0].strip() if sentences else ABSTAIN_MSG


def answer_correct(gold_answer, generated_text):
    if gold_answer is None:
        return None
    gold_kw = _keywords(gold_answer)
    if not gold_kw:
        return gold_answer.lower() in generated_text.lower()
    gen_kw = _keywords(generated_text)
    return len(gold_kw & gen_kw) / len(gold_kw) >= 0.6
