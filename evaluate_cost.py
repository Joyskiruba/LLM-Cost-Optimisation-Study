"""
Runs the LLM Cost Optimisation Study: baseline cost/1000 requests, then
three optimisations applied cumulatively, with quality measured before and
after each. See cost_model.py and generation_tiers.py for the caveats on
what's exactly-measured vs. modeled in this network-restricted environment.
"""
import json
import csv
from collections import defaultdict, Counter

from chunker import load_chunks
from retrieval import Retriever
from traffic import build_traffic
from cost_model import count_tokens, cost_of_request
from generation_tiers import strong_generate, weak_generate, answer_correct

LONG_PROMPT = (
    "You are a helpful and knowledgeable assistant for Northwind SaaS employees. "
    "Your job is to answer employee questions about internal company policy accurately "
    "and helpfully, using only the context passages provided to you below. Please read "
    "the context carefully before answering. If the answer is not contained in the "
    "context provided, let the user know politely that you don't have that information "
    "rather than guessing. Keep your answer clear, professional, and concise, and cite "
    "which policy area the answer comes from where possible."
)  # ~90 words

SHORT_PROMPT = (
    "Answer the question using only the context below. "
    "If it's not covered, say you don't know."
)  # ~15 words

RETRIEVAL_ROUTE_THRESHOLD = 0.30  # >= this similarity -> confident enough for the cheap tier
CONFIG_BATCH_SIZE_FINAL = 10


def make_context(retrieved, k):
    return "\n\n".join(r["text"] for r in retrieved[:k])


def run_config(name, chunks, retriever, traffic, *, context_k, system_prompt,
                route_by_confidence, use_cache, batch_size):
    """
    Simulates one configuration over the traffic sample.
    Returns per-request records; cost accounting respects caching and batching.
    """
    cache = {}  # question_id -> cached answer text (cache hit = no LLM call)
    records = []

    # Pre-compute retrieval + tier + answer once per UNIQUE question (deterministic),
    # then apply that to every occurrence in the traffic stream.
    unique_qs = {q["id"]: q for q in traffic}
    per_question = {}
    for qid, qa in unique_qs.items():
        retrieved = retriever.query(qa["question"], k=max(context_k, 1))
        top_score = retrieved[0]["score"]
        context_text = make_context(retrieved, context_k)

        if route_by_confidence and top_score >= RETRIEVAL_ROUTE_THRESHOLD:
            model, gen_fn = "gpt-4o-mini", weak_generate
        else:
            model, gen_fn = "gpt-4o", strong_generate

        answer = gen_fn(qa["question"], context_text)
        input_text = system_prompt + "\n\n" + context_text + "\n\nQuestion: " + qa["question"]
        input_tokens = count_tokens(input_text)
        output_tokens = count_tokens(answer)

        correct = answer_correct(qa["answer"], answer) if qa["answerable"] else None
        per_question[qid] = {
            "model": model, "answer": answer, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "correct": correct,
            "answerable": qa["answerable"],
        }

    # Now walk the traffic stream in order, applying caching and batching.
    total_cost = 0.0
    cache_hits = 0
    batch_queue = defaultdict(list)  # model -> pending requests waiting to be "flushed" as a batch

    def flush_batch(model):
        nonlocal total_cost
        pending = batch_queue[model]
        if not pending:
            return
        # system prompt billed once per batch instead of once per request
        sys_tokens = count_tokens(system_prompt)
        total_cost_local = cost_of_request(model, sys_tokens, 0)
        for input_tok_minus_sys, out_tok in pending:
            total_cost_local += cost_of_request(model, input_tok_minus_sys, out_tok)
        total_cost += total_cost_local
        batch_queue[model] = []

    for qa in traffic:
        qid = qa["id"]
        pq = per_question[qid]

        if use_cache and qid in cache:
            cache_hits += 1
            records.append({**pq, "id": qid, "cache_hit": True, "cost": 0.0})
            continue
        if use_cache:
            cache[qid] = pq["answer"]

        if batch_size > 1:
            sys_tokens = count_tokens(system_prompt)
            rest_input_tokens = max(1, pq["input_tokens"] - sys_tokens)
            batch_queue[pq["model"]].append((rest_input_tokens, pq["output_tokens"]))
            records.append({**pq, "id": qid, "cache_hit": False, "cost": None})  # cost settled on flush
            if len(batch_queue[pq["model"]]) >= batch_size:
                flush_batch(pq["model"])
        else:
            cost = cost_of_request(pq["model"], pq["input_tokens"], pq["output_tokens"])
            total_cost += cost
            records.append({**pq, "id": qid, "cache_hit": False, "cost": cost})

    # flush any remaining partial batches
    for model in list(batch_queue.keys()):
        flush_batch(model)

    n = len(traffic)
    n_answerable_hits = sum(1 for r in records if r["answerable"])
    n_correct = sum(1 for r in records if r["answerable"] and r["correct"])
    model_mix = Counter(r["model"] for r in records)

    return {
        "config": name,
        "cost_per_1000_requests": round(total_cost / n * 1000, 4),
        "cache_hit_rate": round(cache_hits / n, 3) if use_cache else None,
        "pct_requests_on_mini": round(model_mix.get("gpt-4o-mini", 0) / n, 3),
        "quality_accuracy": round(n_correct / n_answerable_hits, 3) if n_answerable_hits else None,
        "n_requests": n,
    }


def main():
    chunks = load_chunks()
    retriever = Retriever(chunks)
    traffic = build_traffic()

    configs = [
        dict(name="0_baseline", context_k=3, system_prompt=LONG_PROMPT,
             route_by_confidence=False, use_cache=False, batch_size=1),
        dict(name="1_shorter_prompt_and_context", context_k=1, system_prompt=SHORT_PROMPT,
             route_by_confidence=False, use_cache=False, batch_size=1),
        dict(name="2_add_model_routing", context_k=1, system_prompt=SHORT_PROMPT,
             route_by_confidence=True, use_cache=False, batch_size=1),
        dict(name="2b_caching_no_routing_ablation", context_k=1, system_prompt=SHORT_PROMPT,
             route_by_confidence=False, use_cache=True, batch_size=1),
        dict(name="3_add_caching", context_k=1, system_prompt=SHORT_PROMPT,
             route_by_confidence=True, use_cache=True, batch_size=1),
        dict(name="4_add_batching_FINAL", context_k=1, system_prompt=SHORT_PROMPT,
             route_by_confidence=True, use_cache=True, batch_size=CONFIG_BATCH_SIZE_FINAL),
    ]

    results = []
    for cfg in configs:
        r = run_config(cfg["name"], chunks, retriever, traffic,
                        context_k=cfg["context_k"], system_prompt=cfg["system_prompt"],
                        route_by_confidence=cfg["route_by_confidence"],
                        use_cache=cfg["use_cache"], batch_size=cfg["batch_size"])
        results.append(r)

    with open("results/results_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    baseline_cost = results[0]["cost_per_1000_requests"]
    for r in results:
        r["pct_savings_vs_baseline"] = round((1 - r["cost_per_1000_requests"] / baseline_cost) * 100, 1)

    print(json.dumps(results, indent=2))
    with open("results/summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
