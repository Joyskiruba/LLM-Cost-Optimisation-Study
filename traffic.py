"""
Generates a 1,000-request traffic sample from the 41-question set.

ASSUMPTION (flagged explicitly): real production traffic wasn't available
in this environment, so repeat-question volume is modeled on a Zipf
distribution (a small number of questions account for most of the volume),
which is the well-documented shape of real FAQ/support traffic. This
assumption feeds the CACHING optimization's measured hit rate below --
change `ZIPF_A` or swap in real query logs to re-measure with actual data.

Everything downstream of this sample (token counts, costs, cache hits) is
computed exactly from the sample, not guessed.
"""
import json
import numpy as np

ZIPF_A = 1.3
N_REQUESTS = 1000
SEED = 42


def build_traffic(qa_path="qa_set.json"):
    with open(qa_path) as f:
        qa_set = json.load(f)

    rng = np.random.default_rng(SEED)
    n = len(qa_set)
    # Zipf-style weights: question i (0-indexed, already in arbitrary order)
    # gets weight 1/(rank^a); shuffle order first so it's not tied to topic order.
    order = rng.permutation(n)
    weights = np.array([1.0 / ((rank + 1) ** ZIPF_A) for rank in range(n)])
    weights = weights / weights.sum()

    sampled_positions = rng.choice(n, size=N_REQUESTS, p=weights)
    traffic = [qa_set[order[pos]] for pos in sampled_positions]
    return traffic


if __name__ == "__main__":
    traffic = build_traffic()
    from collections import Counter
    counts = Counter(q["id"] for q in traffic)
    print(f"{len(traffic)} requests, {len(counts)} distinct questions")
    top5 = counts.most_common(5)
    print("Most repeated questions:", top5)
    print(f"Top 5 questions account for {sum(c for _, c in top5)/len(traffic):.1%} of traffic")
