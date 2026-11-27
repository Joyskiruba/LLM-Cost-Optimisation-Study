# LLM Cost Optimisation Study

Baseline cost per 1,000 requests, three-plus optimisations applied, quality
measured before/after each, and a final recommendation with the trade-off
stated. Built on the same policy-doc Q&A feature as the RAG Evaluation Suite.

Interactive dashboard: https://joyskiruba.github.io/LLM-Cost-Optimisation-Study/dashboard/

## Structure
```
corpus/               8 policy docs (reused from the RAG Evaluation Suite)
qa_set.json           41 questions with gold answers (reused)
chunker.py            document chunking (reused)
retrieval.py          TF-IDF retriever (reused)
cost_model.py         token counting + real OpenAI pricing -> $ per request
generation_tiers.py   strong/weak generator standing in for gpt-4o / gpt-4o-mini
traffic.py            1,000-request simulated traffic w/ Zipf repeat distribution
evaluate_cost.py       runs baseline + each optimisation, writes results/
results/
  results_table.csv   one row per config: cost, savings %, quality, cache/routing stats
  summary.json
  analysis.md          written interpretation + final recommendation
```

## Run it
```bash
pip install -r requirements.txt
python3 evaluate_cost.py
```

## Read this before trusting the numbers
This environment has no live LLM API access. Read the docstrings at the top
of `cost_model.py` and `generation_tiers.py` for exactly what's a real
deterministic measurement (token counts of real prompts, real published
pricing, real cache hit rate) versus a stated modeling assumption (traffic
repeat distribution, the strong/weak tiers standing in for two real models).
Both files explain exactly how to swap in real `tiktoken` counts and real
`llm_generate()` calls (from the RAG Evaluation Suite repo) once you have API
access, without changing the rest of the harness.
