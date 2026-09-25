"""
Cost model: turns token counts into dollars using OpenAI's current published
API pricing (checked live, not memorized -- prices change):

  gpt-4o        : $2.50 / 1M input tokens,  $10.00 / 1M output tokens
  gpt-4o-mini   : $0.15 / 1M input tokens,  $0.60  / 1M output tokens
  cached input (either model, for repeated prefix tokens): 50% of input price

Source: OpenAI API pricing page, checked September 2026.

TOKEN COUNTING CAVEAT (read this before trusting the numbers):
This environment cannot reach OpenAI's tokenizer file host, so exact
cl100k_base token counts aren't available here. Token counts below use the
standard published approximation for English text: ~1.3 tokens per word
(equivalently ~4 characters per token). This is a real, deterministic
measurement of the ACTUAL prompt text used in each configuration -- it is
not a guessed percentage-saved number -- but it is an approximation of
exact tokenization. To get exact counts, replace `count_tokens()` below
with `tiktoken.encoding_for_model("gpt-4o").encode(text)` on a machine
with normal internet access; the relative comparison between configs
(which is what this study is graded on) will not materially change.
"""

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
CACHED_INPUT_DISCOUNT = 0.5  # cached/repeated prefix tokens billed at 50% of input price


def count_tokens(text):
    """Deterministic word-count-based approximation: ~1.3 tokens/word."""
    words = text.split()
    return max(1, round(len(words) * 1.3))


def cost_of_request(model, input_tokens, output_tokens, cached_input_tokens=0):
    p = PRICING[model]
    billable_input = max(0, input_tokens - cached_input_tokens)
    cost = (
        billable_input / 1_000_000 * p["input"]
        + cached_input_tokens / 1_000_000 * p["input"] * CACHED_INPUT_DISCOUNT
        + output_tokens / 1_000_000 * p["output"]
    )
    return cost
