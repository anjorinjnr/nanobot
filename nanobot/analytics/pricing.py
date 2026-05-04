"""Single-source-of-truth LLM pricing table + cost estimator.

Both nanobot's per-call ``$ai_generation`` telemetry and Homer's
``tools/analytics/llm_call.py`` import from here, so a price bump is a
one-file change that ships to both producers via the next image rebuild.

Public API:

* :data:`USD_PER_MTOK` — model-id -> price tuple. Tuples are
  ``(input, output)`` or ``(input, output, cache_read)``; if cache_read
  is omitted, cache reads are billed at the input rate.
* :func:`estimate_cost_usd` — compute the USD cost of a single call.
* :func:`normalize_model_name` — accept the bare API name
  (``gemini-2.5-flash``) or the prefixed form
  (``gemini/gemini-2.5-flash``) and resolve to the price-table key.

Privacy / contract: this module deals only in counts and identifiers.
No prompt or completion content ever flows through it.
"""

from __future__ import annotations


# ── Pricing table (USD per 1M tokens) ────────────────────────────────────
# Tuples: (input, output) or (input, output, cache_read).
USD_PER_MTOK: dict[str, tuple[float, float] | tuple[float, float, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": (1.00, 5.00, 0.10),
    "claude-sonnet-4-6": (3.00, 15.00, 0.30),
    # Gemini
    "gemini/gemini-2.5-flash": (0.075, 0.30),
    "gemini/gemini-3-flash-preview": (0.30, 2.50),
    "gemini/gemini-3.1-pro-preview": (1.25, 10.00),
    # OpenRouter — DeepSeek
    "openrouter/deepseek/deepseek-chat-v3.2": (0.27, 0.41),
    "openrouter/deepseek/deepseek-chat-v3.2:free": (0.0, 0.0),
    # OpenRouter — Cerebras Qwen
    "cerebras/qwen-3-235b-a22b-instruct": (0.60, 1.20),
}


def normalize_model_name(model: str) -> str:
    """Match the emitted model string against the price-table key.

    Some callers pass the bare API name (``gemini-2.5-flash``) while
    nanobot's config uses the prefixed form (``gemini/gemini-2.5-flash``).
    Accept either.
    """
    if not model:
        return ""
    if model in USD_PER_MTOK:
        return model
    for prefix in ("gemini/", "openrouter/", "cerebras/"):
        if (prefix + model) in USD_PER_MTOK:
            return prefix + model
    return model


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Return a USD estimate for one call. 0.0 if the model isn't priced."""
    key = normalize_model_name(model)
    prices = USD_PER_MTOK.get(key)
    if prices is None:
        return 0.0
    in_rate, out_rate = prices[0], prices[1]
    cache_rate = prices[2] if len(prices) >= 3 else in_rate
    billed_input = max(0, input_tokens - cache_read_tokens)
    return (
        (billed_input * in_rate)
        + (cache_read_tokens * cache_rate)
        + (output_tokens * out_rate)
    ) / 1_000_000.0


__all__ = ["USD_PER_MTOK", "estimate_cost_usd", "normalize_model_name"]
