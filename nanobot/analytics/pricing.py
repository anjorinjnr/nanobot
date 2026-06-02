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
    # Gemini (direct via Google AI)
    "gemini/gemini-2.5-flash": (0.075, 0.30),
    "gemini/gemini-3-flash-preview": (0.30, 2.50),
    "gemini/gemini-3.1-pro-preview": (1.25, 10.00),
    # OpenRouter — DeepSeek
    # V4 (released 2026-04-24) is the current default tier: the
    # `default-cheap`/`cheap`/`deepseek-flash` presets emit
    # `deepseek/deepseek-v4-flash` (litellm prefixes to
    # `openrouter/deepseek/deepseek-v4-flash`); `deepseek-pro` emits the
    # pro variant. Without these rows default-tier calls price at $0.
    # Rates pulled from the OpenRouter models API (per-MTok); the third
    # element is the cache-read rate — V4 cache reads are ~5x (flash) /
    # ~120x (pro) cheaper than fresh input, so omitting it would bill
    # cache hits at the input rate and badly overstate Homer's cost.
    "openrouter/deepseek/deepseek-v4-flash": (0.0983, 0.1966, 0.0197),
    "openrouter/deepseek/deepseek-v4-pro": (0.4350, 0.8700, 0.0036),
    # V3.2 retained for back-compat with pre-V4 tasks/telemetry. Both
    # slugs exist in the wild (litellm-prefixed `deepseek/deepseek-v3.2`
    # and OR's `deepseek-chat-v3.2` / `:free` variants); one price tier.
    "openrouter/deepseek/deepseek-v3.2": (0.27, 0.41),
    "openrouter/deepseek/deepseek-chat-v3.2": (0.27, 0.41),
    "openrouter/deepseek/deepseek-chat-v3.2:free": (0.0, 0.0),
    # OpenRouter — Gemini (routed via OR rather than direct Google AI).
    # Pricing matches Google's published rates; OR pass-throughs cost.
    "openrouter/google/gemini-2.5-flash": (0.075, 0.30),
    "openrouter/google/gemini-2.5-pro": (1.25, 5.00),
    "openrouter/google/gemini-3-flash-preview": (0.30, 2.50),
    "openrouter/google/gemini-3.1-pro-preview": (1.25, 10.00),
    # OpenRouter — Cerebras Qwen
    "cerebras/qwen-3-235b-a22b-instruct": (0.60, 1.20),
}


def _derive_provider_prefixes() -> tuple[str, ...]:
    """Auto-derive provider prefixes from :data:`USD_PER_MTOK` keys.

    Anything appearing before the FIRST ``/`` in a key is a provider
    prefix; we collect them so :func:`normalize_model_name` can resolve
    bare API names without a hard-coded prefix list. Adding a new
    provider key automatically extends the prefix tuple — no second edit
    needed (issue #57).

    Includes ``bedrock/`` as forward-defense for AWS Bedrock model ids
    even before pricing rows land (issue #53).
    """
    prefixes: set[str] = {"bedrock/"}
    for key in USD_PER_MTOK:
        head, sep, _ = key.partition("/")
        if sep:
            prefixes.add(head + "/")
    # Stable order for deterministic iteration in tests / debugging.
    return tuple(sorted(prefixes))


_PROVIDER_PREFIXES: tuple[str, ...] = _derive_provider_prefixes()


def normalize_model_name(model: str) -> str:
    """Match the emitted model string against the price-table key.

    Some callers pass the bare API name (``gemini-2.5-flash``) while
    nanobot's config uses the prefixed form (``gemini/gemini-2.5-flash``).
    Accept either.

    The set of prefixes is auto-derived from :data:`USD_PER_MTOK` keys
    plus a small set of forward-defense entries (e.g. ``bedrock/``) —
    adding a priced model under a new provider auto-extends the lookup.
    """
    if not model:
        return ""
    if model in USD_PER_MTOK:
        return model
    for prefix in _PROVIDER_PREFIXES:
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
    """Return a USD estimate for one call. 0.0 if the model isn't priced.

    Token-counting contract:

    * ``input_tokens`` is the *total* prompt size for the call, including
      any portion served from a provider cache.
    * ``cache_read_tokens`` is the cached subset of ``input_tokens``;
      it is billed at the model's cache-read rate (column 3 of the price
      tuple, falling back to the input rate when omitted).
    * The non-cached portion of the prompt — billed at the full input
      rate — is therefore ``max(0, input_tokens - cache_read_tokens)``.

    Output tokens are billed independently at the output rate.
    """
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
