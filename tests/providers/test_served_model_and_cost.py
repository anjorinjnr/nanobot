"""Tests for OpenRouter served-model + authoritative-cost capture.

Cost dashboards rely on PostHog `$ai_generation` events being able to
answer two questions OpenRouter makes possible:

1. **Which SKU actually ran?** OR's auto-router and fallback chains can
   substitute the served model — we need `response.model` on every
   completion regardless of what we requested.
2. **What did OR actually charge?** OR returns `usage.cost` in USD when
   the request opts in. That number beats our pricing-table estimate
   because it accounts for promo credits, volume tiers, and per-route
   price drift with zero maintenance.

Coverage spans the three LLMResponse construction paths in
`OpenAICompatProvider`: dict-based `_parse`, SDK-based
`_parse_sdk_response`, and the streaming aggregator `_parse_chunks`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nanobot.providers.openai_compat_provider import OpenAICompatProvider


class _FakeSpec:
    supports_prompt_caching = False
    model_id_prefix = None
    strip_model_prefix = False
    max_completion_tokens = False
    reasoning_effort = None


def _provider() -> OpenAICompatProvider:
    p = OpenAICompatProvider.__new__(OpenAICompatProvider)
    p.client = MagicMock()
    p.spec = _FakeSpec()
    return p


# ── _extract_served_model ─────────────────────────────────────────────────


def test_extract_served_model_from_dict():
    assert OpenAICompatProvider._extract_served_model(
        {"model": "google/gemini-2.5-pro"},
    ) == "google/gemini-2.5-pro"


def test_extract_served_model_from_sdk_object():
    class _R:
        model = "openai/gpt-5.4-pro"

    assert OpenAICompatProvider._extract_served_model(_R()) == "openai/gpt-5.4-pro"


def test_extract_served_model_returns_none_when_absent():
    assert OpenAICompatProvider._extract_served_model({}) is None
    assert OpenAICompatProvider._extract_served_model(object()) is None


# ── _extract_cost_usd ─────────────────────────────────────────────────────


def test_extract_cost_usd_from_dict_usage():
    """OR returns `cost` (USD float) inside `usage` when the request
    opts into authoritative cost reporting."""
    assert OpenAICompatProvider._extract_cost_usd(
        {"usage": {"prompt_tokens": 100, "cost": 0.0123}},
    ) == 0.0123


def test_extract_cost_usd_from_sdk_usage_object():
    class _Usage:
        cost = 0.05

    class _R:
        usage = _Usage()

    assert OpenAICompatProvider._extract_cost_usd(_R()) == 0.05


def test_extract_cost_usd_returns_none_when_absent():
    """Direct-Gemini / direct-Anthropic responses don't carry `cost` —
    callers fall back to `$ai_total_cost_usd` (the estimate)."""
    assert OpenAICompatProvider._extract_cost_usd({"usage": {"prompt_tokens": 100}}) is None
    assert OpenAICompatProvider._extract_cost_usd({}) is None


def test_extract_cost_usd_handles_non_numeric_gracefully():
    """Telemetry must never crash callers. A bogus cost field (string,
    None, ...) returns None instead of raising."""
    assert OpenAICompatProvider._extract_cost_usd(
        {"usage": {"cost": "not-a-number"}},
    ) is None


# ── _parse: dict path populates model_served + cost_usd ───────────────────


def test_parse_dict_path_carries_served_model_and_cost():
    """The dict branch of `_parse` is the most common shape — used when
    the provider returns raw JSON (no SDK wrapper). Both fields must
    flow onto the LLMResponse so the agent's `_emit_ai_generation_event`
    can ship them as `$ai_model_served` / `$ai_cost_usd_served`."""
    p = _provider()
    response = {
        "model": "openai/gpt-5.4-pro",  # routed-to SKU, differs from request
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.0042,
        },
    }
    result = p._parse(response)
    assert result.model_served == "openai/gpt-5.4-pro"
    assert result.cost_usd == 0.0042


def test_parse_dict_path_no_cost_falls_through_to_none():
    """Provider returns model but no cost — cost_usd is None so the
    telemetry layer falls back to the pricing-table estimate."""
    p = _provider()
    response = {
        "model": "google/gemini-2.5-pro",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result = p._parse(response)
    assert result.model_served == "google/gemini-2.5-pro"
    assert result.cost_usd is None


# ── _parse: SDK-object path ───────────────────────────────────────────────


def test_parse_sdk_path_carries_served_model_and_cost():
    """The SDK-object branch of `_parse` is hit when the provider's
    Python SDK returns a Pydantic-like object instead of a raw dict.
    Same contract as the dict path."""
    class _Msg:
        content = "Hello"
        tool_calls = None
        reasoning_content = None

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Usage:
        prompt_tokens = 100
        completion_tokens = 50
        total_tokens = 150
        cost = 0.0042

    class _R:
        model = "openai/gpt-5.4-pro"
        choices = [_Choice()]
        usage = _Usage()

    p = _provider()
    result = p._parse(_R())
    assert result.model_served == "openai/gpt-5.4-pro"
    assert result.cost_usd == 0.0042


# ── _parse_chunks: streaming aggregator ──────────────────────────────────


def test_parse_chunks_picks_up_served_model_from_any_chunk():
    """Streaming responses scatter the routed-to model + final cost
    across multiple chunks. The aggregator must track the latest
    non-None of each so we don't lose them to a chunk that happened to
    have no model field."""
    chunks = [
        {
            "model": "openai/gpt-5.4-pro",
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        },
        # Mid-stream chunk without model — must NOT clobber.
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        # Final chunk with usage + cost.
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0099},
        },
    ]
    result = OpenAICompatProvider._parse_chunks(chunks)
    assert result.model_served == "openai/gpt-5.4-pro"
    assert result.cost_usd == 0.0099
    assert result.content == "Hello world"
    assert result.finish_reason == "stop"


def test_parse_chunks_no_model_no_cost_returns_none():
    """Non-OR streams (direct Anthropic, etc.) don't carry these — the
    aggregator returns None for both."""
    chunks = [
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
    ]
    result = OpenAICompatProvider._parse_chunks(chunks)
    assert result.model_served is None
    assert result.cost_usd is None
