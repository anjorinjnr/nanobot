"""Tests for :mod:`nanobot.analytics.pricing`.

These exercise the canonical pricing table + cost estimator that both
nanobot's per-call ``$ai_generation`` telemetry and Homer's
``tools/analytics/llm_call.py`` rely on. A regression here is a
cross-repo regression.
"""

from __future__ import annotations

import pytest

from nanobot.analytics.pricing import (
    USD_PER_MTOK,
    _PROVIDER_PREFIXES,
    estimate_cost_usd,
    normalize_model_name,
)


# ── cost table ───────────────────────────────────────────────────────────


def test_cost_table_haiku_one_million_each():
    """1M input + 1M output @ Haiku rates = $1.00 + $5.00 = $6.00."""
    cost = estimate_cost_usd(
        "claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx(6.00, abs=1e-9)


def test_cost_table_unknown_model_returns_zero():
    assert estimate_cost_usd("not-a-real-model", input_tokens=10_000, output_tokens=10_000) == 0.0


def test_cost_table_normalizes_bare_gemini_name():
    """Caller may pass ``gemini-2.5-flash`` instead of ``gemini/gemini-2.5-flash``."""
    bare = estimate_cost_usd("gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    full = estimate_cost_usd("gemini/gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    assert bare == full == pytest.approx(0.075 + 0.30, abs=1e-9)


def test_cost_table_cache_read_billed_at_cache_rate():
    """Anthropic cache read is billed at 0.10/MTok, not 1.00 input."""
    full_input = estimate_cost_usd(
        "claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=0,
    )
    half_cached = estimate_cost_usd(
        "claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=500_000,
    )
    # 500k @ $1/MTok + 500k @ $0.10/MTok = $0.55, vs all-fresh $1.00.
    assert full_input == pytest.approx(1.00, abs=1e-9)
    assert half_cached == pytest.approx(0.55, abs=1e-9)


def test_cost_table_free_models_are_zero():
    cost = estimate_cost_usd(
        "openrouter/deepseek/deepseek-chat-v3.2:free",
        input_tokens=10_000_000,
        output_tokens=10_000_000,
    )
    assert cost == 0.0


def test_cost_table_prices_deepseek_v32_slug():
    """`switch_model.py --model default-cheap` emits
    `deepseek/deepseek-v3.2` (no `-chat-`), which litellm prefixes to
    `openrouter/deepseek/deepseek-v3.2`. Before this row was added the
    lookup missed and every default-tier chat/heartbeat tick recorded
    `$ai_total_cost_usd = 0`, which is what hid Esther's pre-hotfix
    spam window from cost dashboards.
    """
    cost = estimate_cost_usd(
        "openrouter/deepseek/deepseek-v3.2",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    # 0.27 input + 0.41 output per Mtok
    assert cost == pytest.approx(0.68, abs=1e-9)


def test_cost_table_prices_openrouter_gemini_routes():
    """Hosted-default tenants on `google/gemini-2.5-pro` via OR were
    showing $0 cost on every chat turn until these rows landed —
    pricing parity with direct-Gemini, just routed through OR.
    """
    cost = estimate_cost_usd(
        "openrouter/google/gemini-2.5-pro",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    # 1.25 input + 5.00 output per Mtok
    assert cost == pytest.approx(6.25, abs=1e-9)


# ── normalize_model_name ─────────────────────────────────────────────────


def test_normalize_returns_empty_for_empty_input():
    assert normalize_model_name("") == ""


def test_normalize_passthrough_when_already_keyed():
    assert normalize_model_name("claude-haiku-4-5-20251001") == "claude-haiku-4-5-20251001"


def test_normalize_adds_gemini_prefix():
    assert normalize_model_name("gemini-2.5-flash") == "gemini/gemini-2.5-flash"


def test_normalize_returns_input_when_unmatched():
    """Unknown models pass through unchanged so the price lookup misses cleanly."""
    assert normalize_model_name("not-a-model") == "not-a-model"


def test_pricing_table_keys_are_strings_and_tuples_well_formed():
    for key, prices in USD_PER_MTOK.items():
        assert isinstance(key, str) and key
        assert isinstance(prices, tuple)
        assert len(prices) in (2, 3)
        for rate in prices:
            assert isinstance(rate, (int, float))
            assert rate >= 0


# ── Issue #53: bedrock/ prefix forward-defense ────────────────────────────


def test_provider_prefixes_includes_bedrock():
    """``bedrock/`` is a known provider prefix even when no bedrock/* row
    is yet priced — keeps ``normalize_model_name`` future-proof."""
    assert "bedrock/" in _PROVIDER_PREFIXES


def test_normalize_unmatched_bedrock_passthrough():
    """Bare bedrock model name passes through (no priced row → returned as-is).

    The point of #53 is that ADDING ``bedrock/foo`` to ``USD_PER_MTOK``
    later will make ``normalize_model_name("foo")`` resolve without a
    second edit. We can't assert that without seeding the table; the
    weaker contract is that the prefix is present in the lookup tuple
    so the future case doesn't silently miss.
    """
    assert normalize_model_name("foo-bedrock-model") == "foo-bedrock-model"


# ── Issue #57: prefix tuple is auto-derived from USD_PER_MTOK ─────────────


def test_provider_prefixes_covers_every_prefixed_key():
    """Every priced model with a ``provider/`` prefix must be lookup-resolvable
    via the bare name. Locks the auto-derivation so a new provider key
    can't sneak in without ``normalize_model_name`` learning about it.
    """
    for key in USD_PER_MTOK:
        head, sep, _rest = key.partition("/")
        if not sep:
            continue
        prefix = head + "/"
        assert prefix in _PROVIDER_PREFIXES, (
            f"Provider prefix {prefix!r} (from key {key!r}) is missing "
            "from auto-derived _PROVIDER_PREFIXES — issue #57 regression."
        )


def test_provider_prefixes_resolves_bare_names_for_every_prefixed_key():
    """Sanity: each prefixed key's bare form normalizes back to the prefixed key.
    Catches any auto-derivation bug that drops keys with multiple ``/``s
    (e.g. ``openrouter/deepseek/deepseek-chat-v3.2``).
    """
    for key in USD_PER_MTOK:
        head, sep, rest = key.partition("/")
        if not sep:
            continue
        # The bare name is everything after the first ``/``.
        assert normalize_model_name(rest) == key, (
            f"normalize_model_name({rest!r}) failed to resolve to {key!r}"
        )


# ── Issue #56: estimate_cost_usd docstring is now precise about the contract ──


def test_estimate_cost_usd_docstring_documents_cache_read_subset():
    """Lock that the docstring spells out the cache_read_tokens contract."""
    doc = estimate_cost_usd.__doc__ or ""
    assert "input_tokens" in doc
    assert "cache_read_tokens" in doc
    # The exact arithmetic is in the docstring so callers don't have to
    # read the source. (#56)
    assert "max(0, input_tokens - cache_read_tokens)" in doc


def test_cache_read_subset_arithmetic_is_correct():
    """Direct algebraic check of the subset contract from the docstring.

    cache_read_tokens is the cached subset of input_tokens; non-cached
    portion is input - cache_read, billed at the input rate.
    """
    cost = estimate_cost_usd(
        "claude-haiku-4-5-20251001",  # (1.00, 5.00, 0.10) per MTok
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=300_000,
    )
    # 700k @ $1.00/MTok + 300k @ $0.10/MTok = $0.70 + $0.03 = $0.73
    assert cost == pytest.approx(0.73, abs=1e-9)


def test_cache_read_clamps_when_exceeds_input():
    """If cache_read > input (schema drift), billed_input clamps to 0."""
    cost = estimate_cost_usd(
        "claude-haiku-4-5-20251001",
        input_tokens=100_000,
        output_tokens=0,
        cache_read_tokens=500_000,
    )
    # 0 @ input + 500k @ $0.10/MTok = $0.05
    assert cost == pytest.approx(0.05, abs=1e-9)
