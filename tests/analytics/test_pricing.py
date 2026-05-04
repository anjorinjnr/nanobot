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
