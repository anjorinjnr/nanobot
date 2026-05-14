"""Tests for analytics classifier API-key + route resolution.

Three-way precedence so the classifier can run on either side of the
OpenRouter consolidation:

  1. ``LLM_SYSTEM_API_KEY`` → OpenRouter (post-consolidation; platform-
     funded system sub-key).
  2. ``HOMER_ANALYTICS_GEMINI_API_KEY`` → direct Gemini (legacy Homer-
     owned analytics key).
  3. ``GEMINI_API_KEY`` → direct Gemini (dev/local fallback, charges
     tenant quota).
"""

from __future__ import annotations

from nanobot.analytics.classify import _resolve_api_key, _resolve_route


def _clear(monkeypatch):
    for v in ("LLM_SYSTEM_API_KEY", "HOMER_ANALYTICS_GEMINI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(v, raising=False)


def test_llm_system_key_wins_when_set(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_SYSTEM_API_KEY", "sk-or-v1-system")
    monkeypatch.setenv("HOMER_ANALYTICS_GEMINI_API_KEY", "homer_owned")
    monkeypatch.setenv("GEMINI_API_KEY", "tenant_owned")
    chosen = _resolve_route()
    assert chosen is not None
    api_key, url, model, provider = chosen
    assert api_key == "sk-or-v1-system"
    assert "openrouter.ai" in url
    assert model == "openrouter/auto"
    assert provider == "openrouter"


def test_legacy_homer_analytics_key_is_second(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("HOMER_ANALYTICS_GEMINI_API_KEY", "homer_owned")
    monkeypatch.setenv("GEMINI_API_KEY", "tenant_owned")
    chosen = _resolve_route()
    assert chosen is not None
    api_key, url, model, provider = chosen
    assert api_key == "homer_owned"
    assert "generativelanguage" in url
    assert provider == "gemini"


def test_falls_back_to_tenant_key_when_others_unset(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "tenant_owned")
    assert _resolve_api_key() == "tenant_owned"


def test_returns_empty_when_nothing_set(monkeypatch):
    _clear(monkeypatch)
    assert _resolve_api_key() == ""
    assert _resolve_route() is None


def test_strips_whitespace(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_SYSTEM_API_KEY", "  sk-or-v1-padded  ")
    assert _resolve_api_key() == "sk-or-v1-padded"


def test_blank_higher_priority_key_falls_through(monkeypatch):
    """An accidentally-empty higher-priority key must not blackhole the
    classifier."""
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_SYSTEM_API_KEY", "   ")
    monkeypatch.setenv("HOMER_ANALYTICS_GEMINI_API_KEY", "homer_owned")
    assert _resolve_api_key() == "homer_owned"
