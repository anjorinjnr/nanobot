"""Tests for analytics classifier API-key resolution.

Hosted tenants get a Homer-owned `HOMER_ANALYTICS_GEMINI_API_KEY` so the
classifier never charges tenant Gemini quota and never silently fails when
the tenant doesn't use Gemini for chat. The tenant key is the dev/local
fallback.
"""

from __future__ import annotations

from nanobot.analytics.classify import _resolve_api_key


def test_prefers_analytics_key_when_both_set(monkeypatch):
    monkeypatch.setenv("HOMER_ANALYTICS_GEMINI_API_KEY", "homer_owned")
    monkeypatch.setenv("GEMINI_API_KEY", "tenant_owned")
    assert _resolve_api_key() == "homer_owned"


def test_falls_back_to_tenant_key_when_analytics_unset(monkeypatch):
    monkeypatch.delenv("HOMER_ANALYTICS_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "tenant_owned")
    assert _resolve_api_key() == "tenant_owned"


def test_returns_empty_when_neither_set(monkeypatch):
    monkeypatch.delenv("HOMER_ANALYTICS_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert _resolve_api_key() == ""


def test_strips_whitespace(monkeypatch):
    monkeypatch.setenv("HOMER_ANALYTICS_GEMINI_API_KEY", "  padded  ")
    assert _resolve_api_key() == "padded"


def test_blank_analytics_key_falls_back(monkeypatch):
    """An accidentally-empty analytics key must not blackhole the classifier."""
    monkeypatch.setenv("HOMER_ANALYTICS_GEMINI_API_KEY", "   ")
    monkeypatch.setenv("GEMINI_API_KEY", "tenant_owned")
    assert _resolve_api_key() == "tenant_owned"
