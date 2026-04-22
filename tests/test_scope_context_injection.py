"""Contract tests for AgentLoop scope_context_provider injection hook.

These tests lock down the wire between the nanobot fork and homer's scope_store:
a single "module:function" string is resolved once, called with sender_id, and
its return value (if a non-empty string) is injected as an ephemeral system
message. The fork has no knowledge of scope_store internals — it just honors
this contract.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from nanobot.agent.loop import AgentLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_loop(workspace: Path, provider: str = "") -> AgentLoop:
    """Build a minimal AgentLoop shell — bypasses __init__ side-effects."""
    loop = AgentLoop.__new__(AgentLoop)
    loop.workspace = workspace
    loop._scope_context_provider = provider
    loop._scope_context_fn = None
    return loop


def _install_mock_module(
    monkeypatch, mod_name: str, fn_name: str, fn,
) -> None:
    """Register a fake Python module so importlib.import_module finds it."""
    module = types.ModuleType(mod_name)
    setattr(module, fn_name, fn)
    monkeypatch.setitem(sys.modules, mod_name, module)


# ---------------------------------------------------------------------------
# Provider disabled (empty string)
# ---------------------------------------------------------------------------

class TestProviderUnset:
    def test_returns_none_when_provider_is_empty(self, tmp_path):
        loop = _mk_loop(tmp_path, provider="")
        assert loop._get_scope_context("sender@example.com") is None

    def test_returns_none_when_sender_is_empty(self, tmp_path):
        loop = _mk_loop(tmp_path, provider="some_mod:some_fn")
        assert loop._get_scope_context("") is None


# ---------------------------------------------------------------------------
# Provider valid + callable — happy paths
# ---------------------------------------------------------------------------

class TestProviderHappyPath:
    def test_returns_provider_output(self, tmp_path, monkeypatch):
        def provider_fn(sender_id: str) -> str:
            return f"# Scope Context\nsender={sender_id}\n"
        _install_mock_module(monkeypatch, "fake_scope_mod", "render_ctx", provider_fn)

        loop = _mk_loop(tmp_path, provider="fake_scope_mod:render_ctx")
        out = loop._get_scope_context("15551234567@s.whatsapp.net")
        assert out is not None
        assert "sender=15551234567@s.whatsapp.net" in out

    def test_function_is_cached_after_first_call(self, tmp_path, monkeypatch):
        call_count = {"imports": 0}

        def provider_fn(sender_id: str) -> str:
            call_count["imports"] += 1
            return "x"
        _install_mock_module(monkeypatch, "fake_mod_cache", "fn", provider_fn)

        loop = _mk_loop(tmp_path, provider="fake_mod_cache:fn")
        loop._get_scope_context("a")
        loop._get_scope_context("b")
        loop._get_scope_context("c")
        # Function call is NOT what we cache — we cache the function reference.
        # Verify by confirming the fn was called once per sender (3x):
        assert call_count["imports"] == 3
        # And the cached reference matches:
        assert loop._scope_context_fn is provider_fn

    def test_empty_provider_output_becomes_none(self, tmp_path, monkeypatch):
        _install_mock_module(monkeypatch, "empty_mod", "fn", lambda s: "")
        loop = _mk_loop(tmp_path, provider="empty_mod:fn")
        assert loop._get_scope_context("anyone") is None

    def test_whitespace_provider_output_becomes_none(self, tmp_path, monkeypatch):
        _install_mock_module(monkeypatch, "ws_mod", "fn", lambda s: "   \n  ")
        loop = _mk_loop(tmp_path, provider="ws_mod:fn")
        assert loop._get_scope_context("anyone") is None

    def test_non_string_output_becomes_none(self, tmp_path, monkeypatch):
        """Fail-safe against a provider that returns None or a dict by mistake."""
        _install_mock_module(monkeypatch, "nonstr_mod", "fn", lambda s: {"x": 1})
        loop = _mk_loop(tmp_path, provider="nonstr_mod:fn")
        assert loop._get_scope_context("anyone") is None


# ---------------------------------------------------------------------------
# Provider misconfigured — graceful degradation
# ---------------------------------------------------------------------------

class TestProviderMisconfigured:
    def test_invalid_format_disables_provider(self, tmp_path, caplog):
        loop = _mk_loop(tmp_path, provider="no_colon_here")
        assert loop._get_scope_context("anyone") is None
        # Auto-disables so we don't re-log the same error per turn:
        assert loop._scope_context_provider == ""

    def test_missing_module_disables_provider(self, tmp_path):
        loop = _mk_loop(tmp_path, provider="totally_nonexistent_module_xyz:fn")
        assert loop._get_scope_context("anyone") is None
        assert loop._scope_context_provider == ""

    def test_missing_attribute_disables_provider(self, tmp_path, monkeypatch):
        # Register a module but without the named attribute
        _install_mock_module(monkeypatch, "has_mod_no_fn", "other_fn", lambda s: "x")
        loop = _mk_loop(tmp_path, provider="has_mod_no_fn:missing_fn")
        assert loop._get_scope_context("anyone") is None
        assert loop._scope_context_provider == ""


# ---------------------------------------------------------------------------
# Provider raises — agent must not crash
# ---------------------------------------------------------------------------

class TestProviderRaises:
    def test_exception_in_provider_returns_none(self, tmp_path, monkeypatch):
        def exploding(sender_id: str) -> str:
            raise RuntimeError("scope_store db locked")
        _install_mock_module(monkeypatch, "boom_mod", "fn", exploding)

        loop = _mk_loop(tmp_path, provider="boom_mod:fn")
        # Must not raise — agent path must survive provider failure
        assert loop._get_scope_context("anyone") is None
        # Provider stays configured (transient failures should retry next turn)
        assert loop._scope_context_provider == "boom_mod:fn"


# ---------------------------------------------------------------------------
# Homer-compatible contract: signature shape match
# ---------------------------------------------------------------------------

class TestHomerContract:
    """Lock down the exact call signature the homer side must satisfy."""

    def test_provider_called_with_single_positional_sender_id(self, tmp_path, monkeypatch):
        captured = {}

        def provider_fn(sender_id):  # positional, single arg
            captured["arg"] = sender_id
            captured["type"] = type(sender_id).__name__
            return "ok"
        _install_mock_module(monkeypatch, "contract_mod", "fn", provider_fn)

        loop = _mk_loop(tmp_path, provider="contract_mod:fn")
        loop._get_scope_context("16072348189@s.whatsapp.net")
        assert captured["arg"] == "16072348189@s.whatsapp.net"
        assert captured["type"] == "str"
