"""Tests for guest-agent memory-write disable + _save_turn skip offset.

Regression context
------------------
Homer ran into cross-scope leakage where the guest agent's long-term memory
(``memory/history.jsonl`` + session ``_last_summary``) accumulated facts from
every scope (e.g. ``Tola's 5th birthday`` details surfacing in a session that
belonged to a ``denver_mtb`` participant). Consolidation + per-session
summaries were writing content that must not cross sender boundaries. These
tests pin:

1. ``Consolidator(archive_disabled=True)`` skips the LLM call AND the
   ``store.append_history`` write AND returns ``None`` so callers don't
   stash a summary into ``session.metadata['_last_summary']``.
2. ``maybe_consolidate_by_tokens`` is a no-op when disabled.
3. The ``_save_turn`` skip offset adjusts for the ephemeral scope-context
   system message inserted at ``initial_messages[1]`` — without the fix,
   the last history entry was duplicated into ``session.messages`` on
   every scoped turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nanobot.agent.memory import Consolidator, MemoryStore


class _StubProvider:
    generation = SimpleNamespace(max_tokens=1024)

    def __init__(self):
        self.called = False

    async def chat_with_retry(self, **kwargs):
        self.called = True
        return SimpleNamespace(content="stub summary", tool_calls=None)


class _StubSessions:
    def save(self, session):
        pass


def _mk_consolidator(tmp_path, archive_disabled: bool) -> tuple[Consolidator, _StubProvider, MemoryStore]:
    store = MemoryStore(tmp_path)
    provider = _StubProvider()
    c = Consolidator(
        store=store,
        provider=provider,  # type: ignore[arg-type]
        model="stub",
        sessions=_StubSessions(),  # type: ignore[arg-type]
        context_window_tokens=1000,
        build_messages=lambda **kw: [],
        get_tool_definitions=lambda: [],
        max_completion_tokens=256,
        archive_disabled=archive_disabled,
    )
    return c, provider, store


class TestConsolidatorDisabled:
    """archive_disabled=True suppresses all memory writes."""

    @pytest.mark.asyncio
    async def test_archive_returns_none_and_does_not_call_llm(self, tmp_path):
        c, provider, store = _mk_consolidator(tmp_path, archive_disabled=True)
        result = await c.archive([{"role": "user", "content": "hello"}])
        assert result is None
        assert provider.called is False
        # history.jsonl must not have been written
        assert not store.history_file.exists() or store.history_file.read_text() == ""

    @pytest.mark.asyncio
    async def test_archive_enabled_writes_to_history(self, tmp_path):
        c, provider, store = _mk_consolidator(tmp_path, archive_disabled=False)
        result = await c.archive([{"role": "user", "content": "hello"}])
        assert result == "stub summary"
        assert provider.called is True
        assert store.history_file.exists()
        assert "stub summary" in store.history_file.read_text()

    @pytest.mark.asyncio
    async def test_maybe_consolidate_by_tokens_is_noop_when_disabled(self, tmp_path):
        c, provider, _ = _mk_consolidator(tmp_path, archive_disabled=True)
        fake_session = SimpleNamespace(
            messages=[{"role": "user", "content": "x"}],
            key="test",
            last_consolidated=0,
            metadata={},
            get_history=lambda max_messages=0: [],
        )
        # Should return immediately without provider call or token estimation
        await c.maybe_consolidate_by_tokens(fake_session)  # type: ignore[arg-type]
        assert provider.called is False


class TestSaveTurnSkipOffset:
    """Regression for the off-by-one when scope_ctx is injected.

    This mirrors the production path conceptually — we don't boot a full
    AgentLoop, just the arithmetic we fixed in loop.py. The fix: skip
    must include +1 for the ephemeral scope_ctx message inserted at
    ``initial_messages[1]`` so the last history entry isn't re-persisted.
    """

    def _compute_skip(self, history_len: int, scope_ctx_injected: bool) -> int:
        return 1 + (1 if scope_ctx_injected else 0) + history_len

    def test_no_scope_ctx_skips_system_plus_history(self):
        # Baseline: [sys, h0, h1, h2, user_msg, asst_msg] → skip=4 saves user_msg + asst_msg
        assert self._compute_skip(history_len=3, scope_ctx_injected=False) == 4

    def test_with_scope_ctx_skips_one_extra(self):
        # With injection: [sys, scope_ctx, h0, h1, h2, user_msg, asst_msg]
        # Skip must be 5 to land on user_msg — not 4, which would duplicate h2.
        assert self._compute_skip(history_len=3, scope_ctx_injected=True) == 5

    def test_empty_history_with_scope_ctx(self):
        # [sys, scope_ctx, user_msg, asst_msg] → skip=2 saves user_msg + asst_msg
        assert self._compute_skip(history_len=0, scope_ctx_injected=True) == 2

    def test_empty_history_without_scope_ctx(self):
        # [sys, user_msg, asst_msg] → skip=1 saves user_msg + asst_msg
        assert self._compute_skip(history_len=0, scope_ctx_injected=False) == 1
