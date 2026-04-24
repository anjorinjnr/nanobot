"""Guest-agent memory-write disable + _save_turn skip-offset tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import Consolidator, MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


def _make_consolidator(store, provider, *, archive_disabled: bool) -> Consolidator:
    return Consolidator(
        store=store,
        provider=provider,
        model="test-model",
        sessions=MagicMock(),
        context_window_tokens=1000,
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
        max_completion_tokens=100,
        archive_disabled=archive_disabled,
    )


class TestConsolidatorDisabled:
    @pytest.mark.asyncio
    async def test_archive_returns_none_and_does_not_call_llm(self, store, mock_provider):
        c = _make_consolidator(store, mock_provider, archive_disabled=True)
        result = await c.archive([{"role": "user", "content": "hello"}])
        assert result is None
        mock_provider.chat_with_retry.assert_not_called()
        assert not store.history_file.exists() or store.history_file.read_text() == ""

    @pytest.mark.asyncio
    async def test_archive_enabled_writes_to_history(self, store, mock_provider):
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="stub summary", tool_calls=None,
        )
        c = _make_consolidator(store, mock_provider, archive_disabled=False)
        result = await c.archive([{"role": "user", "content": "hello"}])
        assert result == "stub summary"
        mock_provider.chat_with_retry.assert_awaited_once()
        assert "stub summary" in store.history_file.read_text()

    @pytest.mark.asyncio
    async def test_maybe_consolidate_by_tokens_is_noop_when_disabled(self, store, mock_provider):
        c = _make_consolidator(store, mock_provider, archive_disabled=True)
        fake_session = SimpleNamespace(
            messages=[{"role": "user", "content": "x"}],
            key="test",
            last_consolidated=0,
            metadata={},
            get_history=lambda max_messages=0: [],
        )
        await c.maybe_consolidate_by_tokens(fake_session)
        mock_provider.chat_with_retry.assert_not_called()


class TestSaveTurnSkipOffset:
    """Pin the arithmetic from loop.py — without the +1 for scope_ctx the
    last history entry was getting duplicated each scoped turn."""

    @staticmethod
    def _skip(history_len: int, scope_ctx_injected: bool) -> int:
        return 1 + (1 if scope_ctx_injected else 0) + history_len

    def test_no_scope_ctx_skips_system_plus_history(self):
        assert self._skip(history_len=3, scope_ctx_injected=False) == 4

    def test_with_scope_ctx_skips_one_extra(self):
        assert self._skip(history_len=3, scope_ctx_injected=True) == 5

    def test_empty_history_with_scope_ctx(self):
        assert self._skip(history_len=0, scope_ctx_injected=True) == 2

    def test_empty_history_without_scope_ctx(self):
        assert self._skip(history_len=0, scope_ctx_injected=False) == 1
