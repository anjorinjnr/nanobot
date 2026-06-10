"""Tests for the ChatPersistAgentHook adapter.

The underlying ChatPersistHook impl is covered by test_chat_persist.py.
These tests verify only the AgentHook lifecycle bridge — that before_turn /
after_turn call into the impl with the right args, that synthetic turns are
skipped, and that errors are swallowed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.hook import TurnMetadata
from nanobot.analytics.chat_persist_hook import ChatPersistAgentHook, _STATE_KEY


def _turn(**overrides) -> TurnMetadata:
    defaults = dict(
        channel="whatsapp",
        sender_id="14125551234",
        chat_id="14125551234",
        content="hello",
        media=[],
        timestamp=datetime.now(timezone.utc),
        is_synthetic=False,
        message_id="msg-1",
    )
    defaults.update(overrides)
    return TurnMetadata(**defaults)


@pytest.mark.asyncio
async def test_before_turn_routes_to_impl_and_stashes_ctx() -> None:
    impl = MagicMock()
    impl.on_message_received = AsyncMock(return_value={"contributor_id": "c-1", "channel": "whatsapp"})
    sched = MagicMock()
    hook = ChatPersistAgentHook(impl=impl, schedule_background=sched)

    turn = _turn()
    await hook.before_turn(turn)

    impl.on_message_received.assert_awaited_once()
    kwargs = impl.on_message_received.await_args.kwargs
    assert kwargs["channel"] == "whatsapp"
    assert kwargs["sender_id"] == "14125551234"
    assert kwargs["content"] == "hello"
    assert kwargs["media"] == []
    assert kwargs["schedule_background"] is sched
    assert turn.state[_STATE_KEY] == {"contributor_id": "c-1", "channel": "whatsapp"}


@pytest.mark.asyncio
async def test_before_turn_skips_synthetic() -> None:
    impl = MagicMock()
    impl.on_message_received = AsyncMock()
    hook = ChatPersistAgentHook(impl=impl)

    await hook.before_turn(_turn(is_synthetic=True))

    impl.on_message_received.assert_not_awaited()


@pytest.mark.asyncio
async def test_before_turn_swallows_errors() -> None:
    impl = MagicMock()
    impl.on_message_received = AsyncMock(side_effect=RuntimeError("supabase down"))
    hook = ChatPersistAgentHook(impl=impl)

    turn = _turn()
    await hook.before_turn(turn)  # must not raise

    assert _STATE_KEY not in turn.state


@pytest.mark.asyncio
async def test_before_turn_no_stash_when_impl_returns_none() -> None:
    """Channel unsupported / sender unresolved → impl returns None → no ctx."""
    impl = MagicMock()
    impl.on_message_received = AsyncMock(return_value=None)
    hook = ChatPersistAgentHook(impl=impl)

    turn = _turn(channel="telegram")
    await hook.before_turn(turn)

    assert _STATE_KEY not in turn.state


@pytest.mark.asyncio
async def test_after_turn_pairs_with_stashed_ctx() -> None:
    impl = MagicMock()
    impl.on_response_sent = AsyncMock()
    sched = MagicMock()
    hook = ChatPersistAgentHook(impl=impl, schedule_background=sched)

    turn = _turn()
    turn.state[_STATE_KEY] = {"contributor_id": "c-1", "channel": "whatsapp"}
    turn.response_content = "hi back"
    await hook.after_turn(turn)

    impl.on_response_sent.assert_awaited_once()
    args, kwargs = impl.on_response_sent.await_args
    assert args[0] == {"contributor_id": "c-1", "channel": "whatsapp"}
    assert kwargs["response_content"] == "hi back"
    assert kwargs["schedule_background"] is sched


@pytest.mark.asyncio
async def test_after_turn_noop_when_no_ctx() -> None:
    """When before_turn didn't stash (synthetic, unsupported channel), skip."""
    impl = MagicMock()
    impl.on_response_sent = AsyncMock()
    hook = ChatPersistAgentHook(impl=impl)

    await hook.after_turn(_turn())  # no state key

    impl.on_response_sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_after_turn_swallows_errors() -> None:
    impl = MagicMock()
    impl.on_response_sent = AsyncMock(side_effect=RuntimeError("supabase down"))
    hook = ChatPersistAgentHook(impl=impl)

    turn = _turn()
    turn.state[_STATE_KEY] = {"contributor_id": "c-1", "channel": "whatsapp"}
    turn.response_content = "x"
    await hook.after_turn(turn)  # must not raise
