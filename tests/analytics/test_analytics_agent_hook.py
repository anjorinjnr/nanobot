"""Tests for the AnalyticsAgentHook adapter.

The underlying AnalyticsHook impl is covered by test_hook.py.
These tests verify only the AgentHook lifecycle bridge — that before_turn /
after_turn route to the impl with the right args for both the user-turn
and synthetic-turn paths, and that errors are swallowed.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.hook import TurnMetadata
from nanobot.agent.runner import STOP_EMPTY_FINAL, STOP_INTENTIONAL_SILENCE
from nanobot.analytics.analytics_agent_hook import (
    AnalyticsAgentHook,
    _STATE_KEY,
    _escalation_triggered,
)


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
        is_guest=False,
    )
    defaults.update(overrides)
    return TurnMetadata(**defaults)


# ── before_turn: user path ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_before_turn_user_routes_to_impl_and_stashes_ctx() -> None:
    impl = MagicMock()
    impl.on_message_received = MagicMock(return_value={"turn_id": "t-1"})
    hook = AnalyticsAgentHook(impl=impl)

    turn = _turn(is_guest=True)
    await hook.before_turn(turn)

    impl.on_message_received.assert_called_once()
    kwargs = impl.on_message_received.call_args.kwargs
    assert kwargs["channel"] == "whatsapp"
    assert kwargs["sender_id"] == "14125551234"
    assert kwargs["content"] == "hello"
    assert kwargs["is_guest"] is True
    assert turn.state[_STATE_KEY] == {"turn_id": "t-1"}


@pytest.mark.asyncio
async def test_before_turn_synthetic_skipped() -> None:
    impl = MagicMock()
    impl.on_message_received = MagicMock()
    hook = AnalyticsAgentHook(impl=impl)

    await hook.before_turn(_turn(is_synthetic=True, trigger_kind="heartbeat"))

    impl.on_message_received.assert_not_called()


@pytest.mark.asyncio
async def test_before_turn_swallows_errors() -> None:
    impl = MagicMock()
    impl.on_message_received = MagicMock(side_effect=RuntimeError("posthog down"))
    hook = AnalyticsAgentHook(impl=impl)

    turn = _turn()
    await hook.before_turn(turn)  # must not raise

    assert _STATE_KEY not in turn.state


# ── after_turn: user path ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_after_turn_user_routes_to_impl_with_tools_and_escalation() -> None:
    impl = MagicMock()
    impl.on_response_sent = AsyncMock()
    sched = MagicMock()
    hook = AnalyticsAgentHook(impl=impl, schedule_background=sched)

    turn = _turn()
    turn.state[_STATE_KEY] = {"turn_id": "t-1"}
    turn.response_content = "reply"
    turn.tools_used = ["web_search", "escalate"]
    await hook.after_turn(turn)

    impl.on_response_sent.assert_awaited_once()
    args, kwargs = impl.on_response_sent.await_args
    assert args[0] == {"turn_id": "t-1"}
    assert kwargs["response_content"] == "reply"
    assert kwargs["tools_used"] == ["web_search", "escalate"]
    assert kwargs["escalation_triggered"] is True
    assert kwargs["schedule_background"] is sched


@pytest.mark.asyncio
async def test_after_turn_user_no_escalation_when_neutral_tools() -> None:
    impl = MagicMock()
    impl.on_response_sent = AsyncMock()
    hook = AnalyticsAgentHook(impl=impl)

    turn = _turn()
    turn.state[_STATE_KEY] = {"turn_id": "t-1"}
    turn.tools_used = ["web_search", "memory_write"]
    await hook.after_turn(turn)

    kwargs = impl.on_response_sent.await_args.kwargs
    assert kwargs["escalation_triggered"] is False


@pytest.mark.asyncio
async def test_after_turn_user_noop_when_no_ctx() -> None:
    """If before_turn didn't stash (impl disabled), after_turn skips."""
    impl = MagicMock()
    impl.on_response_sent = AsyncMock()
    hook = AnalyticsAgentHook(impl=impl)

    await hook.after_turn(_turn())  # no state key

    impl.on_response_sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_after_turn_user_swallows_errors() -> None:
    impl = MagicMock()
    impl.on_response_sent = AsyncMock(side_effect=RuntimeError("posthog down"))
    hook = AnalyticsAgentHook(impl=impl)

    turn = _turn()
    turn.state[_STATE_KEY] = {"turn_id": "t-1"}
    turn.response_content = "x"
    await hook.after_turn(turn)  # must not raise


# ── after_turn: synthetic path ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_after_turn_synthetic_fires_agent_initiated_action() -> None:
    impl = MagicMock()
    impl.track_agent_initiated_action = MagicMock()
    hook = AnalyticsAgentHook(impl=impl)

    turn = _turn(is_synthetic=True, trigger_kind="heartbeat")
    turn.started_at_monotonic = time.monotonic() - 0.5
    turn.response_content = "tick"
    turn.tools_used = ["message"]
    turn.stop_reason = "stop"
    await hook.after_turn(turn)

    impl.track_agent_initiated_action.assert_called_once()
    kwargs = impl.track_agent_initiated_action.call_args.kwargs
    assert kwargs["trigger_kind"] == "heartbeat"
    assert kwargs["response_content"] == "tick"
    assert kwargs["tools_used"] == ["message"]
    assert kwargs["latency_ms"] >= 400  # ~500ms elapsed


@pytest.mark.asyncio
async def test_after_turn_synthetic_drops_empty_final_placeholder() -> None:
    """STOP_EMPTY_FINAL response_content shouldn't be reported — had_outbound
    is the dashboard signal and the placeholder string would lie."""
    impl = MagicMock()
    impl.track_agent_initiated_action = MagicMock()
    hook = AnalyticsAgentHook(impl=impl)

    turn = _turn(is_synthetic=True, trigger_kind="heartbeat")
    turn.response_content = "placeholder"
    turn.stop_reason = STOP_EMPTY_FINAL
    await hook.after_turn(turn)

    kwargs = impl.track_agent_initiated_action.call_args.kwargs
    assert kwargs["response_content"] is None


@pytest.mark.asyncio
async def test_after_turn_synthetic_drops_intentional_silence() -> None:
    impl = MagicMock()
    impl.track_agent_initiated_action = MagicMock()
    hook = AnalyticsAgentHook(impl=impl)

    turn = _turn(is_synthetic=True, trigger_kind="cron")
    turn.response_content = "..."
    turn.stop_reason = STOP_INTENTIONAL_SILENCE
    await hook.after_turn(turn)

    kwargs = impl.track_agent_initiated_action.call_args.kwargs
    assert kwargs["response_content"] is None


@pytest.mark.asyncio
async def test_after_turn_synthetic_defaults_trigger_kind() -> None:
    impl = MagicMock()
    impl.track_agent_initiated_action = MagicMock()
    hook = AnalyticsAgentHook(impl=impl)

    turn = _turn(is_synthetic=True)  # trigger_kind=None
    await hook.after_turn(turn)

    assert impl.track_agent_initiated_action.call_args.kwargs["trigger_kind"] == "synthetic"


@pytest.mark.asyncio
async def test_after_turn_synthetic_swallows_errors() -> None:
    impl = MagicMock()
    impl.track_agent_initiated_action = MagicMock(side_effect=RuntimeError("boom"))
    hook = AnalyticsAgentHook(impl=impl)

    await hook.after_turn(_turn(is_synthetic=True))  # must not raise


# ── escalation helper ────────────────────────────────────────────────────


def test_escalation_helper() -> None:
    assert _escalation_triggered(["escalate"]) is True
    assert _escalation_triggered(["resolve_escalation"]) is True
    assert _escalation_triggered(["escalate", "web_search"]) is True
    assert _escalation_triggered(["web_search"]) is False
    assert _escalation_triggered([]) is False
