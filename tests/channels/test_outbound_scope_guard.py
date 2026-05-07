"""Tests for the outbound scope guard wired into ChannelManager._send_once."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Callable
from unittest.mock import AsyncMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels import scope_guard
from nanobot.channels.manager import ChannelManager, _check_outbound_authorized
from nanobot.channels.scope_guard import (
    OutboundScopeError,
    ScopeLookupResult,
    set_scope_lookup,
)


class _RecordingChannel(BaseChannel):
    name = "whatsapp"

    def __init__(self) -> None:
        super().__init__({"allow_from": ["*"]}, MessageBus())
        self.sent: list[OutboundMessage] = []
        self.deltas: list[tuple[str, str]] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, msg: OutboundMessage) -> None:
        self.sent.append(msg)

    async def send_delta(self, chat_id: str, delta: str, metadata=None) -> None:
        self.deltas.append((chat_id, delta))


@pytest.fixture(autouse=True)
def _reset_lookup():
    """Each test starts with no installed lookup. Clean up after."""
    set_scope_lookup(None)
    yield
    set_scope_lookup(None)


def _install(fn: Callable[[str, str], ScopeLookupResult]) -> None:
    set_scope_lookup(fn)


def _msg(chat_id: str = "14126920720@s.whatsapp.net", **meta) -> OutboundMessage:
    return OutboundMessage(channel="whatsapp", chat_id=chat_id, content="hi", metadata=meta)


# --- _check_outbound_authorized direct tests ----------------------------------

def test_no_lookup_installed_allows_send() -> None:
    channel = _RecordingChannel()
    # No exception, no refusal — vanilla nanobot behavior preserved.
    _check_outbound_authorized(channel, _msg())


def test_household_member_allowed() -> None:
    _install(lambda ch, cid: ScopeLookupResult(authorized=True, reason="household_member"))
    channel = _RecordingChannel()
    _check_outbound_authorized(channel, _msg())


def test_active_scope_with_context_allowed() -> None:
    _install(lambda ch, cid: ScopeLookupResult(
        authorized=True, reason="active_scope", scope_ids=["s1"]
    ))
    channel = _RecordingChannel()
    _check_outbound_authorized(channel, _msg())


def test_no_scope_refused_with_remediation() -> None:
    _install(lambda ch, cid: ScopeLookupResult(
        authorized=False,
        reason="no_scope",
        remediation="Run manage_interaction --create --purpose '...'",
    ))
    channel = _RecordingChannel()
    with pytest.raises(OutboundScopeError) as exc_info:
        _check_outbound_authorized(channel, _msg())
    assert exc_info.value.reason == "no_scope"
    assert "manage_interaction" in str(exc_info.value)


def test_scope_without_context_refused() -> None:
    _install(lambda ch, cid: ScopeLookupResult(
        authorized=False, reason="scope_no_context"
    ))
    channel = _RecordingChannel()
    with pytest.raises(OutboundScopeError) as exc_info:
        _check_outbound_authorized(channel, _msg())
    assert exc_info.value.reason == "scope_no_context"


def test_no_reply_scope_outbound_allowed() -> None:
    _install(lambda ch, cid: ScopeLookupResult(
        authorized=True, reason="no_reply_scope", suppress_inbound=True,
    ))
    channel = _RecordingChannel()
    _check_outbound_authorized(channel, _msg())


def test_stream_continuation_skips_re_check() -> None:
    """Stream deltas/end-frames inherit the initial decision — don't re-check."""
    refused = lambda ch, cid: ScopeLookupResult(authorized=False, reason="no_scope")
    _install(refused)
    channel = _RecordingChannel()
    _check_outbound_authorized(channel, _msg(_stream_delta=True))
    _check_outbound_authorized(channel, _msg(_stream_end=True))


def test_empty_chat_id_skips_check() -> None:
    """Broadcasts / system events without a chat_id can't be scope-checked."""
    refused = lambda ch, cid: ScopeLookupResult(authorized=False, reason="no_scope")
    _install(refused)
    channel = _RecordingChannel()
    msg = OutboundMessage(channel="whatsapp", chat_id="", content="", metadata={})
    _check_outbound_authorized(channel, msg)  # no exception


def test_lookup_exception_fails_open() -> None:
    def boom(ch, cid):
        raise RuntimeError("scope_store crashed")
    _install(boom)
    channel = _RecordingChannel()
    _check_outbound_authorized(channel, _msg())  # no exception — fails open


# --- inbound suppression ------------------------------------------------------

def test_inbound_suppression_when_no_lookup() -> None:
    assert scope_guard.check_inbound_suppressed("whatsapp", "anyone") is False


def test_inbound_suppression_two_way_scope_not_suppressed() -> None:
    _install(lambda ch, cid: ScopeLookupResult(
        authorized=True, reason="active_scope", suppress_inbound=False,
    ))
    assert scope_guard.check_inbound_suppressed("whatsapp", "14126920720") is False


def test_inbound_suppression_no_reply_scope_suppressed() -> None:
    _install(lambda ch, cid: ScopeLookupResult(
        authorized=True, reason="no_reply_scope", suppress_inbound=True,
    ))
    assert scope_guard.check_inbound_suppressed("whatsapp", "14126920720") is True


def test_inbound_suppression_lookup_error_fails_open() -> None:
    """A broken lookup must not silently drop inbound traffic."""
    def boom(ch, cid):
        raise RuntimeError("boom")
    _install(boom)
    assert scope_guard.check_inbound_suppressed("whatsapp", "x") is False


# --- _send_with_retry behavior -----------------------------------------------

@pytest.mark.asyncio
async def test_outbound_scope_error_is_terminal_no_retry() -> None:
    """Refused sends must not exhaust the retry budget — the rejection is permanent."""
    _install(lambda ch, cid: ScopeLookupResult(
        authorized=False, reason="no_scope", remediation="create a scope first",
    ))

    # Bypass __init__ — we only need _send_with_retry's behavior, not the full
    # channel manager bootstrap.
    mgr = ChannelManager.__new__(ChannelManager)
    mgr.config = SimpleNamespace(
        channels=SimpleNamespace(
            send_max_retries=3,
            spam_guard=SimpleNamespace(window_s=60, max_repeats=10),
        ),
    )
    channel = _RecordingChannel()

    msg = _msg()
    loop = asyncio.get_running_loop()
    msg._delivery_future = loop.create_future()

    await mgr._send_with_retry(channel, msg)

    # The future captured the OutboundScopeError — the agent's MessageTool will
    # see the structured remediation when it awaits delivery.
    assert msg._delivery_future.done()
    exc = msg._delivery_future.exception()
    assert isinstance(exc, OutboundScopeError)
    assert exc.reason == "no_scope"
    assert channel.sent == []
