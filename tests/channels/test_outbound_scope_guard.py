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


def test_stream_events_are_checked_too() -> None:
    """Stream deltas + end-frames must be checked, not skipped.

    Earlier behaviour skipped stream continuations on the assumption that
    an "initial send" had already been authorized. For streaming-enabled
    channels (e.g. WhatsApp), the actual content arrives in delta chunks
    and there is no separate non-streaming initial — the skip silently
    let the entire reply through while only the cosmetic _streamed
    finalizer was refused. Regression coverage for the 2026-05-07 incident
    where Adam's reply via LID JID was 5x-warning'd but still delivered.
    """
    refused = lambda ch, cid: ScopeLookupResult(authorized=False, reason="no_scope")
    _install(refused)
    channel = _RecordingChannel()
    with pytest.raises(OutboundScopeError):
        _check_outbound_authorized(channel, _msg(_stream_delta=True))
    with pytest.raises(OutboundScopeError):
        _check_outbound_authorized(channel, _msg(_stream_end=True))
    with pytest.raises(OutboundScopeError):
        _check_outbound_authorized(channel, _msg(_streamed=True))


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


# --- is_allowed boundary: outbound scope lookup must NOT gate inbound --------
#
# Revert of PR #81 (`feat(channels): Phase 2 — converge inbound is_allowed
# onto the scope lookup`). #81 made any sender with an active scope pass
# is_allowed regardless of static allow_from. That collapsed two independent
# ACLs into one and, in deployments that run a privileged "main" agent + a
# scoped "guest" agent on the same channel account (homer's split-agent
# architecture), let guests reach the main agent's privileged context.
# Concrete incident: 2026-05-27 — a guest's active trip scope authorized
# them at the main agent's inbound, and a separate heartbeat-dispatch bug
# leaked kid-related email content to them. The fundamental error in #81
# was using the *outbound* scope lookup as the *inbound* ACL; outbound
# authorization is per-recipient consent, inbound authorization is per-
# process trust, and they need to stay separate. These tests lock the
# revert in so the "additive convergence" idea can't slide back unnoticed.

def test_is_allowed_does_not_consult_outbound_scope_lookup() -> None:
    """A scope-authorized sender absent from allow_from must NOT pass is_allowed.

    Regression for the 2026-05-27 incident — the outbound scope lookup
    governs sends, not inbound. Don't let a host's scope ACL silently
    open a privileged process's inbound surface.
    """
    _install(lambda ch, sid: ScopeLookupResult(authorized=True, reason="active_scope"))
    channel = _RecordingChannel()
    channel.config = {"allow_from": []}
    assert channel.is_allowed("14129739891@s.whatsapp.net") is False


def test_is_allowed_static_allow_from_when_no_lookup() -> None:
    """No lookup installed → behave exactly like the static check."""
    channel = _RecordingChannel()
    channel.config = {"allow_from": ["alice"]}
    assert channel.is_allowed("alice") is True
    assert channel.is_allowed("eve") is False


def test_is_allowed_static_allow_from_even_when_lookup_present() -> None:
    """Static allow_from is authoritative; lookup says yes-or-no doesn't matter."""
    _install(lambda ch, sid: ScopeLookupResult(authorized=True, reason="active_scope"))
    channel = _RecordingChannel()
    channel.config = {"allow_from": ["alice"]}
    assert channel.is_allowed("alice") is True
    assert channel.is_allowed("eve") is False  # lookup says yes; still rejected


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
