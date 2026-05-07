"""Outbound scope guard for channel sends.

Vanilla nanobot ships with a no-op default: every outbound is allowed.
A host application (homer) installs a lookup callable via ``set_scope_lookup``
to enforce that every outbound either targets a known household member or
maps to an active scope-with-context. Replies from no-reply scopes can be
suppressed on inbound via ``check_inbound_suppressed``.

Design: nanobot does not depend on the host's scope store; the host injects
a callable so this stays a clean overlay. See homer's
docs/features/outbound_scope_enforcement_plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from loguru import logger


@dataclass
class ScopeLookupResult:
    """Result of resolving (channel, participant_id) → authorization.

    ``reason`` is one of:
      household_member  — recipient is a household member; bypass.
      active_scope      — active scope with non-empty context exists.
      no_reply_scope    — active scope with mode='no_reply'; outbound allowed,
                          inbound suppressed.
      no_scope          — no scope found.
      scope_no_context  — scope exists but has no purpose / event envelope.
      scope_inactive    — scope exists but status != 'active'.
      lookup_error      — host lookup raised; failed open.
    """

    authorized: bool
    reason: str
    scope_ids: list[str] = field(default_factory=list)
    remediation: str | None = None
    suppress_inbound: bool = False


ScopeLookup = Callable[[str, str], ScopeLookupResult]


_lookup: ScopeLookup | None = None


def set_scope_lookup(fn: ScopeLookup | None) -> None:
    """Install the host's lookup callable. Pass ``None`` to clear."""
    global _lookup
    _lookup = fn


def get_scope_lookup() -> ScopeLookup | None:
    return _lookup


class OutboundScopeError(Exception):
    """Send refused because the recipient has no active scope-with-context."""

    def __init__(self, channel: str, chat_id: str, result: ScopeLookupResult):
        self.channel = channel
        self.chat_id = chat_id
        self.reason = result.reason
        self.scope_ids = list(result.scope_ids)
        self.remediation = result.remediation
        super().__init__(self._format())

    def _format(self) -> str:
        msg = f"Send refused on {self.channel} to {self.chat_id}: {self.reason}."
        if self.remediation:
            msg = f"{msg}\n{self.remediation}"
        return msg


def check_outbound(channel: str, chat_id: str) -> ScopeLookupResult | None:
    """Run the lookup if one is installed.

    Returns ``None`` when no lookup is configured (vanilla nanobot — allow).
    Lookup exceptions fail open with a logged warning so a broken host
    integration does not silently kill outbound traffic.
    """
    fn = _lookup
    if fn is None:
        return None
    try:
        return fn(channel, chat_id)
    except Exception as e:
        logger.warning(
            "scope_guard: lookup failed for {}:{} ({}: {}); allowing send",
            channel, chat_id, type(e).__name__, e,
        )
        return ScopeLookupResult(
            authorized=True, reason="lookup_error"
        )


def check_inbound_suppressed(channel: str, sender_id: str) -> bool:
    """Return True if inbound from this sender should be dropped (no-reply scope)."""
    fn = _lookup
    if fn is None:
        return False
    try:
        result = fn(channel, sender_id)
    except Exception:
        return False
    return result.authorized and result.suppress_inbound
