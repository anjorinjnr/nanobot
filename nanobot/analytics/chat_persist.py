"""Per-turn persistence to homer's `hist_chat_messages` (family-history schema).

Mirrors the AnalyticsHook seam: AgentLoop._process_message calls
on_message_received() at the start of every non-synthetic turn and
on_response_sent(ctx, response_content=...) once the assistant reply is
built. Each pair becomes two rows in hist_chat_messages — role='user' and
role='assistant' — sharing the same contributor_id, written via Supabase
service-role REST.

Channel-agnostic: the portal's hourly extraction cron reads any unprocessed
row regardless of which channel originated the turn.

Failure-tolerant: every Supabase call is best-effort. A failed insert logs
WARN and is dropped. Chat persistence must never block the agent's reply.

Enable via env (all required when enabled):
    HOMER_CHAT_PERSIST_ENABLED=1
    SUPABASE_URL=...
    SUPABASE_SERVICE_KEY=...
    HOMER_HOUSEHOLD_ID=...

Channels supported today: whatsapp (lookup by hist_contributors.phone) and
email (lookup by hist_contributors.email). Other channels — telegram, slack,
etc — log a single WARN and are dropped; supporting them needs a join table
on the portal side.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Channel → hist_contributors lookup column. Channels not in this map are
# dropped at lookup time (logged once per (channel, sender) via the cache).
_CHANNEL_TO_COLUMN: dict[str, str] = {
    "whatsapp": "phone",
    "email": "email",
}


class ChatPersistHook:
    """Lazy, env-driven write-through hook for hist_chat_messages.

    Init is deferred to first call so import is free in environments where
    chat persistence isn't configured (CI, unit tests, OSS deploys).
    """

    def __init__(self) -> None:
        self._enabled = False
        self._initialized = False
        self._supabase_url = ""
        self._service_key = ""
        self._household_id = ""
        # Cache (channel, sender_id) -> contributor_id (or None on confirmed
        # miss). Process-lifetime, no TTL: contributor identity is stable.
        # An archived contributor remains cached as their id; the row stays
        # FK-valid, and inbound messages from archived contributors are
        # filtered upstream by the channel's allow_from list anyway.
        # No lock: AgentLoop._process_message is single-tasked per session,
        # and dict assignment is atomic in CPython.
        self._contrib_cache: dict[tuple[str, str], Optional[str]] = {}
        self._client: httpx.AsyncClient | None = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def _ensure_init(self) -> bool:
        if self._initialized:
            return self._enabled
        self._initialized = True
        flag = os.environ.get("HOMER_CHAT_PERSIST_ENABLED", "").strip().lower()
        if flag not in ("1", "true", "yes"):
            logger.debug("chat_persist disabled (HOMER_CHAT_PERSIST_ENABLED unset)")
            return False
        url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
        hh = os.environ.get("HOMER_HOUSEHOLD_ID", "").strip()
        if not url or not key or not hh:
            logger.warning(
                "chat_persist enabled but missing one of "
                "SUPABASE_URL / SUPABASE_SERVICE_KEY / HOMER_HOUSEHOLD_ID — disabling"
            )
            return False
        self._supabase_url = url
        self._service_key = key
        self._household_id = hh
        self._enabled = True
        logger.info("chat_persist initialized (household=%s)", hh)
        return True

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=f"{self._supabase_url}/rest/v1",
                headers={
                    "apikey": self._service_key,
                    "Authorization": f"Bearer {self._service_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying httpx client. Safe to call repeatedly."""
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    # ── public hooks (called from AgentLoop) ─────────────────────────────

    async def on_message_received(
        self,
        *,
        channel: str,
        sender_id: str,
        content: str,
        media: list[str],
        timestamp: datetime,
    ) -> dict[str, Any] | None:
        """Persist the inbound user turn; return ctx for on_response_sent.

        Returns None when persistence is disabled, the channel is unsupported,
        or the sender doesn't resolve to a contributor — callers should treat
        None as "skip on_response_sent for this turn".
        """
        if not self._ensure_init():
            return None
        contributor_id = await self._resolve_contributor(channel, sender_id)
        if contributor_id is None:
            return None
        text = content or ""
        if text or media:
            await self._insert(role="user", contributor_id=contributor_id, text=text)
        return {"contributor_id": contributor_id, "channel": channel}

    async def on_response_sent(
        self,
        ctx: dict[str, Any] | None,
        *,
        response_content: str | None,
        schedule_background: Any = None,
    ) -> None:
        """Persist the assistant reply.

        When `schedule_background` is provided (AgentLoop's `_schedule_background`),
        the insert is fired as a background task so the user-visible response
        isn't held up by Supabase round-trips. With strict ordering already
        guaranteed by the user-row insert that completed in
        `on_message_received`, the assistant write doesn't need to block the
        OutboundMessage return.
        """
        if ctx is None or not self._ensure_init():
            return
        if not (response_content or "").strip():
            return
        coro = self._insert(
            role="assistant",
            contributor_id=ctx["contributor_id"],
            text=response_content,
        )
        if schedule_background is not None:
            schedule_background(coro)
        else:
            await coro

    # ── internals ────────────────────────────────────────────────────────

    async def _resolve_contributor(
        self, channel: str, sender_id: str,
    ) -> Optional[str]:
        normalized = self._normalize_sender(channel, sender_id)
        cache_key = (channel, normalized)
        if cache_key in self._contrib_cache:
            return self._contrib_cache[cache_key]

        column = _CHANNEL_TO_COLUMN.get(channel)
        if not column:
            logger.warning(
                "chat_persist: channel not supported — channel=%s sender_id=%s "
                "(no hist_contributors lookup column; add a join table to extend)",
                channel, sender_id,
            )
            self._contrib_cache[cache_key] = None
            return None

        if not normalized:
            logger.warning(
                "chat_persist: empty normalized sender_id (channel=%s raw=%r)",
                channel, sender_id,
            )
            self._contrib_cache[cache_key] = None
            return None

        try:
            r = await self._http().get(
                "/hist_contributors",
                params={
                    "select": "id",
                    "household_id": f"eq.{self._household_id}",
                    column: f"eq.{normalized}",
                    "limit": "1",
                },
            )
            r.raise_for_status()
            rows = r.json()
        except Exception:
            # Don't cache transient failures — let the next message retry.
            logger.warning(
                "chat_persist: contributor lookup failed (channel=%s)",
                channel, exc_info=True,
            )
            return None

        cid: Optional[str] = rows[0]["id"] if rows else None
        if cid is None:
            logger.warning(
                "chat_persist: unknown sender — channel=%s sender_id=%s",
                channel, sender_id,
            )
        self._contrib_cache[cache_key] = cid
        return cid

    @staticmethod
    def _normalize_sender(channel: str, sender_id: str) -> str:
        """Normalize sender_id to match the shape stored on hist_contributors.

        - whatsapp: digits only (matches `tools/history_invite.py:_normalise_phone`,
          which stores E.164 without leading "+").
        - email: lowercased, stripped.
        - others: stripped (channel will be rejected at column lookup).
        """
        if sender_id is None:
            return ""
        if channel == "whatsapp":
            return "".join(c for c in sender_id if c.isdigit())
        if channel == "email":
            return sender_id.strip().lower()
        return sender_id.strip()

    async def _insert(
        self, *, role: str, contributor_id: str, text: str,
    ) -> None:
        try:
            r = await self._http().post(
                "/hist_chat_messages",
                json={
                    "household_id": self._household_id,
                    "contributor_id": contributor_id,
                    "role": role,
                    "text": text,
                },
            )
            r.raise_for_status()
        except Exception:
            logger.warning(
                "chat_persist: insert failed (role=%s contributor=%s)",
                role, contributor_id, exc_info=True,
            )


# ── singleton accessor ───────────────────────────────────────────────────

_HOOK: ChatPersistHook | None = None


def get_chat_persist_hook() -> ChatPersistHook:
    """Return the process-wide ChatPersistHook (lazy)."""
    global _HOOK
    if _HOOK is None:
        _HOOK = ChatPersistHook()
    return _HOOK
