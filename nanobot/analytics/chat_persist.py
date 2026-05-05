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
import mimetypes
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Channel → hist_contributors lookup column. Channels not in this map are
# dropped at lookup time (logged once per (channel, sender) via the cache).
_CHANNEL_TO_COLUMN: dict[str, str] = {
    "whatsapp": "phone",
    "email": "email",
}

# Media bigger than this gets logged + skipped rather than uploaded. WhatsApp
# voice notes can be multi-MB; the portal cap is generous so contributors
# don't lose context. Above this is almost certainly a misconfiguration.
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Curator bootstrap role constants.
_CURATOR_ROLE = "curator"
_ADMIN_ROLE = "admin"


def _kind_from_mime(mime: str | None) -> str | None:
    """Map a mime type to the `pending_upload.kind` enum (image/audio/video).

    Returns None for unknown / non-media — the caller skips the upload.
    """
    if not mime:
        return None
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    return None


def _safe_filename(name: str) -> str:
    """Sanitize a filename to match the portal's _STORAGE_PATH_RE charset."""
    cleaned = "".join(c if c.isalnum() or c in ".-_" else "_" for c in name)
    return cleaned[:200] or "file"


def _build_storage_path(filename: str) -> str:
    """`YYYY/MM/<12-hex>-<safe-filename>` — matches portal create_signed_upload_url."""
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y/%m')}/{secrets.token_hex(6)}-{_safe_filename(filename)}"


def _bucket_for(household_id: str) -> str:
    return f"history-media-{household_id}"


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
        # Separate client for /storage/v1: different base path, different
        # default timeout (uploads run minutes for big voice notes; the REST
        # client's 10s would clobber them).
        self._storage_client: httpx.AsyncClient | None = None

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

    def _storage_http(self) -> httpx.AsyncClient:
        if self._storage_client is None:
            self._storage_client = httpx.AsyncClient(
                base_url=f"{self._supabase_url}/storage/v1",
                headers=self._service_role_headers(),
                timeout=60.0,
            )
        return self._storage_client

    def _service_role_headers(self) -> dict[str, str]:
        return {
            "apikey": self._service_key,
            "Authorization": f"Bearer {self._service_key}",
        }

    async def aclose(self) -> None:
        """Close underlying httpx clients. Safe to call repeatedly."""
        for attr in ("_client", "_storage_client"):
            client = getattr(self, attr)
            if client is not None:
                try:
                    await client.aclose()
                finally:
                    setattr(self, attr, None)

    # ── public hooks (called from AgentLoop) ─────────────────────────────

    async def on_message_received(
        self,
        *,
        channel: str,
        sender_id: str,
        content: str,
        media: list[str],
        timestamp: datetime,
        schedule_background: Any = None,
    ) -> dict[str, Any] | None:
        """Persist the inbound user turn; return ctx for on_response_sent.

        Behavior:

        - Inserts the text portion of the turn as a single user row immediately
          (with ``pending_upload=None``). The agent's reply is never gated on
          a media upload — even multi-MB voice notes don't delay the response.
        - For each media item attached to the turn, schedules an independent
          background task that uploads the file to Supabase storage and then
          inserts a separate user row carrying the resulting ``pending_upload``
          metadata. Multi-attachment turns produce one chat row per file —
          no media is dropped, ever.

        When ``schedule_background`` is provided (AgentLoop's
        ``_schedule_background``), media uploads are dispatched onto the
        background-task pool and ``on_message_received`` returns as soon as
        the text row is written. When omitted (tests, OSS deploys with no
        loop wiring), uploads run sequentially via ``await`` so callers can
        rely on completion-by-return.

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
        media_paths = [p for p in (media or []) if isinstance(p, str) and p]

        # Insert the text row first so the chat timeline preserves the
        # contributor's actual message before any media-upload side effects
        # land. ``pending_upload`` rides on its own row (one per file)
        # written from the background task — keeps the timeline correct even
        # under multi-attachment turns.
        if text or media_paths:
            await self._insert(
                role="user",
                contributor_id=contributor_id,
                text=text,
                pending_upload=None,
            )

        for path in media_paths:
            coro = self._upload_and_insert_media(contributor_id, path)
            if schedule_background is not None:
                schedule_background(coro)
            else:
                await coro

        return {"contributor_id": contributor_id, "channel": channel}

    async def _upload_and_insert_media(
        self, contributor_id: str, path: str,
    ) -> None:
        """Upload one media file to storage; on success, insert a user row
        carrying the resulting `pending_upload` metadata.

        Failures (file missing, unsupported mime, oversize, upload error) are
        already logged inside `_upload_one_media` and result in `None` —
        when that happens we drop this media item without inserting a row.
        The text row from `on_message_received` already captured the textual
        content of the turn, so a failed media upload doesn't lose the
        contributor's words.
        """
        pending = await self._upload_one_media(path)
        if pending is None:
            return
        await self._insert(
            role="user",
            contributor_id=contributor_id,
            text="",
            pending_upload=pending,
        )

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
        if cid is None and channel == "whatsapp":
            # Bootstrap path: if the inbound matches the household's admin
            # phone, ensure the curator's hist_contributors row exists with
            # phone populated. Curator rows created by portal signup have
            # phone=null, so curator-on-WhatsApp wouldn't otherwise resolve.
            cid = await self._bootstrap_curator_if_admin(normalized)
        if cid is None:
            logger.warning(
                "chat_persist: unknown sender — channel=%s sender_id=%s",
                channel, sender_id,
            )
        self._contrib_cache[cache_key] = cid
        return cid

    async def _bootstrap_curator_if_admin(
        self, sender_phone: str,
    ) -> Optional[str]:
        """Idempotent curator-row upsert when sender matches HOMER_ADMIN_PHONE.

        Three branches:
        - Existing curator row with phone matching → return its id (no-op
          beyond returning).
        - Existing curator row with phone null → PATCH phone, return id.
        - Existing curator row with a different phone → log warning and
          decline to override (defends against the rare co-curator case).
        - No curator row → look up the admin from household_members, INSERT
          a new curator row mirroring the shape `ensure_contributor_for_user`
          Path 2 produces (auth_user_id from members.user_id, display_name
          from members.name, status='active'), but with phone populated.

        Returns the curator's contributor_id on success, None on miss
        (sender isn't admin, or any Supabase call failed).
        """
        admin_phone = os.environ.get("HOMER_ADMIN_PHONE", "").strip()
        if not admin_phone or admin_phone != sender_phone:
            return None

        # Find an existing curator row for this household.
        try:
            r = await self._http().get(
                "/hist_contributors",
                params={
                    "select": "id,phone",
                    "household_id": f"eq.{self._household_id}",
                    "role": f"eq.{_CURATOR_ROLE}",
                    "limit": "1",
                },
            )
            r.raise_for_status()
            existing = r.json()
        except Exception:
            logger.warning(
                "chat_persist: curator lookup failed for bootstrap", exc_info=True,
            )
            return None

        if existing:
            cur = existing[0]
            cur_phone = cur.get("phone")
            if cur_phone == sender_phone:
                logger.debug("chat_persist: curator row already has matching phone")
                return cur["id"]
            if cur_phone:
                # A different phone is already on the curator row. Don't
                # override — that's almost certainly an admin's previous
                # number, and overwriting silently would lose audit trail.
                logger.warning(
                    "chat_persist: curator row has phone=%s but sender=%s; not overriding",
                    cur_phone, sender_phone,
                )
                return None
            # phone is null — patch it.
            try:
                r = await self._http().patch(
                    "/hist_contributors",
                    params={"id": f"eq.{cur['id']}"},
                    json={"phone": sender_phone},
                )
                r.raise_for_status()
            except Exception:
                logger.warning(
                    "chat_persist: failed to patch curator phone", exc_info=True,
                )
                return None
            logger.info(
                "chat_persist: bootstrapped curator phone for id=%s", cur["id"],
            )
            return cur["id"]

        # No curator row yet — insert one. Mirror what
        # `ensure_contributor_for_user` Path 2 does on the portal side, but
        # with phone populated since we have it from the inbound sender.
        try:
            r = await self._http().get(
                "/household_members",
                params={
                    "select": "user_id,name",
                    "household_id": f"eq.{self._household_id}",
                    "role": f"eq.{_ADMIN_ROLE}",
                    "limit": "1",
                },
            )
            r.raise_for_status()
            admin_rows = r.json()
        except Exception:
            logger.warning(
                "chat_persist: household_members lookup failed for bootstrap",
                exc_info=True,
            )
            return None

        if not admin_rows:
            logger.warning(
                "chat_persist: no admin in household_members for household=%s — "
                "cannot bootstrap curator",
                self._household_id,
            )
            return None

        admin = admin_rows[0]
        new_row = {
            "household_id": self._household_id,
            "role": _CURATOR_ROLE,
            "display_name": (admin.get("name") or "").strip() or "Curator",
            "phone": sender_phone,
            "auth_user_id": admin.get("user_id"),
            "status": "active",
        }
        try:
            r = await self._http().post(
                "/hist_contributors",
                headers={"Prefer": "return=representation"},
                json=new_row,
            )
            r.raise_for_status()
            inserted = r.json()
        except Exception:
            logger.warning(
                "chat_persist: curator insert failed for bootstrap", exc_info=True,
            )
            return None

        if isinstance(inserted, list) and inserted:
            new_id = inserted[0].get("id")
            if new_id:
                logger.info(
                    "chat_persist: bootstrapped new curator row id=%s for household=%s",
                    new_id, self._household_id,
                )
                return new_id
        return None

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
        self,
        *,
        role: str,
        contributor_id: str,
        text: str,
        pending_upload: dict[str, Any] | None = None,
    ) -> None:
        body: dict[str, Any] = {
            "household_id": self._household_id,
            "contributor_id": contributor_id,
            "role": role,
            "text": text,
        }
        if pending_upload is not None:
            body["pending_upload"] = pending_upload
        try:
            r = await self._http().post("/hist_chat_messages", json=body)
            r.raise_for_status()
        except Exception:
            logger.warning(
                "chat_persist: insert failed (role=%s contributor=%s)",
                role, contributor_id, exc_info=True,
            )

    async def _upload_one_media(
        self, path_str: str,
    ) -> dict[str, Any] | None:
        """Upload a single media file to Supabase storage; return pending_upload.

        Caller (typically `_upload_and_insert_media` running on the
        background-task pool) loops over the inbound's full media list and
        invokes this once per file — every attachment lands in storage as
        its own `hist_chat_messages` row, no media is dropped.

        Returns None if the file is missing, the kind isn't recognized as
        image/audio/video, the file is too big, or the upload fails — in
        every case the failure is logged WARN. The text row from
        `on_message_received` already captured the textual content, so a
        failed media upload never loses the contributor's words.
        """
        if not path_str:
            return None
        path = Path(path_str)
        if not path.is_file():
            logger.warning("chat_persist: media file not found at %s — skipping", path_str)
            return None

        mime, _ = mimetypes.guess_type(path_str)
        kind = _kind_from_mime(mime)
        if kind is None:
            logger.warning(
                "chat_persist: unsupported media kind (mime=%r path=%s) — skipping",
                mime, path_str,
            )
            return None

        try:
            size = path.stat().st_size
        except OSError:
            logger.warning("chat_persist: stat failed on %s — skipping", path_str, exc_info=True)
            return None
        if size > _MAX_UPLOAD_BYTES:
            logger.warning(
                "chat_persist: media exceeds %d bytes (size=%d path=%s) — skipping",
                _MAX_UPLOAD_BYTES, size, path_str,
            )
            return None

        filename = path.name
        storage_path = _build_storage_path(filename)
        bucket = _bucket_for(self._household_id)

        try:
            data = path.read_bytes()
        except OSError:
            logger.warning("chat_persist: read failed on %s — skipping", path_str, exc_info=True)
            return None

        if not await self._upload_object(bucket, storage_path, data, mime):
            return None

        return {
            "storage_path": f"{bucket}/{storage_path}",
            "filename": filename,
            "mime": mime,
            "kind": kind,
        }

    async def _upload_object(
        self, bucket: str, object_path: str, data: bytes, mime: str | None,
    ) -> bool:
        """PUT object bytes; auto-provision the bucket on first 404/400.

        Returns True on success, False on any failure. Logs WARN with the
        underlying error so a failed upload can be diagnosed without crashing
        the conversation.
        """
        path = f"/object/{bucket}/{object_path}"
        headers = {"Content-Type": mime or "application/octet-stream"}

        async def _put() -> httpx.Response | None:
            try:
                return await self._storage_http().post(path, headers=headers, content=data)
            except Exception:
                logger.warning(
                    "chat_persist: storage upload error (bucket=%s path=%s)",
                    bucket, object_path, exc_info=True,
                )
                return None

        resp = await _put()
        if resp is None:
            return False
        if resp.status_code in (200, 201):
            return True
        if resp.status_code in (400, 404):
            # Bucket likely missing — log the original error so a non-bucket
            # 400 (malformed path, etc.) is still diagnosable post-retry.
            logger.debug(
                "chat_persist: storage upload returned %d, attempting bucket provision (body=%s)",
                resp.status_code, resp.text[:200],
            )
            if not await self._ensure_bucket(bucket):
                return False
            resp = await _put()
            if resp is None:
                return False
            if resp.status_code in (200, 201):
                return True
        logger.warning(
            "chat_persist: storage upload failed (status=%d bucket=%s path=%s body=%s)",
            resp.status_code, bucket, object_path, resp.text[:200],
        )
        return False

    async def _ensure_bucket(self, bucket: str) -> bool:
        """Create the per-household private bucket if missing.

        Mirrors `backend/services/history_service.py:_ensure_history_bucket`:
        Storage returns HTTP 400 with a 409-shaped JSON body when a bucket
        already exists, so we have to read the body to distinguish "already
        there" (success) from a real validation error.
        """
        try:
            resp = await self._storage_http().post(
                "/bucket",
                headers={"Content-Type": "application/json"},
                json={"id": bucket, "name": bucket, "public": False},
                timeout=15.0,
            )
        except Exception:
            logger.warning("chat_persist: bucket ensure error (bucket=%s)", bucket, exc_info=True)
            return False
        if resp.status_code in (200, 201, 409):
            return True
        if resp.status_code == 400:
            try:
                body = resp.json()
            except ValueError:
                body = {}
            if str(body.get("statusCode")) == "409" or body.get("error") == "Duplicate":
                return True
        logger.warning(
            "chat_persist: bucket ensure failed (status=%d bucket=%s body=%s)",
            resp.status_code, bucket, resp.text[:200],
        )
        return False


# ── singleton accessor ───────────────────────────────────────────────────

_HOOK: ChatPersistHook | None = None


def get_chat_persist_hook() -> ChatPersistHook:
    """Return the process-wide ChatPersistHook (lazy)."""
    global _HOOK
    if _HOOK is None:
        _HOOK = ChatPersistHook()
    return _HOOK
