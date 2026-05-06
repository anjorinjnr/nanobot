"""
In-process WhatsApp client wrapping neonize (whatsmeow Go bindings).

Replaces the previous Node.js Baileys bridge: same callback surface
(on_message / on_qr / on_status), no WebSocket IPC, no separate process.

Surface mirrors the TS bridge's WhatsAppClient class so the channel layer
needs only a thin refactor — sender/pn JIDs are emitted in the legacy
"@s.whatsapp.net" / "@lid.whatsapp.net" form even though neonize internally
uses "lid" as the server, so existing identity-resolution code keeps working.
"""

from __future__ import annotations

import asyncio
import mimetypes
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

from loguru import logger

from neonize.aioze.client import NewAClient
from neonize.aioze.events import (
    ConnectedEv,
    DisconnectedEv,
    LoggedOutEv,
    MessageEv,
)
from neonize.proto.Neonize_pb2 import JID
from neonize.utils.enum import ChatPresence, ChatPresenceMedia
from neonize.utils.jid import Jid2String, build_jid


# Public servers neonize exposes — kept identical to the bridge's wire format
# so existing channel code that greps for "@s.whatsapp.net" / "@lid.whatsapp.net"
# does not have to change.
_LEGACY_LID_SUFFIX = "@lid.whatsapp.net"
_LEGACY_USER_SUFFIX = "@s.whatsapp.net"
_LEGACY_GROUP_SUFFIX = "@g.us"


@dataclass
class InboundMessage:
    id: str
    sender: str  # full JID-like string ("12345@s.whatsapp.net" or "12345@lid.whatsapp.net" or "12345@g.us")
    pn: str  # phone-number JID when distinct from sender (else "")
    content: str
    timestamp: int
    is_group: bool
    was_mentioned: bool = False
    media: list[str] = field(default_factory=list)
    push_name: str = ""


@dataclass
class WhatsAppClientOptions:
    auth_dir: Path
    on_message: Callable[[InboundMessage], Awaitable[None]]
    on_qr: Callable[[str], Awaitable[None]]
    on_status: Callable[[str], Awaitable[None]]


def _legacy_jid_string(jid: JID) -> str:
    """Render a neonize JID in the legacy bridge format.

    neonize uses ``lid`` as the server for hidden users; the bridge emitted
    ``@lid.whatsapp.net``. Group JIDs use ``g.us``; the bridge emitted ``@g.us``.
    The User segment includes any device/agent suffix as Jid2String formats it,
    but for sender identity we strip the device — callers split on ":" downstream.
    """
    user = jid.User
    server = jid.Server
    if server == "lid":
        return f"{user}{_LEGACY_LID_SUFFIX}"
    if server == "s.whatsapp.net":
        return f"{user}{_LEGACY_USER_SUFFIX}"
    if server == "g.us":
        return f"{user}{_LEGACY_GROUP_SUFFIX}"
    # Status broadcast etc. — pass through as-is using neonize's renderer
    return Jid2String(jid)


def _parse_legacy_jid(s: str) -> JID:
    """Parse a legacy "user@server" string back into a neonize JID for outbound use."""
    if "@" not in s:
        return build_jid(s)
    user, _, server = s.partition("@")
    # Strip device suffix if present ("12345:42@s.whatsapp.net")
    user = user.split(":", 1)[0]
    # Bridge surfaced LIDs as "@lid.whatsapp.net"; neonize uses "lid".
    # All other servers (s.whatsapp.net, g.us, broadcast, newsletter) are
    # the same string in both representations and pass through unchanged.
    if server == "lid.whatsapp.net":
        server = "lid"
    return build_jid(user, server=server)


class WhatsAppClient:
    """Async WhatsApp client backed by neonize."""

    def __init__(self, options: WhatsAppClientOptions) -> None:
        self.options = options
        self.options.auth_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = options.auth_dir / "neonize.db"
        self._media_dir = options.auth_dir.parent / "media"
        self._media_dir.mkdir(parents=True, exist_ok=True)
        self._client: NewAClient | None = None
        self._idle_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ lifecycle

    async def connect(self) -> None:
        client = NewAClient(str(self._db_path))
        self._client = client

        @client.qr
        async def _on_qr(_: NewAClient, qr_bytes: bytes) -> None:
            # neonize delivers QR data as bytes — comma-separated WhatsApp
            # ref-pubkey-deviceid-advsecret blob, ready to render directly.
            try:
                code = qr_bytes.decode("utf-8", errors="replace")
            except Exception:
                code = ""
            if code:
                await self.options.on_qr(code)

        @client.event(ConnectedEv)
        async def _on_connected(_: NewAClient, __: ConnectedEv) -> None:
            await self.options.on_status("connected")

        @client.event(DisconnectedEv)
        async def _on_disconnected(_: NewAClient, __: DisconnectedEv) -> None:
            await self.options.on_status("disconnected")

        @client.event(LoggedOutEv)
        async def _on_logged_out(_: NewAClient, __: LoggedOutEv) -> None:
            logger.warning("WhatsApp session logged out — re-pair required")
            await self.options.on_status("logged_out")

        @client.event(MessageEv)
        async def _on_message(_: NewAClient, ev: MessageEv) -> None:
            try:
                await self._handle_message(ev)
            except Exception as e:
                logger.exception("WhatsApp inbound handler error: {}", e)

        await client.connect()
        # connect() returns a Task running the cgo loop; await idle so this
        # coroutine doesn't return until the loop ends or is cancelled.
        self._idle_task = asyncio.create_task(self._await_idle())

    async def _await_idle(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.idle()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("WhatsApp idle loop ended: {}", e)

    async def wait_until_idle(self) -> None:
        """Block until the underlying neonize idle loop returns.

        ``connect()`` schedules an idle task that runs until the WhatsApp
        socket disconnects. Callers (e.g. the channel adapter) await this
        to keep the channel alive between connect/reconnect cycles.
        """
        if self._idle_task is None:
            return
        try:
            await self._idle_task
        except asyncio.CancelledError:
            raise
        except Exception:
            # Idle loop swallows its own exception inside _await_idle; a
            # raise here would only happen on cancellation propagation.
            pass

    async def disconnect(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.debug("WhatsApp disconnect error (ignored): {}", e)
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except (asyncio.CancelledError, Exception):
                pass
        self._client = None
        self._idle_task = None

    # ------------------------------------------------------------------ inbound

    async def _handle_message(self, ev: MessageEv) -> None:
        info = ev.Info
        source = info.MessageSource
        if source.IsFromMe:
            return
        chat_jid = source.Chat
        chat_str = _legacy_jid_string(chat_jid)
        if chat_str.startswith("status@"):
            return
        is_group = chat_jid.Server == "g.us"
        # The bridge mirrored Baileys' remoteJidAlt — the alt form of the chat,
        # which only carries useful LID↔phone info for 1:1. Surface SenderAlt
        # only for 1:1; for groups leave pn empty so identity-mapping logic
        # doesn't try to learn against the group JID.
        pn_str = ""
        if not is_group:
            alt = getattr(source, "SenderAlt", None)
            if alt and alt.User:
                pn_str = _legacy_jid_string(alt)
        reply_to = chat_str

        msg = ev.Message
        text, fallback, is_audio, media_proto = self._extract_text_and_media_kind(msg)
        media_paths: list[str] = []
        if media_proto is not None:
            path = await self._download_media(msg, media_proto, is_audio=is_audio)
            if path:
                media_paths.append(path)

        final_content = text or ((fallback if (is_audio or not media_paths) else "") or "")
        if not final_content and not media_paths:
            return

        was_mentioned = self._was_self_mentioned(msg, is_group)

        try:
            push_name = info.Pushname or ""
        except Exception:
            push_name = ""

        try:
            ts = int(info.Timestamp.seconds) if info.Timestamp else int(time.time())
        except Exception:
            ts = int(time.time())

        # ``sender`` carries the full chat JID (group or 1:1 user) so the
        # channel can reply to the right destination. Group participant
        # identity is intentionally NOT exposed — the bridge couldn't, and
        # callers don't yet handle it. If we expose it later, prefer
        # ``MessageSource.Sender`` (already available locally).
        await self.options.on_message(
            InboundMessage(
                id=info.ID or "",
                sender=reply_to,
                pn=pn_str,
                content=final_content,
                timestamp=ts,
                is_group=is_group,
                was_mentioned=was_mentioned,
                media=media_paths,
                push_name=push_name,
            )
        )

    def _extract_text_and_media_kind(self, msg) -> tuple[Optional[str], Optional[str], bool, object]:
        """Return (text, fallback_content, is_audio, media_proto_or_None)."""
        if not msg:
            return None, None, False, None

        if msg.HasField("conversation") and msg.conversation:
            return msg.conversation, None, False, None
        if msg.HasField("extendedTextMessage") and msg.extendedTextMessage.text:
            return msg.extendedTextMessage.text, None, False, None
        if msg.HasField("imageMessage"):
            return msg.imageMessage.caption or "", "[Image]", False, msg.imageMessage
        if msg.HasField("videoMessage"):
            return msg.videoMessage.caption or "", "[Video]", False, msg.videoMessage
        if msg.HasField("documentMessage"):
            return msg.documentMessage.caption or "", "[Document]", False, msg.documentMessage
        if msg.HasField("audioMessage"):
            # Voice note (PTT) or audio file — channel keys off the
            # "[Voice Message]" sentinel for Whisper transcription.
            return None, "[Voice Message]", True, msg.audioMessage
        if msg.HasField("contactMessage"):
            cm = msg.contactMessage
            name = cm.displayName or "Unknown"
            return f"[Contact: {name}]\n{cm.vcard or ''}", None, False, None
        return None, None, False, None

    def _was_self_mentioned(self, msg, is_group: bool) -> bool:
        if not is_group or self._client is None:
            return False
        # Read self-identity lazily from neonize's client.me — populated by
        # whatsmeow's "Me" event, which races against ConnectedEv. Reading at
        # call time avoids a startup window where group mentions get missed.
        me = getattr(self._client, "me", None)
        if me is None or not getattr(me, "User", ""):
            return False
        self_user = me.User.split(":", 1)[0]
        ctx_candidates = []
        for field_name in (
            "extendedTextMessage",
            "imageMessage",
            "videoMessage",
            "documentMessage",
            "audioMessage",
        ):
            try:
                if msg.HasField(field_name):
                    sub = getattr(msg, field_name)
                    if sub.HasField("contextInfo"):
                        ctx_candidates.append(sub.contextInfo)
            except (ValueError, AttributeError):
                continue
        for ctx in ctx_candidates:
            try:
                for jid_str in ctx.mentionedJID:
                    bare = jid_str.split("@", 1)[0].split(":", 1)[0]
                    if bare == self_user:
                        return True
            except (AttributeError, TypeError):
                continue
        return False

    async def _download_media(self, msg, media_proto, *, is_audio: bool) -> Optional[str]:
        if self._client is None:
            return None
        try:
            data = await self._client.download_any(msg)
        except Exception as e:
            logger.error("WhatsApp media download failed: {}", e)
            return None
        if not data:
            return None

        mimetype = ""
        file_name = ""
        try:
            mimetype = media_proto.mimetype or ""
        except AttributeError:
            mimetype = ""
        try:
            file_name = getattr(media_proto, "fileName", "") or ""
        except AttributeError:
            file_name = ""

        suffix = self._suffix_for(mimetype, file_name, is_audio=is_audio)
        prefix = f"wa_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
        out_name = f"{prefix}_{file_name}" if file_name else f"{prefix}{suffix}"
        out_path = self._media_dir / out_name
        try:
            out_path.write_bytes(data)
        except OSError as e:
            logger.error("WhatsApp media write failed: {}", e)
            return None
        return str(out_path)

    @staticmethod
    def _suffix_for(mimetype: str, file_name: str, *, is_audio: bool) -> str:
        if file_name and "." in file_name:
            return ""  # filename will carry its own extension
        mt = (mimetype or "").split(";", 1)[0].strip()
        if mt:
            ext = mimetypes.guess_extension(mt)
            if ext:
                return ext
        if is_audio:
            return ".ogg"  # PTT default codec
        return ".bin"

    # ------------------------------------------------------------------ outbound

    async def send_message(self, to: str, text: str) -> None:
        # Bridge era returned ``{"lid": resolved_jid}`` so the channel could
        # learn outbound phone↔LID. neonize's SendResponse doesn't carry the
        # resolved JID; inbound MessageSource is the single LID-learning path
        # now (see whatsapp.py:_save_lid_mapping). Audited 2026-05-05 —
        # nothing in nanobot or homer reads the return value.
        if self._client is None:
            raise RuntimeError("WhatsApp client not connected")
        await self._client.send_message(_parse_legacy_jid(to), text)

    async def send_media(
        self,
        to: str,
        file_path: str,
        mimetype: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> None:
        if self._client is None:
            raise RuntimeError("WhatsApp client not connected")
        jid = _parse_legacy_jid(to)
        category = (mimetype or "").split("/", 1)[0]
        is_ptt = bool(re.search(r"audio/.*opus", mimetype or "", re.IGNORECASE))
        if category == "image":
            await self._client.send_image(jid, file_path, caption=caption or None)
        elif category == "video":
            await self._client.send_video(jid, file_path, caption=caption or None)
        elif category == "audio":
            await self._client.send_audio(jid, file_path, ptt=is_ptt)
        else:
            await self._client.send_document(
                jid,
                file_path,
                caption=caption or None,
                filename=file_name or Path(file_path).name,
                mimetype=mimetype or None,
            )

    async def send_typing(self, to: str, composing: bool) -> None:
        if self._client is None:
            return
        jid = _parse_legacy_jid(to)
        state = (
            ChatPresence.CHAT_PRESENCE_COMPOSING if composing else ChatPresence.CHAT_PRESENCE_PAUSED
        )
        try:
            await self._client.send_chat_presence(
                jid, state, ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT
            )
        except Exception:
            # Typing indicators are best-effort — the bridge swallowed errors here too.
            pass


