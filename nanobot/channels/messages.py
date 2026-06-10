"""Provider-neutral phone-messaging channel (iMessage / RCS / SMS).

This channel reaches people at a phone number across whatever protocol the
upstream provider can deliver (blue-bubble iMessage, RCS, or SMS/MMS). It is
deliberately **provider-agnostic**: the homer-portal *messaging gateway* owns the
provider (Linq, Telnyx, …) and its credentials, so this channel never holds a
provider key. Swapping providers is a portal-side change and never touches nanobot.

Flow (both hops are plain authenticated HTTP, shared secret ``gateway_token``):

- Inbound: provider → gateway webhook → gateway normalizes + resolves household →
  POST to this channel's tiny built-in HTTP server (``ingest_port``) → bus.
- Outbound: ``send()`` → POST to ``{gateway_url}/api/messaging/send`` → gateway
  fans out to the active provider.

The built-in HTTP server mirrors the Microsoft Teams channel
(:mod:`nanobot.channels.msteams`): a stdlib ``ThreadingHTTPServer`` on a daemon
thread that hands work to the asyncio loop via ``run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base

# Audio MIME prefixes that should be transcribed rather than handed to the model
# as an opaque file path (matches the voice-note handling in other channels).
_AUDIO_EXTS = {".mp3", ".m4a", ".ogg", ".oga", ".opus", ".wav", ".amr", ".aac", ".caf"}


class MessagesConfig(Base):
    """Configuration for the provider-neutral phone-messaging channel."""

    enabled: bool = False
    allow_from: list[str] = Field(default_factory=list)  # E.164 numbers / handles
    address: str = ""  # this household's own sender number (informational)
    gateway_url: str = ""  # homer-portal base URL, e.g. https://portal.example.com
    gateway_token: str = ""  # shared secret authenticating both directions
    ingest_host: str = "0.0.0.0"
    ingest_port: int = 0  # set by portal provisioning; 0 disables the listener
    ingest_path: str = "/inbound"


class MessagesChannel(BaseChannel):
    """Phone-messaging channel backed by a portal-side provider gateway."""

    name = "messages"
    display_name = "Text / iMessage"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return MessagesConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = MessagesConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: MessagesConfig = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Start the inbound webhook listener and the outbound HTTP client."""
        if not self.config.gateway_url or not self.config.gateway_token:
            self.logger.error("gateway_url/gateway_token not configured; channel disabled")
            return
        if not self.config.ingest_port:
            self.logger.error("ingest_port not configured; channel disabled")
            return

        self._loop = asyncio.get_running_loop()
        self._http = httpx.AsyncClient(timeout=30.0)
        self._running = True

        channel = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, code: int, body: bytes = b"{}") -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # health check for the portal preflight
                self._reply(200, b'{"status":"ok"}')

            def do_POST(self) -> None:
                if self.path.split("?", 1)[0] != channel.config.ingest_path:
                    self._reply(404)
                    return
                if not channel._authorized(self.headers.get("Authorization", "")):
                    self._reply(401, b'{"error":"unauthorized"}')
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    payload = json.loads(raw.decode("utf-8"))
                except Exception as e:
                    channel.logger.warning("Invalid inbound body: {}", e)
                    self._reply(400, b'{"error":"bad request"}')
                    return
                try:
                    fut = asyncio.run_coroutine_threadsafe(
                        channel._handle_inbound(payload), channel._loop
                    )
                    fut.result(timeout=30)
                except Exception as e:
                    channel.logger.warning("Inbound handling failed: {}", e)
                    self._reply(500, b'{"error":"internal"}')
                    return
                self._reply(200)

            def log_message(self, fmt: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self.config.ingest_host, self.config.ingest_port), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, name="nanobot-messages", daemon=True
        )
        self._server_thread.start()
        self.logger.info(
            "Inbound listener on http://{}:{}{}",
            self.config.ingest_host, self.config.ingest_port, self.config.ingest_path,
        )

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2)
        self._server_thread = None
        if self._http:
            await self._http.aclose()
            self._http = None

    # -------------------------------------------------------------------- inbound

    def _authorized(self, auth_header: str) -> bool:
        """Constant-time check of the gateway shared secret on inbound requests."""
        if not auth_header.lower().startswith("bearer "):
            return False
        token = auth_header.split(" ", 1)[1].strip()
        return bool(token) and hmac.compare_digest(token, self.config.gateway_token)

    async def _handle_inbound(self, payload: dict[str, Any]) -> None:
        """Normalize a gateway-delivered inbound message and publish it to the bus.

        Expected (provider-neutral) payload from the portal::

            {"from": "+1555…", "body": "hi", "media": ["https://…"],
             "modality": "imessage"|"rcs"|"sms", "message_id": "…", "chat_id": "…"}
        """
        sender = str(payload.get("from") or "").strip()
        if not sender:
            self.logger.warning("Inbound dropped: missing 'from'")
            return

        content = str(payload.get("body") or "")
        media_urls = payload.get("media") or []
        modality = str(payload.get("modality") or "").strip().lower()

        media_paths: list[str] = []
        for url in media_urls:
            path = await self._download_media(str(url), str(payload.get("message_id") or sender))
            if not path:
                continue
            # Transcribe voice notes inline so the model sees text, not a file path.
            if Path(path).suffix.lower() in _AUDIO_EXTS:
                transcript = await self.transcribe_audio(path)
                if transcript:
                    content = (content + "\n" + transcript).strip() if content else transcript
                    continue
            media_paths.append(path)

        if not content and not media_paths:
            self.logger.debug("Inbound from {} had no text or media; ignoring", sender)
            return

        # 1:1 texting is a DM — unknown senders get a pairing code (base handles it).
        await self._handle_message(
            sender_id=sender,
            chat_id=sender,
            content=content,
            media=media_paths,
            metadata={
                "modality": modality,
                "provider_message_id": payload.get("message_id"),
                "provider_chat_id": payload.get("chat_id"),
            },
            is_dm=True,
        )

    async def _download_media(self, url: str, tag: str) -> str | None:
        """Fetch an inbound media URL into the per-channel media dir; return its path."""
        if not self._http:
            return None
        try:
            resp = await self._http.get(url)
            resp.raise_for_status()
        except Exception as e:
            self.logger.warning("Media download failed for {}: {}", url, e)
            return None
        ext = Path(urlparse(url).path).suffix
        if not ext:
            ext = mimetypes.guess_extension(resp.headers.get("Content-Type", "").split(";")[0]) or ".bin"
        safe_tag = re.sub(r"[^A-Za-z0-9_-]", "_", tag)[:40]
        dest = get_media_dir("messages") / f"{safe_tag}_{abs(hash(url)) % 10_000_000}{ext}"
        dest.write_bytes(resp.content)
        return str(dest)

    # ------------------------------------------------------------------- outbound

    async def send(self, msg: OutboundMessage) -> None:
        """Send a reply by handing it to the portal messaging gateway."""
        if not self._http:
            raise RuntimeError("messages channel HTTP client not initialized")

        url = self.config.gateway_url.rstrip("/") + "/api/messaging/send"
        headers = {"Authorization": f"Bearer {self.config.gateway_token}"}
        data = {
            "to": str(msg.chat_id),
            "from": self.config.address,
            "text": _to_plain_text(msg.content or ""),
        }
        reply_to = (msg.metadata or {}).get("provider_chat_id")
        if reply_to:
            data["chat_id"] = str(reply_to)

        try:
            if msg.media:
                # Stream raw bytes; the gateway uploads to the provider (P2: media).
                files = []
                for path in msg.media:
                    p = Path(path)
                    if p.is_file():
                        mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
                        files.append(("media", (p.name, p.read_bytes(), mime)))
                resp = await self._http.post(url, headers=headers, data=data, files=files or None)
            else:
                resp = await self._http.post(url, headers=headers, json=data)
            resp.raise_for_status()
        except Exception:
            self.logger.exception("Send failed for chat_id={}", msg.chat_id)
            raise
        self.logger.info("Message sent to {}", msg.chat_id)


def _to_plain_text(text: str) -> str:
    """Flatten markdown to plain text — SMS/iMessage render no markup."""
    if not text:
        return text
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images → drop
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)  # links → "label (url)"
    text = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", text)  # inline/code fences
    text = re.sub(r"(\*\*|__|\*|_)(.*?)\1", r"\2", text)  # bold/italic
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # headers
    return text.strip()
