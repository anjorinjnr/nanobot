"""
WhatsApp channel — uses neonize (in-process whatsmeow Go bindings).

Replaces the previous Node.js Baileys bridge: no separate process, no
WebSocket IPC, no BRIDGE_TOKEN. The protocol client lives in
``whatsapp_client.WhatsAppClient`` and is consumed directly here.
"""

import asyncio
import json
import mimetypes
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.whatsapp_client import (
    InboundMessage as WAInboundMessage,
    WhatsAppClient,
    WhatsAppClientOptions,
)
from nanobot.config.schema import Base


class WhatsAppConfig(Base):
    """WhatsApp channel configuration."""

    enabled: bool = False
    allow_from: list[str] = Field(default_factory=list)
    group_policy: Literal["open", "mention"] = "open"
    identity_resolution: bool = False  # Enable LID→name resolution via sender_map/lid_map


def _whatsapp_auth_dir() -> Path:
    """Resolve the WhatsApp session storage directory.

    The neonize sqlite DB held here represents a full WhatsApp Web pairing —
    losing it on a redeploy means re-scanning the QR. Containerized
    deployments must put it on a bind-mounted volume.

    Resolution order:
        1. ``NANOBOT_WHATSAPP_AUTH_DIR`` env var (explicit override). Set this
           in the container/host entrypoint to point at a persistent volume
           (e.g. ``/data/whatsapp-auth`` in homer's container layout).
        2. ``get_runtime_subdir("whatsapp-auth")`` — the per-config-instance
           subdir under the active nanobot data directory (good for local dev,
           ephemeral inside a stock container).
    """
    override = os.environ.get("NANOBOT_WHATSAPP_AUTH_DIR")
    if override:
        return Path(override).expanduser()

    from nanobot.config.paths import get_runtime_subdir

    return get_runtime_subdir("whatsapp-auth")


class WhatsAppChannel(BaseChannel):
    """WhatsApp channel backed by an in-process neonize client."""

    name = "whatsapp"
    display_name = "WhatsApp"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WhatsAppConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = WhatsAppConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: WhatsAppConfig = config
        self._client: WhatsAppClient | None = None
        self._connected = False
        # Set by _on_status when neonize fires ConnectedEv. Reset per-iteration
        # in start()/login() so the reconnect loop can tell whether the current
        # connection attempt actually reached "connected" (used to decide
        # whether to reset the backoff floor).
        self._connected_event: asyncio.Event = asyncio.Event()
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._typing_tasks: dict[str, asyncio.Task] = {}
        # LID identity resolution state
        self._lid_map: dict[str, dict] = {}
        self._lid_map_lock = asyncio.Lock()
        self._lid_map_loaded = False
        self._sender_map: dict[str, str] = {}
        self._greeted_sessions: OrderedDict[str, None] = OrderedDict()
        self._lid_to_phone: dict[str, str] = {}

    # ----------------------------------------------------------- lifecycle

    def _build_client(self) -> WhatsAppClient:
        return WhatsAppClient(
            WhatsAppClientOptions(
                auth_dir=_whatsapp_auth_dir(),
                on_message=self._on_inbound,
                on_qr=self._on_qr,
                on_status=self._on_status,
            )
        )

    async def login(self, force: bool = False) -> bool:
        """Pair the WhatsApp account interactively (QR scan)."""
        client = self._build_client()
        self._client = client
        self._connected_event.clear()
        try:
            await client.connect()
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=300.0)
                logger.info("WhatsApp pairing complete")
                return True
            except asyncio.TimeoutError:
                logger.error("WhatsApp pairing timed out — no QR scanned within 5 minutes")
                return False
        finally:
            await client.disconnect()
            self._client = None

    # Reconnect backoff: start at 5s, double on each failed attempt up to 5min.
    # Reset to the floor on a successful connection so transient blips don't
    # leave us paying long delays after recovery.
    _RECONNECT_MIN_SECONDS = 5
    _RECONNECT_MAX_SECONDS = 300

    async def start(self) -> None:
        """Start the channel by connecting to WhatsApp via neonize."""
        if self.config.identity_resolution:
            await self._ensure_maps_loaded()

        self._running = True
        backoff = self._RECONNECT_MIN_SECONDS

        while self._running:
            client = self._build_client()
            self._client = client
            # Reset the per-iteration connected signal — _on_status will set it
            # if neonize fires ConnectedEv this cycle.
            self._connected_event.clear()
            try:
                await client.connect()
                await client.wait_until_idle()
                logger.info("WhatsApp idle loop ended")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("WhatsApp client error: {}", e)
            finally:
                self._connected = False
                try:
                    await client.disconnect()
                except Exception:
                    pass
                self._client = None

            if not self._running:
                break

            # Reset backoff to floor only if this iteration reached "connected"
            # at least once — transient blips after a stable session shouldn't
            # pay the long delay; persistent failures should keep escalating.
            if self._connected_event.is_set():
                backoff = self._RECONNECT_MIN_SECONDS
            logger.info("Reconnecting WhatsApp in {}s", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._RECONNECT_MAX_SECONDS)

    async def stop(self) -> None:
        self._running = False
        self._connected = False

        for chat_id in list(self._typing_tasks):
            await self._stop_typing(chat_id)

        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None

    # ----------------------------------------------------------- typing

    async def _start_typing(self, chat_id: str) -> None:
        await self._stop_typing(chat_id)
        self._typing_tasks[chat_id] = asyncio.create_task(self._typing_loop(chat_id))

    async def _stop_typing(self, chat_id: str) -> None:
        task = self._typing_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None and self._connected:
            try:
                await self._client.send_typing(chat_id, composing=False)
            except Exception:
                pass

    async def _typing_loop(self, chat_id: str) -> None:
        try:
            while self._client is not None and self._connected:
                await self._client.send_typing(chat_id, composing=True)
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("WhatsApp typing indicator stopped for {}: {}", chat_id, e)

    # ----------------------------------------------------------- outbound

    async def send(self, msg: OutboundMessage) -> None:
        if self._client is None or not self._connected:
            raise ConnectionError("WhatsApp client not connected")

        chat_id = msg.chat_id
        await self._stop_typing(chat_id)

        if msg.content and not msg.media:
            try:
                await self._client.send_message(chat_id, msg.content)
            except Exception as e:
                logger.error("Error sending WhatsApp message: {}", e)
                raise

        already_sent = set(msg.metadata.get("_sent_media", []))
        sent_media: list[str] = list(already_sent)
        for i, media_path in enumerate(msg.media or []):
            if media_path in already_sent:
                continue
            try:
                mime, _ = mimetypes.guess_type(media_path)
                caption = msg.content if (i == 0 and msg.content and not already_sent) else None
                await self._client.send_media(
                    to=chat_id,
                    file_path=media_path,
                    mimetype=mime or "application/octet-stream",
                    caption=caption,
                    file_name=Path(media_path).name,
                )
                sent_media.append(media_path)
            except Exception as e:
                msg.metadata["_sent_media"] = sent_media
                logger.error("Error sending WhatsApp media {}: {}", media_path, e)
                raise

    # ----------------------------------------------------------- inbound

    async def _on_status(self, status: str) -> None:
        logger.info("WhatsApp status: {}", status)
        if status == "connected":
            self._connected = True
            self._connected_event.set()
        elif status in ("disconnected", "logged_out"):
            self._connected = False
            # Don't clear the event here — start()/login() reset it per-cycle.
            # Holding it set across a disconnect lets the reconnect loop know
            # this iteration *did* connect, which is the signal it needs to
            # reset the backoff floor.

    async def _on_qr(self, qr: str) -> None:
        # Print a terminal-friendly QR code. The qrcode library is small
        # and pure-Python; we render to a tty so the user can scan it.
        try:
            import qrcode

            qr_obj = qrcode.QRCode(border=1)
            qr_obj.add_data(qr)
            qr_obj.make(fit=True)
            print("\n📱 Scan this QR code with WhatsApp (Linked Devices):\n", flush=True)
            qr_obj.print_ascii(invert=True)
            print("", flush=True)
        except ImportError:
            logger.info(
                "WhatsApp QR (paste into a QR generator to scan): {}", qr[:32] + "…"
            )

    async def _on_inbound(self, msg: WAInboundMessage) -> None:
        if msg.id and msg.id in self._processed_message_ids:
            return
        if msg.id:
            self._processed_message_ids[msg.id] = None
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

        if msg.is_group and getattr(self.config, "group_policy", "open") == "mention":
            if not msg.was_mentioned:
                return

        # Classify by JID suffix: @s.whatsapp.net = phone, @lid.whatsapp.net = LID
        raw_a = msg.pn or ""
        raw_b = msg.sender or ""
        id_a = raw_a.split("@")[0] if "@" in raw_a else raw_a
        id_b = raw_b.split("@")[0] if "@" in raw_b else raw_b

        phone_id = ""
        lid_id = ""
        for raw, extracted in [(raw_a, id_a), (raw_b, id_b)]:
            if "@s.whatsapp.net" in raw:
                phone_id = extracted
            elif "@lid.whatsapp.net" in raw:
                lid_id = extracted
            elif extracted and not phone_id:
                phone_id = extracted

        if phone_id and lid_id:
            self._lid_to_phone[lid_id] = phone_id
        sender_id = phone_id or self._lid_to_phone.get(lid_id, "") or lid_id or id_a or id_b

        logger.info(
            "Sender phone={} lid={} → sender_id={}",
            phone_id or "(empty)",
            lid_id or "(empty)",
            sender_id,
        )

        if self.config.identity_resolution:
            await self._ensure_maps_loaded()
            if msg.pn and msg.sender and msg.pn != msg.sender:
                await self._save_lid_mapping(msg.pn, msg.sender)

        media_paths = list(msg.media)
        content = msg.content

        # Voice message → transcribe via Whisper. The wrapper sets content
        # to "[Voice Message]" sentinel for PTT/audio (see whatsapp_client).
        if content == "[Voice Message]":
            if media_paths:
                logger.info("Transcribing voice message from {}...", sender_id)
                transcription = await self.transcribe_audio(media_paths[0])
                if transcription:
                    content = transcription
                    logger.info(
                        "Transcribed voice from {}: {}...", sender_id, transcription[:50]
                    )
                else:
                    content = "[Voice Message: Transcription failed]"
            else:
                content = "[Voice Message: Audio not available]"

        # Tag remaining media paths in content (matches Telegram pattern).
        # Voice messages flow through transcription above and intentionally
        # skip tagging — the transcription replaces the audio path.
        if media_paths and msg.content != "[Voice Message]":
            for p in media_paths:
                mime, _ = mimetypes.guess_type(p)
                media_type = "image" if mime and mime.startswith("image/") else "file"
                media_tag = f"[{media_type}: {p}]"
                content = f"{content}\n{media_tag}" if content else media_tag

        if self.is_allowed(sender_id):
            await self._start_typing(msg.sender)

        if (
            self.config.identity_resolution
            and content
            and not content.startswith("/")
        ):
            session_key = f"whatsapp:{msg.sender}"
            sender_name = self._resolve_sender_name(sender_id, session_key)
            if sender_name:
                content = f"[Sender: {sender_name}]\n{content}"

        await self._handle_message(
            sender_id=sender_id,
            chat_id=msg.sender,
            content=content,
            media=media_paths,
            metadata={
                "message_id": msg.id,
                "timestamp": msg.timestamp,
                "is_group": msg.is_group,
            },
        )

    # ----------------------------------------------------------- identity resolution

    def is_allowed(self, sender_id: str) -> bool:
        if super().is_allowed(sender_id):
            return True
        info = self._lid_map.get(sender_id)
        if isinstance(info, dict):
            phone = info.get("phone", "")
            if phone and super().is_allowed(phone):
                return True
        return False

    def _resolve_sender_name(self, sender_id: str, session_key: str) -> str | None:
        if session_key in self._greeted_sessions:
            return None

        name: str | None = None

        if sender_id in self._sender_map:
            name = self._sender_map[sender_id]

        if not name:
            info = self._lid_map.get(sender_id)
            if isinstance(info, dict):
                name = info.get("name") or None
                if not name:
                    phone = info.get("phone", "")
                    if phone and phone in self._sender_map:
                        name = self._sender_map[phone]

        if name:
            self._greeted_sessions[session_key] = None
            while len(self._greeted_sessions) > 500:
                self._greeted_sessions.popitem(last=False)

        return name

    async def _ensure_maps_loaded(self) -> None:
        if not self._lid_map_loaded:
            async with self._lid_map_lock:
                if not self._lid_map_loaded:
                    lid_map, sender_map = await asyncio.to_thread(self._read_maps_from_disk)
                    for k, v in lid_map.items():
                        if k not in self._lid_map:
                            self._lid_map[k] = v
                        elif isinstance(v, dict) and isinstance(self._lid_map[k], dict):
                            for field, val in v.items():
                                self._lid_map[k].setdefault(field, val)
                    self._sender_map = sender_map
                    self._lid_map_loaded = True

    def _read_maps_from_disk(self) -> tuple[dict, dict[str, str]]:
        from nanobot.config.paths import get_persistent_data_dir

        lid_map: dict = {}
        sender_map: dict[str, str] = {}

        lid_map_path = get_persistent_data_dir() / "lid_map.json"
        if lid_map_path.exists():
            try:
                raw = json.loads(lid_map_path.read_text(encoding="utf-8"))
                lid_map = (
                    {k: v for k, v in raw.items() if isinstance(v, dict)}
                    if isinstance(raw, dict)
                    else {}
                )
                logger.debug("Loaded lid_map with {} entries", len(lid_map))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load lid_map.json: {}", e)

        for candidate in self._sender_map_paths():
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        sender_map = {k: v for k, v in data.items() if isinstance(v, str)}
                        logger.debug(
                            "Loaded sender_map with {} entries from {}", len(sender_map), candidate
                        )
                        break
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to load sender_map from {}: {}", candidate, e)

        return lid_map, sender_map

    @staticmethod
    def _sender_map_paths() -> list[Path]:
        from nanobot.config.paths import get_data_dir

        return [get_data_dir() / "sender_map.json"]

    async def _save_lid_mapping(self, phone_jid: str, lid: str) -> None:
        lid_prefix = lid.split("@")[0] if "@" in lid else lid
        phone_digits = phone_jid.split("@")[0] if "@" in phone_jid else phone_jid

        existing = self._lid_map.get(lid_prefix)
        if isinstance(existing, dict) and existing.get("phone") == phone_digits:
            return

        if not isinstance(existing, dict):
            self._lid_map[lid_prefix] = {"phone": phone_digits}
        else:
            existing["phone"] = phone_digits

        async with self._lid_map_lock:
            snapshot = {k: dict(v) for k, v in self._lid_map.items()}
            await asyncio.to_thread(self._write_lid_map, snapshot)
        logger.info("LID mapping saved: {} → {}", lid_prefix, phone_digits)

    @staticmethod
    def _write_lid_map(data: dict) -> None:
        from nanobot.config.paths import get_persistent_data_dir

        map_path = get_persistent_data_dir() / "lid_map.json"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=map_path.parent, suffix=".tmp")
            os.close(fd)
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                Path(tmp_path).replace(map_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning("Failed to save LID mapping: {}", e)
