"""Channel manager for coordinating chat channels."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.bus.events import TASK_TAG_META_KEY, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.scope_guard import (
    OutboundScopeError,
    check_outbound,
)
from nanobot.config.schema import Config
from nanobot.utils.restart import consume_restart_notice_from_env, format_restart_completed_message

_DIGIT_PAT = re.compile(r"\d+")
_NONALNUM_PAT = re.compile(r"[^a-z0-9]+")


def _check_outbound_authorized(channel: BaseChannel, msg: OutboundMessage) -> None:
    """Raise OutboundScopeError when the host's scope guard refuses this send.

    Streaming continuations (``_stream_delta`` / ``_stream_end``) inherit the
    decision made on the initial send, since the chat_id and channel are the
    same — re-checking would just duplicate the lookup. The initial send is
    the gate.
    """
    if msg.metadata.get("_stream_delta") or msg.metadata.get("_stream_end"):
        return
    if not msg.chat_id:
        return
    result = check_outbound(channel.name, msg.chat_id)
    if result is None or result.authorized:
        return
    raise OutboundScopeError(channel.name, msg.chat_id, result)

if TYPE_CHECKING:
    from nanobot.session.manager import SessionManager


def _default_webui_dist() -> Path | None:
    """Return the absolute path to the bundled webui dist directory if it exists."""
    try:
        import nanobot.web as web_pkg  # type: ignore[import-not-found]
    except ImportError:
        return None
    candidate = Path(web_pkg.__file__).resolve().parent / "dist"
    return candidate if candidate.is_dir() else None

# Retry delays for message sending (exponential backoff: 1s, 2s, 4s)
_SEND_RETRY_DELAYS = (1, 2, 4)

# Interval between full cache sweeps (seconds)
_CLEANUP_INTERVAL_S = 60


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels (Telegram, WhatsApp, etc.)
    - Start/stop channels
    - Route outbound messages
    """

    def __init__(
        self,
        config: Config,
        bus: MessageBus,
        *,
        session_manager: "SessionManager | None" = None,
    ):
        self.config = config
        self.bus = bus
        self._session_manager = session_manager
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._channel_queues: dict[str, asyncio.Queue[tuple[BaseChannel, OutboundMessage]]] = {}
        self._channel_workers: dict[str, asyncio.Task] = {}
        # Spam guard: tracks (channel, chat_id, content_hash) → [timestamps]
        self._dedup_log: dict[tuple[str, str, str], list[float]] = {}
        self._dedup_last_cleanup: float = 0.0

        self._init_channels()
        self._install_scope_outbound_lookup()

    def _install_scope_outbound_lookup(self) -> None:
        """Install the host's outbound scope lookup (if configured).

        The config field is a "module:function" string. The callable receives
        ``(channel_name, chat_id)`` and returns ``ScopeLookupResult``.
        Failure to resolve the callable is logged but never fatal — the
        guard simply stays disabled (vanilla allow-all behavior).
        """
        spec = getattr(self.config.channels, "scope_outbound_lookup", "") or ""
        if not spec:
            return
        try:
            mod_name, fn_name = spec.split(":", 1)
        except ValueError:
            logger.error("scope_outbound_lookup {!r} is not 'module:function'", spec)
            return
        try:
            import importlib

            module = importlib.import_module(mod_name)
            fn = getattr(module, fn_name)
        except Exception as e:
            logger.error(
                "scope_outbound_lookup {!r} not importable: {}: {}",
                spec, type(e).__name__, e,
            )
            return
        if not callable(fn):
            logger.error(
                "scope_outbound_lookup {!r} is not callable (got {})", spec, type(fn).__name__,
            )
            return
        from nanobot.channels.scope_guard import set_scope_lookup
        set_scope_lookup(fn)
        logger.info("scope_outbound_lookup installed: {}", spec)

    def _init_channels(self) -> None:
        """Initialize channels discovered via pkgutil scan + entry_points plugins."""
        from nanobot.channels.registry import discover_all

        transcription_provider = self.config.channels.transcription_provider
        transcription_key = self._resolve_transcription_key(transcription_provider)
        transcription_base = self._resolve_transcription_base(transcription_provider)
        transcription_language = self.config.channels.transcription_language

        for name, cls in discover_all().items():
            section = getattr(self.config.channels, name, None)
            if section is None:
                continue
            enabled = (
                section.get("enabled", False)
                if isinstance(section, dict)
                else getattr(section, "enabled", False)
            )
            if not enabled:
                continue
            try:
                kwargs: dict[str, Any] = {}
                # Only the WebSocket channel currently hosts the embedded webui
                # surface; other channels stay oblivious to these knobs.
                if cls.name == "websocket" and self._session_manager is not None:
                    kwargs["session_manager"] = self._session_manager
                    static_path = _default_webui_dist()
                    if static_path is not None:
                        kwargs["static_dist_path"] = static_path
                channel = cls(section, self.bus, **kwargs)
                channel.transcription_provider = transcription_provider
                channel.transcription_api_key = transcription_key
                channel.transcription_api_base = transcription_base
                channel.transcription_language = transcription_language
                self.channels[name] = channel
                logger.info("{} channel enabled", cls.display_name)
            except Exception as e:
                logger.warning("{} channel not available: {}", name, e)

        self._validate_allow_from()

    def _resolve_transcription_key(self, provider: str) -> str:
        """Pick the API key for the configured transcription provider."""
        try:
            if provider == "openai":
                return self.config.providers.openai.api_key
            return self.config.providers.groq.api_key
        except AttributeError:
            return ""

    def _resolve_transcription_base(self, provider: str) -> str:
        """Pick the API base URL for the configured transcription provider."""
        try:
            if provider == "openai":
                return self.config.providers.openai.api_base or ""
            return self.config.providers.groq.api_base or ""
        except AttributeError:
            return ""

    def _validate_allow_from(self) -> None:
        for name, ch in self.channels.items():
            cfg = ch.config
            if isinstance(cfg, dict):
                if "allow_from" in cfg:
                    allow = cfg.get("allow_from")
                else:
                    allow = cfg.get("allowFrom")
            else:
                allow = getattr(cfg, "allow_from", None)
            if allow == []:
                raise SystemExit(
                    f'Error: "{name}" has empty allowFrom (denies all). '
                    f'Set ["*"] to allow everyone, or add specific user IDs.'
                )

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """Start a channel and log any exceptions."""
        try:
            await channel.start()
        except Exception as e:
            logger.error("Failed to start channel {}: {}", name, e)

    async def start_all(self) -> None:
        """Start all channels and the outbound dispatcher."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # Start outbound dispatcher
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # Start channels
        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting {} channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        self._notify_restart_done_if_needed()

        # Wait for all to complete (they should run forever)
        await asyncio.gather(*tasks, return_exceptions=True)

    def _notify_restart_done_if_needed(self) -> None:
        """Send restart completion message when runtime env markers are present."""
        notice = consume_restart_notice_from_env()
        if not notice:
            return
        target = self.channels.get(notice.channel)
        if not target:
            return
        asyncio.create_task(self._send_with_retry(
            target,
            OutboundMessage(
                channel=notice.channel,
                chat_id=notice.chat_id,
                content=format_restart_completed_message(notice.started_at_raw),
            ),
        ))

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("Stopping all channels...")

        # Stop dispatcher
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        # Stop per-channel workers
        for _name, task in self._channel_workers.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._channel_workers.clear()
        self._channel_queues.clear()

        # Stop all channels
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("Stopped {} channel", name)
            except Exception as e:
                logger.error("Error stopping {}: {}", name, e)

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to the appropriate channel."""
        logger.info("Outbound dispatcher started")

        # Buffer for messages that couldn't be processed during delta coalescing
        # (since asyncio.Queue doesn't support push_front)
        pending: list[OutboundMessage] = []

        while True:
            try:
                # First check pending buffer before waiting on queue
                if pending:
                    msg = pending.pop(0)
                else:
                    msg = await asyncio.wait_for(
                        self.bus.consume_outbound(),
                        timeout=1.0
                    )

                if msg.metadata.get("_progress"):
                    if msg.metadata.get("_tool_hint") and not self.config.channels.send_tool_hints:
                        if msg._delivery_future and not msg._delivery_future.done():
                            msg._delivery_future.set_result(None)
                        continue
                    if not msg.metadata.get("_tool_hint") and not self.config.channels.send_progress:
                        if msg._delivery_future and not msg._delivery_future.done():
                            msg._delivery_future.set_result(None)
                        continue

                if msg.metadata.get("_retry_wait"):
                    continue

                # Coalesce consecutive _stream_delta messages for the same (channel, chat_id)
                # to reduce API calls and improve streaming latency
                if msg.metadata.get("_stream_delta") and not msg.metadata.get("_stream_end"):
                    msg, extra_pending = self._coalesce_stream_deltas(msg)
                    pending.extend(extra_pending)

                # Spam guard: suppress near-identical messages to the same recipient
                if self.config.channels.spam_guard.enabled and self._is_spam(msg):
                    logger.warning(
                        "Spam guard: suppressed duplicate message to {}:{}",
                        msg.channel, msg.chat_id,
                    )
                    if msg._delivery_future and not msg._delivery_future.done():
                        msg._delivery_future.set_result(None)
                    continue

                channel = self.channels.get(msg.channel)
                if channel:
                    await self._enqueue_channel_send(msg.channel, channel, msg)
                else:
                    logger.warning("Unknown channel: {}", msg.channel)
                    if msg._delivery_future and not msg._delivery_future.done():
                        msg._delivery_future.set_exception(
                            ValueError(f"Unknown channel: {msg.channel}")
                        )

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    @staticmethod
    def _spam_dedup_key_part(content: str) -> str:
        """Normalized fingerprint that collapses numeric drift.

        Pre-fix the exact-content hash treated "May 18" and "May 19" as
        distinct, so a templated heartbeat message could spam every tick
        as its embedded date crept forward. Digits collapse to a single
        ``0`` so dates and amounts of the same template still alias, but
        non-numeric tokens around them remain distinct (so "$50 to Mom"
        and "$50 to Dad" do not collide).
        """
        digit_collapsed = _DIGIT_PAT.sub("0", content.lower())
        normalized = _NONALNUM_PAT.sub(" ", digit_collapsed).strip()
        if not normalized:
            normalized = content.strip().lower()
        return hashlib.sha256(
            normalized.encode("utf-8", errors="replace")
        ).hexdigest()[:12]

    def _is_spam(self, msg: OutboundMessage) -> bool:
        """Check if a message is a near-duplicate sent too many times recently.

        Streaming deltas and progress messages are exempt.
        """
        meta = msg.metadata or {}
        if meta.get("_stream_delta") or meta.get("_stream_end") or meta.get("_progress"):
            return False
        if not isinstance(msg.content, str) or not msg.content.strip():
            return False

        sg = self.config.channels.spam_guard
        tag = meta.get(TASK_TAG_META_KEY)
        key_part = f"task:{tag}" if tag else self._spam_dedup_key_part(msg.content)
        key = (msg.channel, msg.chat_id or "", key_part)
        now = time.monotonic()

        # Prune expired entries for this key
        if key in self._dedup_log:
            self._dedup_log[key] = [
                t for t in self._dedup_log[key] if now - t < sg.window_s
            ]
            if not self._dedup_log[key]:
                del self._dedup_log[key]

        # Periodic full cache sweep (time-based, not per-message)
        if now - self._dedup_last_cleanup > _CLEANUP_INTERVAL_S:
            self._dedup_last_cleanup = now
            expired = [
                k for k, v in self._dedup_log.items()
                if not v or now - v[-1] >= sg.window_s
            ]
            for k in expired:
                del self._dedup_log[k]

        timestamps = self._dedup_log.get(key, [])
        if len(timestamps) >= sg.max_repeats:
            return True

        self._dedup_log.setdefault(key, []).append(now)
        return False

    async def _enqueue_channel_send(
        self, channel_name: str, channel: BaseChannel, msg: OutboundMessage
    ) -> None:
        """Enqueue a message for per-channel delivery (non-blocking for other channels)."""
        if channel_name not in self._channel_queues:
            q: asyncio.Queue[tuple[BaseChannel, OutboundMessage]] = asyncio.Queue()
            self._channel_queues[channel_name] = q
            self._channel_workers[channel_name] = asyncio.create_task(
                self._channel_worker(channel_name, q)
            )
        await self._channel_queues[channel_name].put((channel, msg))

    async def _channel_worker(
        self, name: str, q: asyncio.Queue[tuple[BaseChannel, OutboundMessage]]
    ) -> None:
        """Process outbound messages for a single channel, preserving order."""
        while True:
            try:
                channel, msg = await q.get()
                await self._send_with_retry(channel, msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Channel worker {} error: {}", name, e)

    @staticmethod
    async def _send_once(channel: BaseChannel, msg: OutboundMessage) -> None:
        """Send one outbound message without retry policy."""
        _check_outbound_authorized(channel, msg)
        if msg.metadata.get("_stream_delta") or msg.metadata.get("_stream_end"):
            await channel.send_delta(msg.chat_id, msg.content, msg.metadata)
        elif not msg.metadata.get("_streamed"):
            await channel.send(msg)

    def _coalesce_stream_deltas(
        self, first_msg: OutboundMessage
    ) -> tuple[OutboundMessage, list[OutboundMessage]]:
        """Merge consecutive _stream_delta messages for the same (channel, chat_id).

        This reduces the number of API calls when the queue has accumulated multiple
        deltas, which happens when LLM generates faster than the channel can process.

        Returns:
            tuple of (merged_message, list_of_non_matching_messages)
        """
        target_key = (first_msg.channel, first_msg.chat_id)
        combined_content = first_msg.content
        final_metadata = dict(first_msg.metadata or {})
        non_matching: list[OutboundMessage] = []

        # Only merge consecutive deltas. As soon as we hit any other message,
        # stop and hand that boundary back to the dispatcher via `pending`.
        while True:
            try:
                next_msg = self.bus.outbound.get_nowait()
            except asyncio.QueueEmpty:
                break

            # Check if this message belongs to the same stream
            same_target = (next_msg.channel, next_msg.chat_id) == target_key
            is_delta = next_msg.metadata and next_msg.metadata.get("_stream_delta")
            is_end = next_msg.metadata and next_msg.metadata.get("_stream_end")

            if same_target and is_delta and not final_metadata.get("_stream_end"):
                # Accumulate content
                combined_content += next_msg.content
                # Resolve consumed message's delivery future (content merged into first_msg)
                if next_msg._delivery_future and not next_msg._delivery_future.done():
                    next_msg._delivery_future.set_result(None)
                # If we see _stream_end, remember it and stop coalescing this stream
                if is_end:
                    final_metadata["_stream_end"] = True
                    # Stream ended - stop coalescing this stream
                    break
            else:
                # First non-matching message defines the coalescing boundary.
                non_matching.append(next_msg)
                break

        merged = OutboundMessage(
            channel=first_msg.channel,
            chat_id=first_msg.chat_id,
            content=combined_content,
            metadata=final_metadata,
        )
        return merged, non_matching

    async def _send_with_retry(self, channel: BaseChannel, msg: OutboundMessage) -> None:
        """Send a message with retry on failure using exponential backoff.

        Note: CancelledError is re-raised to allow graceful shutdown.
        """
        max_attempts = max(self.config.channels.send_max_retries, 1)

        for attempt in range(max_attempts):
            try:
                await self._send_once(channel, msg)
                if msg._delivery_future and not msg._delivery_future.done():
                    msg._delivery_future.set_result(None)
                return  # Send succeeded
            except asyncio.CancelledError:
                raise  # Propagate cancellation for graceful shutdown
            except OutboundScopeError as e:
                # Refusal is permanent — retrying won't help. Surface the error
                # to the caller (typically the agent's MessageTool) so it can
                # show structured remediation to the LLM.
                logger.warning(
                    "scope_guard: refused send to {}:{} ({}): {}",
                    msg.channel, msg.chat_id, e.reason, e.remediation or "",
                )
                if msg._delivery_future and not msg._delivery_future.done():
                    msg._delivery_future.set_exception(e)
                return
            except Exception as e:
                _ = e
                if attempt == max_attempts - 1:
                    logger.error(
                        "Failed to send to {} after {} attempts: {} - {}",
                        msg.channel, max_attempts, type(e).__name__, e
                    )
                    if msg._delivery_future and not msg._delivery_future.done():
                        msg._delivery_future.set_exception(e)
                    return
                delay = _SEND_RETRY_DELAYS[min(attempt, len(_SEND_RETRY_DELAYS) - 1)]
                logger.warning(
                    "Send to {} failed (attempt {}/{}): {}, retrying in {}s",
                    msg.channel, attempt + 1, max_attempts, type(e).__name__, delay
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise  # Propagate cancellation during sleep

    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all channels."""
        return {
            name: {
                "enabled": True,
                "running": channel.is_running
            }
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels.keys())
