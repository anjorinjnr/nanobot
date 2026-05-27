"""Message tool for sending messages to users."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Iterator

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.context import ContextAware, RequestContext
from nanobot.agent.tools.path_utils import resolve_workspace_path
from nanobot.agent.tools.schema import ArraySchema, StringSchema, tool_parameters_schema
from nanobot.bus.events import TASK_TAG_META_KEY, OutboundMessage
from nanobot.channels.scope_guard import OutboundScopeError
from nanobot.config.paths import get_workspace_path


@tool_parameters(
    tool_parameters_schema(
        content=StringSchema(
            "Message content for proactive or cross-channel delivery. "
            "Do not use this for a normal reply in the current chat."
        ),
        channel=StringSchema(
            "Optional target channel for cross-channel/proactive delivery. "
            "Do not set this to the current runtime channel for a normal reply."
        ),
        chat_id=StringSchema(
            "Optional target chat/user ID for cross-channel/proactive delivery. "
            "On WebSocket/WebUI turns: omit chat_id to use the server's conversation id "
            "(never pass client_id values like anon-…). "
            "Do not set this to the current runtime chat for a normal reply."
        ),
        media=ArraySchema(
            StringSchema(""),
            description=(
                "Optional list of existing file paths to attach for proactive or cross-channel delivery. "
                "Do not use this to resend generate_image outputs in the current chat."
            ),
        ),
        buttons=ArraySchema(
            ArraySchema(StringSchema("Button label")),
            description="Optional: inline keyboard buttons as list of rows, each row is list of button labels.",
        ),
        required=["content"],
    )
)
class MessageTool(Tool, ContextAware):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
        workspace: str | Path | None = None,
        restrict_to_workspace: bool = False,
    ):
        self._send_callback = send_callback
        self._workspace = (
            Path(workspace).expanduser() if workspace is not None else get_workspace_path()
        )
        self._restrict_to_workspace = restrict_to_workspace
        self._default_channel: ContextVar[str] = ContextVar(
            "message_default_channel", default=default_channel
        )
        self._default_chat_id: ContextVar[str] = ContextVar(
            "message_default_chat_id", default=default_chat_id
        )
        self._default_message_id: ContextVar[str | None] = ContextVar(
            "message_default_message_id",
            default=default_message_id,
        )
        self._default_metadata: ContextVar[dict[str, Any]] = ContextVar(
            "message_default_metadata",
            default={},
        )
        self._sent_in_turn_var: ContextVar[bool] = ContextVar("message_sent_in_turn", default=False)
        self._allowed_channels: ContextVar[frozenset[str] | None] = ContextVar(
            "message_allowed_channels", default=None
        )
        # Hard guarantee that prevents the LLM from sending a message intended
        # for one conversation/recipient to a different chat_id by overriding
        # the tool argument. When set (non-None), every send must target a
        # (channel, chat_id) in the set; anything else is refused at the tool
        # layer, before reaching the channel or the scope guard.
        #
        # Set by:
        #   - agent loop, per interactive turn → just the inbound sender
        #   - heartbeat dispatcher, per (task, recipient) → just that recipient
        #
        # This is the regression net for 2026-05-27, where the heartbeat's
        # non-prompt-file path called on_execute() with no target= and the
        # LLM guessed a chat_id from memory (a guest's, in that case).
        self._allowed_recipients: ContextVar[
            frozenset[tuple[str, str]] | None
        ] = ContextVar("message_allowed_recipients", default=None)
        self._task_tag: ContextVar[str | None] = ContextVar(
            "message_task_tag", default=None,
        )
        self._turn_delivered_media_var: ContextVar[tuple[str, ...]] = ContextVar(
            "message_turn_delivered_media",
            default=(),
        )
        self._record_channel_delivery_var: ContextVar[bool] = ContextVar(
            "message_record_channel_delivery",
            default=False,
        )

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        send_callback = ctx.bus.publish_outbound if ctx.bus else None
        return cls(
            send_callback=send_callback,
            workspace=ctx.workspace,
            restrict_to_workspace=ctx.config.restrict_to_workspace,
        )

    def set_context(self, ctx: RequestContext) -> None:
        """Set the current message context."""
        self._default_channel.set(ctx.channel)
        self._default_chat_id.set(ctx.chat_id)
        self._default_message_id.set(ctx.message_id)
        self._default_metadata.set(dict(ctx.metadata or {}))

    @contextmanager
    def scoped(
        self,
        *,
        allowed_channels: Iterable[str] | None = None,
        allowed_recipients: Iterable[tuple[str, str]] | None = None,
        task_tag: str | None = None,
    ) -> Iterator[None]:
        """Scope routing constraints to a single ``with`` block.

        ``allowed_channels`` (when not ``None``) refuses ``message`` calls to
        any channel outside the set — used by the heartbeat path to honor a
        task's Recipients field at the channel level.

        ``allowed_recipients`` (when not ``None``) is the **hard guarantee**:
        refuses any send whose ``(channel, chat_id)`` is not in the set.
        Stricter than ``allowed_channels`` — pins both axes. Required for
        per-recipient heartbeat dispatch and interactive turns so the LLM
        cannot send a message intended for one user to a different chat_id
        by overriding the tool arguments.

        ``task_tag`` propagates into outbound metadata so ChannelManager's
        spam guard can dedup per (recipient, task) instead of per content
        hash. All three are ContextVar-scoped, so nested or concurrent
        scopes don't leak.
        """
        channel_token = (
            self._allowed_channels.set(frozenset(c.lower() for c in allowed_channels))
            if allowed_channels is not None
            else None
        )
        recipient_token = (
            self._allowed_recipients.set(
                frozenset((c.lower(), x) for (c, x) in allowed_recipients)
            )
            if allowed_recipients is not None
            else None
        )
        tag_token = self._task_tag.set(task_tag) if task_tag else None
        try:
            yield
        finally:
            if channel_token is not None:
                self._allowed_channels.reset(channel_token)
            if recipient_token is not None:
                self._allowed_recipients.reset(recipient_token)
            if tag_token is not None:
                self._task_tag.reset(tag_token)

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(
        self,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        """Reset per-turn send tracking and (optionally) pin the recipient.

        When ``channel`` and ``chat_id`` are both supplied, ``allowed_recipients``
        is pinned to that single ``(channel, chat_id)`` for the turn — any
        attempt by the agent to ``message`` a different recipient via tool
        argument override is refused at the tool layer. Without the pin
        (defaults), the gate is open and only the channel-level scope guard
        applies.

        The caller is responsible for invoking this at the start of every
        turn so a previous turn's pin can't leak into the next. Heartbeat
        dispatches use the contextmanager ``scoped(allowed_recipients=...)``
        instead, which auto-resets at the end of the with-block.
        """
        self._sent_in_turn = False
        self._turn_delivered_media_var.set(())
        if channel and chat_id:
            self._allowed_recipients.set(
                frozenset([(channel.lower(), chat_id)])
            )
        else:
            self._allowed_recipients.set(None)

    def turn_delivered_media_paths(self) -> list[str]:
        """Absolute paths attached via this tool to the active chat in the current turn."""
        return list(self._turn_delivered_media_var.get())

    def set_record_channel_delivery(self, active: bool):
        """Mark tool-sent messages as proactive channel deliveries."""
        return self._record_channel_delivery_var.set(active)

    def reset_record_channel_delivery(self, token) -> None:
        """Restore previous proactive delivery recording state."""
        self._record_channel_delivery_var.reset(token)

    @property
    def _sent_in_turn(self) -> bool:
        return self._sent_in_turn_var.get()

    @_sent_in_turn.setter
    def _sent_in_turn(self, value: bool) -> None:
        self._sent_in_turn_var.set(value)

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return (
            "Proactively send a message to a user/channel, optionally with file attachments. "
            "Use this for reminders, cross-channel delivery, or explicit proactive sends. "
            "Do not use this for the normal reply in the current chat: answer naturally instead. "
            "If channel/chat_id would target the current runtime conversation, do not call this tool "
            "unless the user explicitly asked you to proactively send an existing file attachment. "
            "When generate_image creates images in the current chat, the final assistant reply "
            "automatically attaches them; do not call message just to announce or resend them. "
            "For proactive attachment delivery, use the 'media' parameter with file paths. "
            "Do NOT use read_file to send files — that only reads content for your own analysis."
        )

    def _resolve_media(self, media: list[str]) -> list[str]:
        """Resolve local media attachments and enforce workspace restriction when enabled."""
        resolved: list[str] = []
        allowed_dir = self._workspace if self._restrict_to_workspace else None
        for p in media:
            if p.startswith(("http://", "https://")):
                resolved.append(p)
            elif not self._restrict_to_workspace:
                path = Path(p).expanduser()
                resolved.append(p if path.is_absolute() else str(self._workspace / path))
            else:
                resolved.append(str(resolve_workspace_path(p, self._workspace, allowed_dir)))
        return resolved

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        buttons: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> str:
        from nanobot.utils.helpers import strip_think

        content = strip_think(content)

        if buttons is not None:
            if not isinstance(buttons, list) or any(
                not isinstance(row, list) or any(not isinstance(label, str) for label in row)
                for row in buttons
            ):
                return "Error: buttons must be a list of list of strings"
        default_channel = self._default_channel.get()
        default_chat_id = self._default_chat_id.get()
        channel = channel or default_channel
        explicit_chat_id = chat_id
        if (
            default_channel == "websocket"
            and channel == "websocket"
            and explicit_chat_id is not None
            and str(explicit_chat_id).strip() != ""
            and str(explicit_chat_id).strip() != str(default_chat_id).strip()
        ):
            return (
                "Error: chat_id does not match the active WebSocket conversation. "
                "Omit chat_id (and usually channel) so delivery uses the current "
                "conversation id from context — WebSocket client_id strings "
                "(e.g. anon-…) are not chat ids."
            )
        chat_id = chat_id or default_chat_id
        # Only inherit default message_id when targeting the same channel+chat.
        # Cross-chat sends must not carry the original message_id, because
        # some channels (e.g. Feishu) use it to determine the target
        # conversation via their Reply API, which would route the message
        # to the wrong chat entirely.
        same_target = channel == default_channel and chat_id == default_chat_id
        if same_target:
            message_id = message_id or self._default_message_id.get()
        else:
            message_id = None

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        allowed = self._allowed_channels.get()
        if allowed is not None and channel.lower() not in allowed:
            allowed_list = ", ".join(sorted(allowed)) or "none"
            return (
                f"Error: channel {channel!r} is not permitted in this context. "
                f"Allowed channels: {allowed_list}."
            )

        # Hard recipient gate. When the caller set allowed_recipients (per-turn
        # for interactive replies, per (task, recipient) for heartbeat
        # dispatch), every send must match. Refuse anything else — this is
        # the kernel-level guarantee that prevents the LLM from sending a
        # message intended for one user to a different chat_id by guessing
        # or overriding the tool args. Returning a string error here means
        # the agent sees the refusal in its tool result; the message never
        # reaches the channel.
        allowed_targets = self._allowed_recipients.get()
        if allowed_targets is not None:
            if (channel.lower(), chat_id) not in allowed_targets:
                allowed_repr = (
                    ", ".join(f"{c}:{x}" for c, x in sorted(allowed_targets))
                    or "<empty>"
                )
                return (
                    f"Error: recipient {channel}:{chat_id} is not permitted "
                    f"in this context. Allowed: {allowed_repr}. "
                    "The agent cannot override recipient mid-turn; this turn "
                    "is scoped to a specific user/conversation."
                )

        if not self._send_callback:
            return "Error: Message sending not configured"

        if media:
            try:
                media = self._resolve_media(media)
            except (OSError, PermissionError, ValueError) as e:
                return f"Error: media path is not allowed: {str(e)}"

        loop = asyncio.get_running_loop()
        delivery_future: asyncio.Future[None] = loop.create_future()

        metadata = dict(self._default_metadata.get()) if same_target else {}
        if message_id:
            metadata["message_id"] = message_id
        if (tag := self._task_tag.get()):
            metadata[TASK_TAG_META_KEY] = tag
        if self._record_channel_delivery_var.get() or media:
            metadata["_record_channel_delivery"] = True

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            buttons=buttons or [],
            metadata=metadata,
            _delivery_future=delivery_future,
        )

        try:
            await self._send_callback(msg)
            # Wait for actual delivery confirmation from the dispatcher
            await asyncio.wait_for(delivery_future, timeout=60.0)
            if channel == default_channel and chat_id == default_chat_id:
                self._sent_in_turn = True
                if media:
                    prev = self._turn_delivered_media_var.get()
                    self._turn_delivered_media_var.set(prev + tuple(str(p) for p in media))
            media_info = f" with {len(media)} attachments" if media else ""
            button_info = f" with {sum(len(row) for row in buttons)} button(s)" if buttons else ""
            return f"Message sent to {channel}:{chat_id}{media_info}{button_info}"
        except asyncio.TimeoutError:
            # Delivery timed out — message may or may not have been sent
            if channel == default_channel and chat_id == default_chat_id:
                self._sent_in_turn = True
            return f"Message delivery timed out for {channel}:{chat_id} — it may not have been delivered"
        except OutboundScopeError as e:
            # Surface the structured remediation so the LLM creates a scope
            # before retrying instead of looping or fabricating a purpose.
            return str(e)
        except Exception as e:
            return f"Error sending message: {str(e)}"
