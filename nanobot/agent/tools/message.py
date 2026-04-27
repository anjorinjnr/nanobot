"""Message tool for sending messages to users."""

import asyncio
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import ArraySchema, StringSchema, tool_parameters_schema
from nanobot.bus.events import OutboundMessage


@tool_parameters(
    tool_parameters_schema(
        content=StringSchema("The message content to send"),
        channel=StringSchema("Optional: target channel (telegram, discord, etc.)"),
        chat_id=StringSchema("Optional: target chat/user ID"),
        media=ArraySchema(
            StringSchema(""),
            description="Optional: list of file paths to attach (images, audio, documents)",
        ),
        buttons=ArraySchema(
            ArraySchema(StringSchema("Button label")),
            description="Optional: inline keyboard buttons as list of rows, each row is list of button labels.",
        ),
        required=["content"],
    )
)
class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel: ContextVar[str] = ContextVar("message_default_channel", default=default_channel)
        self._default_chat_id: ContextVar[str] = ContextVar("message_default_chat_id", default=default_chat_id)
        self._default_message_id: ContextVar[str | None] = ContextVar(
            "message_default_message_id",
            default=default_message_id,
        )
        self._sent_in_turn_var: ContextVar[bool] = ContextVar("message_sent_in_turn", default=False)
        # Heartbeat path sets these per-tick so the model can't message channels
        # outside the task's Recipients (the LLM has been observed rotating
        # whatsapp/email/telegram while Recipients only listed whatsapp), and so
        # the spam guard can dedup per-task instead of per-content-hash.
        self._allowed_channels: ContextVar[frozenset[str] | None] = ContextVar(
            "message_allowed_channels", default=None
        )
        self._task_tag: ContextVar[str | None] = ContextVar(
            "message_task_tag", default=None
        )

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel.set(channel)
        self._default_chat_id.set(chat_id)
        self._default_message_id.set(message_id)

    def set_allowed_channels(self, channels: set[str] | frozenset[str] | None):
        """Restrict this tool to a fixed set of channels for the current ContextVar scope.

        ``None`` (default) means no restriction. Pass an empty set to block all
        sends. Returns the token from ContextVar.set so the caller can ``reset``.
        """
        return self._allowed_channels.set(
            frozenset(c.lower() for c in channels) if channels is not None else None
        )

    def reset_allowed_channels(self, token) -> None:
        self._allowed_channels.reset(token)

    def set_task_tag(self, tag: str | None):
        """Tag outgoing messages with a stable task identifier for spam-guard keying."""
        return self._task_tag.set(tag)

    def reset_task_tag(self, token) -> None:
        self._task_tag.reset(token)

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False

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
            "Send a message to the user, optionally with file attachments. "
            "This is the ONLY way to deliver files (images, documents, audio, video) to the user. "
            "Use the 'media' parameter with file paths to attach files. "
            "Do NOT use read_file to send files — that only reads content for your own analysis."
        )

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        buttons: list[list[str]] | None = None,
        **kwargs: Any
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
        chat_id = chat_id or default_chat_id
        # Only inherit default message_id when targeting the same channel+chat.
        # Cross-chat sends must not carry the original message_id, because
        # some channels (e.g. Feishu) use it to determine the target
        # conversation via their Reply API, which would route the message
        # to the wrong chat entirely.
        if channel == default_channel and chat_id == default_chat_id:
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

        if not self._send_callback:
            return "Error: Message sending not configured"

        loop = asyncio.get_running_loop()
        delivery_future: asyncio.Future[None] = loop.create_future()

        metadata: dict[str, Any] = {}
        if message_id:
            metadata["message_id"] = message_id
        if (tag := self._task_tag.get()):
            metadata["_task_tag"] = tag

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
            media_info = f" with {len(media)} attachments" if media else ""
            button_info = f" with {sum(len(row) for row in buttons)} button(s)" if buttons else ""
            return f"Message sent to {channel}:{chat_id}{media_info}{button_info}"
        except asyncio.TimeoutError:
            # Delivery timed out — message may or may not have been sent
            if channel == default_channel and chat_id == default_chat_id:
                self._sent_in_turn = True
            return f"Message delivery timed out for {channel}:{chat_id} — it may not have been delivered"
        except Exception as e:
            return f"Error sending message: {str(e)}"
