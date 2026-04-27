from typing import Any

import pytest

from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import OutboundMessage


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


def _capture_send_callback() -> tuple[Any, list[OutboundMessage]]:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    return _send, sent


@pytest.mark.asyncio
async def test_allowed_channels_blocks_disallowed_channel() -> None:
    # The heartbeat path clamps MessageTool to the channels listed in a task's
    # Recipients field so the LLM can't fan out to email/telegram when a task
    # asked for whatsapp only — the prod Balance-check failure mode.
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    token = tool.set_allowed_channels({"whatsapp"})
    try:
        result = await tool.execute(content="hi", channel="email", chat_id="x@y")
    finally:
        tool.reset_allowed_channels(token)
    assert "not permitted" in result
    assert sent == []


@pytest.mark.asyncio
async def test_allowed_channels_permits_listed_channel() -> None:
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    token = tool.set_allowed_channels({"whatsapp"})
    try:
        result = await tool.execute(content="hi", channel="WhatsApp", chat_id="123")
    finally:
        tool.reset_allowed_channels(token)
    assert result.startswith("Message sent")
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_task_tag_propagates_to_outbound_metadata() -> None:
    # Spam guard relies on the _task_tag metadata to dedup heartbeat repeats
    # per (recipient, task) instead of per content hash.
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    token = tool.set_task_tag("Balance check")
    try:
        await tool.execute(content="hi", channel="whatsapp", chat_id="123")
    finally:
        tool.reset_task_tag(token)
    assert sent[0].metadata.get("_task_tag") == "Balance check"


@pytest.mark.asyncio
async def test_no_allow_list_means_no_restriction() -> None:
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    result = await tool.execute(content="hi", channel="email", chat_id="x@y")
    assert result.startswith("Message sent")
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "not a list",
        [["ok"], "row-not-a-list"],
        [["ok", 42]],
        [[None]],
    ],
)
async def test_message_tool_rejects_malformed_buttons(bad) -> None:
    """``buttons`` must be ``list[list[str]]``; the tool validates the shape
    up front so a malformed LLM payload errors visibly instead of slipping
    into the channel layer where Telegram would silently reject the frame."""
    tool = MessageTool()
    result = await tool.execute(
        content="hi", channel="telegram", chat_id="1", buttons=bad,
    )
    assert result == "Error: buttons must be a list of list of strings"
