from typing import Any

import pytest

from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import TASK_TAG_META_KEY, OutboundMessage


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
async def test_scoped_blocks_disallowed_channel() -> None:
    # Heartbeat clamps MessageTool to the task's Recipients channels so the
    # LLM can't fan out to email/telegram when a task asked for whatsapp.
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    with tool.scoped(allowed_channels={"whatsapp"}):
        result = await tool.execute(content="hi", channel="email", chat_id="x@y")
    assert "not permitted" in result
    assert sent == []


@pytest.mark.asyncio
async def test_scoped_permits_listed_channel_case_insensitive() -> None:
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    with tool.scoped(allowed_channels={"whatsapp"}):
        result = await tool.execute(content="hi", channel="WhatsApp", chat_id="123")
    assert result.startswith("Message sent")
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_scoped_task_tag_propagates_to_outbound_metadata() -> None:
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    with tool.scoped(task_tag="Balance check"):
        await tool.execute(content="hi", channel="whatsapp", chat_id="123")
    assert sent[0].metadata.get(TASK_TAG_META_KEY) == "Balance check"


@pytest.mark.asyncio
async def test_scoped_releases_on_exit() -> None:
    # Exit unblocks the channel and clears the tag — verifies finally runs.
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    with tool.scoped(allowed_channels={"whatsapp"}, task_tag="X"):
        pass
    result = await tool.execute(content="hi", channel="email", chat_id="x@y")
    assert result.startswith("Message sent")
    assert TASK_TAG_META_KEY not in sent[0].metadata


@pytest.mark.asyncio
async def test_no_scope_means_no_restriction() -> None:
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    result = await tool.execute(content="hi", channel="email", chat_id="x@y")
    assert result.startswith("Message sent")
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_heartbeat_context_refuses_bare_call_without_chat_id() -> None:
    """During heartbeat dispatch (task_tag set by heartbeat_execute_context),
    the default chat_id points at a synthetic 'user' placeholder. Falling back
    to it silently routes the send to nowhere. The tool must refuse and prompt
    the agent to pass chat_id explicitly from the task's Recipients."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    # Even with defaults set (as would happen during chat-mode), the heartbeat
    # task_tag must override and require explicit chat_id.
    tool.set_context(channel="whatsapp", chat_id="user")
    with tool.scoped(task_tag="Morning briefing"):
        result = await tool.execute(content="Good morning!")
    assert result.startswith("Error: chat_id is required")
    assert "Recipients" in result
    assert sent == []


@pytest.mark.asyncio
async def test_heartbeat_context_refuses_bare_call_without_channel() -> None:
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    tool.set_context(channel="whatsapp", chat_id="user")
    with tool.scoped(task_tag="Morning briefing"):
        # chat_id passed but channel omitted — still must refuse.
        result = await tool.execute(content="hi", chat_id="246157477413033@lid")
    assert result.startswith("Error: channel is required")
    assert sent == []


@pytest.mark.asyncio
async def test_heartbeat_context_with_full_routing_args_succeeds() -> None:
    """When the agent does the right thing — passes both chat_id and channel
    explicitly — the message delivers normally even with the heartbeat tag set."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    tool.set_context(channel="whatsapp", chat_id="user")
    with tool.scoped(task_tag="Morning briefing"):
        result = await tool.execute(
            content="Good morning Ebby!",
            chat_id="246157477413033@lid",
            channel="whatsapp",
        )
    assert result.startswith("Message sent")
    assert len(sent) == 1
    assert sent[0].chat_id == "246157477413033@lid"


@pytest.mark.asyncio
async def test_chat_mode_still_falls_back_to_defaults() -> None:
    """No heartbeat context (no task_tag) — bare call falls back to the
    current chat-session defaults, the way real user replies have always
    worked. The new guard must not regress this path."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    tool.set_context(channel="whatsapp", chat_id="14155551234")
    result = await tool.execute(content="Sure, here you go")
    assert result.startswith("Message sent")
    assert sent[0].chat_id == "14155551234"


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
