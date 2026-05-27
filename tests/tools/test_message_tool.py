import os
from typing import Any

import pytest

from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import TASK_TAG_META_KEY, OutboundMessage
from nanobot.config.paths import get_workspace_path


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


# allowed_recipients — the hard recipient gate. Stricter than allowed_channels;
# pins the (channel, chat_id) pair so the LLM cannot override chat_id by tool
# argument. Regression net for the 2026-05-27 leak where the heartbeat agent
# put a guest's chat_id in a `message(...)` call meant for primary.

@pytest.mark.asyncio
async def test_scoped_blocks_disallowed_recipient() -> None:
    """A send to a chat_id not in allowed_recipients is refused at the tool
    layer — the message never reaches the channel and the agent sees the
    refusal in its tool result."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    with tool.scoped(allowed_recipients={("whatsapp", "primary_lid")}):
        result = await tool.execute(
            content="leaked!", channel="whatsapp", chat_id="guest_lid",
        )
    assert "is not permitted" in result
    assert "guest_lid" in result
    assert sent == []


@pytest.mark.asyncio
async def test_scoped_allows_listed_recipient() -> None:
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    with tool.scoped(allowed_recipients={("whatsapp", "primary_lid")}):
        result = await tool.execute(
            content="hi", channel="whatsapp", chat_id="primary_lid",
        )
    assert result.startswith("Message sent")
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_scoped_recipient_check_normalizes_channel_case() -> None:
    """Channel matching is case-insensitive (consistent with allowed_channels)
    so 'WhatsApp' in the allow-set matches a 'whatsapp' send."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    with tool.scoped(allowed_recipients={("WhatsApp", "primary_lid")}):
        result = await tool.execute(
            content="hi", channel="whatsapp", chat_id="primary_lid",
        )
    assert result.startswith("Message sent")


@pytest.mark.asyncio
async def test_scoped_recipient_chat_id_is_exact_match() -> None:
    """chat_ids are opaque tokens — no substring or normalization. A
    `chat_id` that differs by a single character is rejected."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    with tool.scoped(allowed_recipients={("whatsapp", "primary_lid")}):
        result = await tool.execute(
            content="hi", channel="whatsapp", chat_id="primary_lid ",  # trailing space
        )
    assert "is not permitted" in result
    assert sent == []


@pytest.mark.asyncio
async def test_scoped_recipient_releases_on_exit() -> None:
    """After the `with` block, the recipient pin is released — verifies the
    ContextVar reset runs."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    with tool.scoped(allowed_recipients={("whatsapp", "primary_lid")}):
        pass
    result = await tool.execute(
        content="hi", channel="whatsapp", chat_id="other",
    )
    assert result.startswith("Message sent")


@pytest.mark.asyncio
async def test_start_turn_pins_recipient_for_interactive_turn() -> None:
    """The agent loop calls start_turn(channel, chat_id) per turn to lock the
    `message` tool to the inbound sender. Any attempt to message a different
    chat_id mid-turn (e.g., the LLM hallucinating another user's LID) is
    refused."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    tool.start_turn(channel="whatsapp", chat_id="ebby_lid")

    bad = await tool.execute(content="hi", channel="whatsapp", chat_id="emeka_lid")
    assert "is not permitted" in bad
    assert sent == []

    good = await tool.execute(content="hi", channel="whatsapp", chat_id="ebby_lid")
    assert good.startswith("Message sent")
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_start_turn_without_args_opens_the_gate() -> None:
    """Backward compatibility: start_turn() with no args resets the per-turn
    pin to None, so callers that don't yet pass channel/chat_id behave as
    before (the channel-level scope guard is the only constraint)."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    tool.start_turn(channel="whatsapp", chat_id="ebby_lid")
    # Now reset: a new turn with no pin.
    tool.start_turn()
    result = await tool.execute(content="hi", channel="whatsapp", chat_id="anyone")
    assert result.startswith("Message sent")


@pytest.mark.asyncio
async def test_start_turn_resets_previous_pin() -> None:
    """If a turn pinned to ebby ends and the next turn pins to seun, the
    seun pin must not leak ebby into the allowed set."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    tool.start_turn(channel="whatsapp", chat_id="ebby_lid")
    tool.start_turn(channel="whatsapp", chat_id="seun_lid")
    bad = await tool.execute(content="hi", channel="whatsapp", chat_id="ebby_lid")
    assert "is not permitted" in bad


@pytest.mark.asyncio
async def test_recipient_gate_passes_when_inside_allowed_channel_but_not_recipient() -> None:
    """The two gates are independent. A send can pass the channel gate
    (whatsapp is allowed) but still fail the recipient gate (chat_id is not).
    Recipient is the stricter, final gate."""
    send, sent = _capture_send_callback()
    tool = MessageTool(send_callback=send)
    with tool.scoped(
        allowed_channels={"whatsapp"},
        allowed_recipients={("whatsapp", "primary_lid")},
    ):
        result = await tool.execute(
            content="leaked!", channel="whatsapp", chat_id="guest_lid",
        )
    assert "is not permitted" in result
    assert "guest_lid" in result
    assert sent == []


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
        content="hi",
        channel="telegram",
        chat_id="1",
        buttons=bad,
    )
    assert result == "Error: buttons must be a list of list of strings"


@pytest.mark.asyncio
async def test_message_tool_marks_channel_delivery_only_when_enabled() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)

    await tool.execute(content="normal", channel="telegram", chat_id="1")
    token = tool.set_record_channel_delivery(True)
    try:
        await tool.execute(content="cron", channel="telegram", chat_id="1")
    finally:
        tool.reset_record_channel_delivery(token)

    assert sent[0].metadata == {}
    assert sent[1].metadata == {"_record_channel_delivery": True}


@pytest.mark.asyncio
async def test_message_tool_records_media_deliveries() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)

    await tool.execute(
        content="image",
        channel="websocket",
        chat_id="chat-1",
        media=["/tmp/generated.png"],
    )

    assert sent[0].metadata == {"_record_channel_delivery": True}


@pytest.mark.asyncio
async def test_message_tool_inherits_metadata_for_same_target() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)
    slack_meta = {"slack": {"thread_ts": "111.222", "channel_type": "channel"}}
    from nanobot.agent.tools.context import RequestContext

    tool.set_context(RequestContext(channel="slack", chat_id="C123", metadata=slack_meta))

    await tool.execute(content="thread reply")

    assert sent[0].metadata == slack_meta


@pytest.mark.asyncio
async def test_message_tool_clears_metadata_when_context_has_none() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)
    from nanobot.agent.tools.context import RequestContext

    tool.set_context(
        RequestContext(
            channel="slack",
            chat_id="C123",
            metadata={"slack": {"thread_ts": "111.222", "channel_type": "channel"}},
        ),
    )
    tool.set_context(RequestContext(channel="slack", chat_id="C123", metadata={}))

    await tool.execute(content="plain reply")

    assert sent[0].metadata == {}


@pytest.mark.asyncio
async def test_message_tool_does_not_inherit_metadata_for_cross_target() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)
    from nanobot.agent.tools.context import RequestContext

    tool.set_context(
        RequestContext(
            channel="slack",
            chat_id="C123",
            metadata={"slack": {"thread_ts": "111.222", "channel_type": "channel"}},
        ),
    )

    await tool.execute(content="channel reply", channel="slack", chat_id="C999")

    assert sent[0].metadata == {}


@pytest.mark.asyncio
async def test_message_tool_resolves_relative_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=["output/image.png"],
    )

    expected = str(get_workspace_path() / "output/image.png")
    assert sent[0].media == [expected]


@pytest.mark.asyncio
async def test_message_tool_resolves_relative_media_paths_from_active_workspace(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    workspace = tmp_path / "workspace"
    tool = MessageTool(send_callback=_send, workspace=workspace)

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=["output/image.png"],
    )

    assert sent[0].media == [str(workspace / "output/image.png")]


@pytest.mark.asyncio
async def test_message_tool_rejects_outside_workspace_absolute_media_when_restricted(
    tmp_path,
) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    tool = MessageTool(send_callback=_send, workspace=workspace, restrict_to_workspace=True)

    result = await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[str(outside)],
    )

    assert result.startswith("Error: media path is not allowed:")
    assert "outside allowed directory" in result
    assert sent == []


@pytest.mark.asyncio
async def test_message_tool_allows_workspace_absolute_media_when_restricted(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "image.png"
    image.write_text("image", encoding="utf-8")
    tool = MessageTool(send_callback=_send, workspace=workspace, restrict_to_workspace=True)

    result = await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[str(image)],
    )

    assert result == "Message sent to telegram:1 with 1 attachments"
    assert sent[0].media == [str(image.resolve())]


@pytest.mark.asyncio
async def test_message_tool_passes_through_absolute_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)

    abs_path = os.path.abspath(os.path.join(os.sep, "tmp", "abs_image.png"))

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[abs_path],
    )

    assert sent[0].media == [abs_path]


@pytest.mark.asyncio
async def test_message_tool_passes_through_url_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)

    url = "https://example.com/image.png"

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[url],
    )

    assert sent[0].media == [url]


@pytest.mark.asyncio
async def test_message_tool_resolves_mixed_media_paths() -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)

    abs_path = os.path.abspath(os.path.join(os.sep, "tmp", "absolute.png"))

    await tool.execute(
        content="see attached",
        channel="telegram",
        chat_id="1",
        media=[
            "output/relative.png",
            abs_path,
            "https://example.com/url.png",
            "http://example.com/http.png",
        ],
    )

    expected_relative = str(get_workspace_path() / "output/relative.png")
    assert sent[0].media == [
        expected_relative,
        abs_path,
        "https://example.com/url.png",
        "http://example.com/http.png",
    ]


@pytest.mark.asyncio
async def test_message_tool_tracks_turn_media_for_same_target(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)
    from nanobot.agent.tools.context import RequestContext

    tool.set_context(RequestContext(channel="websocket", chat_id="chat-1", metadata={}))
    tool.start_turn()
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    await tool.execute(content="see file", channel="websocket", chat_id="chat-1", media=[str(f)])

    assert tool.turn_delivered_media_paths() == [str(f.resolve())]


@pytest.mark.asyncio
async def test_message_tool_start_turn_clears_tracked_media(tmp_path) -> None:
    async def _send(msg: OutboundMessage) -> None:
        pass

    tool = MessageTool(send_callback=_send)
    from nanobot.agent.tools.context import RequestContext

    tool.set_context(RequestContext(channel="websocket", chat_id="chat-1", metadata={}))
    tool.start_turn()
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    await tool.execute(content="see file", media=[str(f)])
    tool.start_turn()
    assert tool.turn_delivered_media_paths() == []


@pytest.mark.asyncio
async def test_message_tool_cross_target_does_not_track_turn_media(tmp_path) -> None:
    async def _send(msg: OutboundMessage) -> None:
        pass

    tool = MessageTool(send_callback=_send)
    from nanobot.agent.tools.context import RequestContext

    tool.set_context(RequestContext(channel="websocket", chat_id="chat-1", metadata={}))
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    await tool.execute(
        content="see file",
        channel="telegram",
        chat_id="tg-other",
        media=[str(f)],
    )
    assert tool.turn_delivered_media_paths() == []


@pytest.mark.asyncio
async def test_message_tool_rejects_wrong_explicit_ws_chat_id(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)
    from nanobot.agent.tools.context import RequestContext

    conv = "550e8400-e29b-41d4-a716-446655440000"
    tool.set_context(RequestContext(channel="websocket", chat_id=conv, metadata={}))
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    result = await tool.execute(
        content="see file",
        channel="websocket",
        chat_id="anon-deadbeefcafe",
        media=[str(f)],
    )
    assert result.startswith("Error: chat_id does not match")
    assert sent == []


@pytest.mark.asyncio
async def test_message_tool_allows_ws_explicit_when_matches_context(tmp_path) -> None:
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)
    from nanobot.agent.tools.context import RequestContext

    conv = "550e8400-e29b-41d4-a716-446655440000"
    tool.set_context(RequestContext(channel="websocket", chat_id=conv, metadata={}))
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    result = await tool.execute(
        content="see file",
        channel="websocket",
        chat_id=conv,
        media=[str(f)],
    )
    assert result.startswith("Message sent")
    assert sent[0].chat_id == conv


@pytest.mark.asyncio
async def test_message_tool_cli_context_may_target_other_ws_chat(tmp_path) -> None:
    """Cron / CLI handlers keep non-websocket defaults; explicit websocket + uuid remains valid."""
    sent: list[OutboundMessage] = []

    async def _send(msg: OutboundMessage) -> None:
        sent.append(msg)
        if msg._delivery_future and not msg._delivery_future.done():
            msg._delivery_future.set_result(None)

    tool = MessageTool(send_callback=_send)
    from nanobot.agent.tools.context import RequestContext

    target = "550e8400-e29b-41d4-a716-446655440000"
    tool.set_context(RequestContext(channel="cli", chat_id="direct", metadata={}))
    f = tmp_path / "doc.md"
    f.write_text("hello", encoding="utf-8")
    result = await tool.execute(
        content="ping",
        channel="websocket",
        chat_id=target,
        media=[str(f)],
    )
    assert result.startswith("Message sent")
    assert sent[0].channel == "websocket"
    assert sent[0].chat_id == target
