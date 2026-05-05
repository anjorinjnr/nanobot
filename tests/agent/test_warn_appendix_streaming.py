"""Regression tests for nanobot issue #58.

When the post-turn quota-warn appendix would be appended AND the turn was
streamed (channel already saw the live deltas), the appendix must be emitted
as a SEPARATE non-streamed OutboundMessage on the bus — otherwise the channel
dispatcher (``ChannelManager._send_once``) skips the returned message because
``_streamed=True`` and the user never sees the warn copy.

For non-streamed turns the existing single-message UX is preserved: the
appendix is concatenated into ``final_content`` and no extra bus message is
emitted.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.analytics.quota_gate import WARN_APPENDIX
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path: Path) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")


def _wire_provider(loop: AgentLoop, content: str) -> None:
    loop.provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content=content, tool_calls=[])
    )

    async def _fake_stream(*_a, on_content_delta=None, **_kw):
        if on_content_delta is not None:
            await on_content_delta(content)
        return LLMResponse(content=content, tool_calls=[])

    loop.provider.chat_stream_with_retry = AsyncMock(side_effect=_fake_stream)
    loop.tools.get_definitions = MagicMock(return_value=[])


def _drain_outbound(loop: AgentLoop) -> list[OutboundMessage]:
    msgs: list[OutboundMessage] = []
    while True:
        try:
            msgs.append(loop.bus.outbound.get_nowait())
        except Exception:
            break
    return msgs


def _patch_pre_turn(monkeypatch, *, warn_pct: int | None) -> None:
    """Replace the pre-turn gate so it sets ``quota_warn_pct`` deterministically."""

    def _stub(turn_ctx):
        if warn_pct is not None:
            turn_ctx["quota_warn_pct"] = warn_pct
        return None

    # ``_process_message`` does ``from nanobot.analytics.quota_gate import
    # check_token_budget_before_turn`` lazily, so the lookup hits the source
    # module attribute at call time — patching here is sufficient.
    monkeypatch.setattr(
        "nanobot.analytics.quota_gate.check_token_budget_before_turn", _stub
    )


@pytest.mark.asyncio
async def test_warn_appendix_emitted_as_separate_message_when_streamed(
    tmp_path: Path, monkeypatch
) -> None:
    """Streamed turn + warn flag → returned content unchanged, appendix on bus."""
    loop = _make_loop(tmp_path)
    _wire_provider(loop, "Hello world.")
    _patch_pre_turn(monkeypatch, warn_pct=87)

    msg = InboundMessage(
        channel="telegram", sender_id="user1", chat_id="chat123", content="Hi"
    )

    async def _on_stream(_delta: str) -> None:
        pass

    result = await loop._process_message(msg, on_stream=_on_stream)

    # Returned content stays clean — the dispatcher will skip this message
    # (``_streamed=True``), and the user already saw it live anyway.
    assert result is not None
    assert result.metadata.get("_streamed") is True
    assert WARN_APPENDIX.format(pct=87) not in result.content
    assert result.content == "Hello world."

    # The appendix must have been published as a separate, non-streamed
    # outbound message so the channel actually delivers it.
    extras = _drain_outbound(loop)
    assert len(extras) == 1, f"expected 1 follow-up, got {len(extras)}"
    follow_up = extras[0]
    assert follow_up.channel == "telegram"
    assert follow_up.chat_id == "chat123"
    # Stripped of leading whitespace so it stands as its own message.
    assert follow_up.content == WARN_APPENDIX.format(pct=87).lstrip()
    # Must NOT carry the _streamed flag — otherwise dispatcher skips it again.
    assert not follow_up.metadata.get("_streamed")
    # Stream-delta / stream-end flags must also be absent so the dispatcher
    # routes the message through the normal ``send`` path.
    assert not follow_up.metadata.get("_stream_delta")
    assert not follow_up.metadata.get("_stream_end")


@pytest.mark.asyncio
async def test_warn_appendix_inlined_when_not_streamed(
    tmp_path: Path, monkeypatch
) -> None:
    """Non-streamed turn keeps the single-message UX (appendix concatenated)."""
    loop = _make_loop(tmp_path)
    _wire_provider(loop, "Hello world.")
    _patch_pre_turn(monkeypatch, warn_pct=87)

    msg = InboundMessage(
        channel="cli", sender_id="user1", chat_id="chat123", content="Hi"
    )

    result = await loop._process_message(msg)  # no on_stream

    assert result is not None
    assert not result.metadata.get("_streamed")
    assert result.content.startswith("Hello world.")
    assert WARN_APPENDIX.format(pct=87) in result.content
    # Bus stays empty — no extra follow-up.
    assert _drain_outbound(loop) == []


@pytest.mark.asyncio
async def test_no_appendix_no_extra_message(
    tmp_path: Path, monkeypatch
) -> None:
    """Streamed turn with no warn flag: nothing extra on the bus."""
    loop = _make_loop(tmp_path)
    _wire_provider(loop, "Hello world.")
    _patch_pre_turn(monkeypatch, warn_pct=None)

    msg = InboundMessage(
        channel="telegram", sender_id="user1", chat_id="chat123", content="Hi"
    )

    async def _on_stream(_delta: str) -> None:
        pass

    result = await loop._process_message(msg, on_stream=_on_stream)
    assert result is not None
    assert result.metadata.get("_streamed") is True
    assert result.content == "Hello world."
    assert _drain_outbound(loop) == []
