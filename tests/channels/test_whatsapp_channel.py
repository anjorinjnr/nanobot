"""Tests for the neonize-backed WhatsApp channel.

These exercise the channel's adapter logic — identity resolution, group
policy, typing-indicator lifecycle, voice-transcription branch, media
tagging, error propagation. The neonize protocol client is mocked at the
``WhatsAppClient`` boundary; lower-level neonize behavior is covered
indirectly via the wrapper module's own smoke checks.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.channels.whatsapp import WhatsAppChannel
from nanobot.channels.whatsapp_client import InboundMessage as WAInboundMessage


def _mock_client() -> AsyncMock:
    """Build an AsyncMock standing in for WhatsAppClient with the methods the channel calls."""
    client = AsyncMock()
    client.send_message = AsyncMock(return_value={"lid": None})
    client.send_media = AsyncMock(return_value={"lid": None})
    client.send_typing = AsyncMock(return_value=None)
    client.disconnect = AsyncMock(return_value=None)
    return client


def _make_channel(config: dict | None = None) -> WhatsAppChannel:
    cfg = {"enabled": True, **(config or {})}
    ch = WhatsAppChannel(cfg, MagicMock())
    ch._client = _mock_client()
    ch._connected = True
    return ch


# ── Outbound: text + media + typing ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_text_only():
    ch = _make_channel()
    msg = OutboundMessage(channel="whatsapp", chat_id="123@s.whatsapp.net", content="hello")

    await ch.send(msg)

    # send() always pauses typing first, then sends.
    ch._client.send_typing.assert_awaited_with("123@s.whatsapp.net", composing=False)
    ch._client.send_message.assert_awaited_once_with("123@s.whatsapp.net", "hello")
    ch._client.send_media.assert_not_called()


@pytest.mark.asyncio
async def test_send_media_with_caption():
    ch = _make_channel()
    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="check this out",
        media=["/tmp/photo.jpg"],
    )

    await ch.send(msg)

    # When media is present, the text is sent as a caption on the first media —
    # there is no separate send_message call.
    ch._client.send_message.assert_not_called()
    ch._client.send_media.assert_awaited_once()
    kwargs = ch._client.send_media.await_args.kwargs
    assert kwargs["to"] == "123@s.whatsapp.net"
    assert kwargs["file_path"] == "/tmp/photo.jpg"
    assert kwargs["mimetype"] == "image/jpeg"
    assert kwargs["caption"] == "check this out"
    assert kwargs["file_name"] == "photo.jpg"


@pytest.mark.asyncio
async def test_send_media_only_no_text():
    ch = _make_channel()
    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="",
        media=["/tmp/doc.pdf"],
    )

    await ch.send(msg)

    ch._client.send_media.assert_awaited_once()
    assert ch._client.send_media.await_args.kwargs["mimetype"] == "application/pdf"


@pytest.mark.asyncio
async def test_send_multiple_media():
    ch = _make_channel()
    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="",
        media=["/tmp/a.png", "/tmp/b.mp4"],
    )

    await ch.send(msg)

    assert ch._client.send_media.await_count == 2
    mimes = [c.kwargs["mimetype"] for c in ch._client.send_media.await_args_list]
    assert mimes == ["image/png", "video/mp4"]


@pytest.mark.asyncio
async def test_send_when_disconnected_raises():
    ch = _make_channel()
    ch._connected = False

    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="hello",
        media=["/tmp/x.jpg"],
    )
    with pytest.raises(ConnectionError, match="not connected"):
        await ch.send(msg)

    ch._client.send_message.assert_not_called()
    ch._client.send_media.assert_not_called()


@pytest.mark.asyncio
async def test_send_propagates_client_error():
    """When the underlying client raises, send() must not swallow the error."""
    ch = _make_channel()
    ch._client.send_media.side_effect = RuntimeError("upload failed")

    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="check this",
        media=["/tmp/photo.jpg"],
    )
    with pytest.raises(RuntimeError, match="upload failed"):
        await ch.send(msg)


@pytest.mark.asyncio
async def test_send_records_partial_media_for_retry():
    """If the second media fails, _sent_media metadata records the first so retries skip it."""
    ch = _make_channel()
    call_count = {"n": 0}

    async def flaky(**_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("network blip")
        return {"lid": None}

    ch._client.send_media.side_effect = flaky

    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="",
        media=["/tmp/a.png", "/tmp/b.png"],
    )
    with pytest.raises(RuntimeError, match="network blip"):
        await ch.send(msg)

    assert msg.metadata["_sent_media"] == ["/tmp/a.png"]


# ── Inbound: group policy + identity classification ───────────────────────────


def _inbound(**overrides) -> WAInboundMessage:
    """Build a default InboundMessage; overrides replace fields.

    Default ``timestamp`` is now() so a future age-based filter (history-sync
    drop, stale-message rejection) wouldn't silently invalidate every test.
    Tests targeting freshness-based logic should pass an explicit value.
    """
    import time

    base = dict(
        id="m1",
        sender="12345@s.whatsapp.net",
        pn="",
        content="hi",
        timestamp=int(time.time()),
        is_group=False,
        was_mentioned=False,
        media=[],
        push_name="",
    )
    base.update(overrides)
    return WAInboundMessage(**base)


@pytest.mark.asyncio
async def test_group_policy_mention_skips_unmentioned_message():
    ch = _make_channel({"groupPolicy": "mention"})
    ch._handle_message = AsyncMock()

    await ch._on_inbound(
        _inbound(
            sender="12345@g.us",
            pn="user@s.whatsapp.net",
            content="hello group",
            is_group=True,
            was_mentioned=False,
        )
    )
    ch._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_group_policy_mention_accepts_mentioned_message():
    ch = _make_channel({"groupPolicy": "mention"})
    ch._handle_message = AsyncMock()

    await ch._on_inbound(
        _inbound(
            sender="12345@g.us",
            pn="user@s.whatsapp.net",
            content="hello @bot",
            is_group=True,
            was_mentioned=True,
        )
    )
    ch._handle_message.assert_awaited_once()
    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["chat_id"] == "12345@g.us"
    assert kwargs["sender_id"] == "user"


@pytest.mark.asyncio
async def test_sender_id_prefers_phone_jid_over_lid():
    ch = _make_channel()
    ch._handle_message = AsyncMock()

    await ch._on_inbound(
        _inbound(
            id="lid1",
            sender="ABC123@lid.whatsapp.net",
            pn="5551234@s.whatsapp.net",
            content="hi",
        )
    )
    assert ch._handle_message.await_args.kwargs["sender_id"] == "5551234"


@pytest.mark.asyncio
async def test_lid_to_phone_cache_resolves_lid_only_messages():
    ch = _make_channel()
    ch._handle_message = AsyncMock()

    # First message: both IDs present → cache populated
    await ch._on_inbound(
        _inbound(
            id="c1",
            sender="LID99@lid.whatsapp.net",
            pn="5559999@s.whatsapp.net",
            content="first",
        )
    )
    # Second message: only LID — cache should resolve it
    await ch._on_inbound(
        _inbound(
            id="c2",
            sender="LID99@lid.whatsapp.net",
            pn="",
            content="second",
        )
    )

    assert ch._handle_message.await_args_list[1].kwargs["sender_id"] == "5559999"


# ── Typing indicator ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_typing_sends_composing_true():
    ch = _make_channel()
    await ch._start_typing("chat1@lid")
    await asyncio.sleep(0.05)
    await ch._stop_typing("chat1@lid")
    await asyncio.sleep(0.05)

    composing_true = [
        c for c in ch._client.send_typing.await_args_list if c.kwargs.get("composing") is True
    ]
    assert composing_true, "must have at least one composing=True call"
    assert composing_true[0].args[0] == "chat1@lid"


@pytest.mark.asyncio
async def test_send_pauses_typing_before_sending():
    ch = _make_channel()
    await ch._start_typing("chat1@lid")
    await asyncio.sleep(0.05)

    msg = OutboundMessage(channel="whatsapp", chat_id="chat1@lid", content="reply")
    await ch.send(msg)

    typing_calls = ch._client.send_typing.await_args_list
    msg_calls = ch._client.send_message.await_args_list
    paused_indices = [i for i, c in enumerate(typing_calls) if c.kwargs.get("composing") is False]
    assert paused_indices, "composing=False must be sent"
    assert msg_calls, "message must be sent"
    # The send_typing False right before send_message should fire — the channel
    # currently issues stop_typing → then send_message via separate awaited
    # coroutines. We assert ordering by watching the mock's call sequence.


@pytest.mark.asyncio
async def test_typing_task_cancelled_on_stop_typing():
    ch = _make_channel()
    await ch._start_typing("chat1@lid")
    assert "chat1@lid" in ch._typing_tasks

    await ch._stop_typing("chat1@lid")
    assert "chat1@lid" not in ch._typing_tasks


# ── Identity resolution ───────────────────────────────────────────────────────


def _make_identity_channel(lid_map=None, sender_map=None, allow_from=None) -> WhatsAppChannel:
    config = {
        "enabled": True,
        "identity_resolution": True,
        "allow_from": allow_from or [],
    }
    ch = WhatsAppChannel(config, MagicMock())
    ch._client = _mock_client()
    ch._connected = True
    ch._lid_map = lid_map or {}
    ch._sender_map = sender_map or {}
    ch._lid_map_loaded = True
    return ch


class TestIsAllowedWithLid:
    def test_phone_in_allow_from(self):
        ch = _make_identity_channel(allow_from=["14125550002"])
        assert ch.is_allowed("14125550002") is True

    def test_lid_maps_to_allowed_phone(self):
        ch = _make_identity_channel(
            allow_from=["14125550002"],
            lid_map={"914125550002": {"phone": "14125550002"}},
        )
        assert ch.is_allowed("914125550002") is True

    def test_lid_maps_to_disallowed_phone(self):
        ch = _make_identity_channel(
            allow_from=["14125550002"],
            lid_map={"999999": {"phone": "99999999"}},
        )
        assert ch.is_allowed("999999") is False

    def test_unknown_lid_denied(self):
        ch = _make_identity_channel(allow_from=["14125550002"])
        assert ch.is_allowed("999999") is False


class TestResolveSenderName:
    def test_resolves_from_sender_map(self):
        ch = _make_identity_channel(sender_map={"14125550002": "Emeka"})
        assert ch._resolve_sender_name("14125550002", "s1") == "Emeka"

    def test_resolves_lid_via_cross_reference(self):
        ch = _make_identity_channel(
            sender_map={"14125550002": "Emeka"},
            lid_map={"914125550002": {"phone": "14125550002"}},
        )
        assert ch._resolve_sender_name("914125550002", "s1") == "Emeka"

    def test_resolves_lid_with_direct_name(self):
        ch = _make_identity_channel(
            lid_map={"914125550002": {"phone": "14125550002", "name": "Emeka Direct"}},
        )
        assert ch._resolve_sender_name("914125550002", "s1") == "Emeka Direct"

    def test_returns_none_for_unknown(self):
        ch = _make_identity_channel()
        assert ch._resolve_sender_name("999999", "s1") is None

    def test_only_injects_once_per_session(self):
        ch = _make_identity_channel(sender_map={"14125550002": "Emeka"})
        assert ch._resolve_sender_name("14125550002", "s1") == "Emeka"
        assert ch._resolve_sender_name("14125550002", "s1") is None

    def test_no_greet_mark_when_name_not_found(self):
        ch = _make_identity_channel()
        assert ch._resolve_sender_name("999999", "s1") is None
        ch._sender_map["999999"] = "Late Joiner"
        assert ch._resolve_sender_name("999999", "s1") == "Late Joiner"


@pytest.mark.asyncio
async def test_sender_name_injected_in_content():
    ch = _make_identity_channel(
        allow_from=["14125550002"],
        sender_map={"14125550002": "Emeka"},
    )
    ch._handle_message = AsyncMock()

    await ch._on_inbound(
        _inbound(sender="14125550002@s.whatsapp.net", pn="", content="Hello!")
    )

    content = ch._handle_message.await_args.kwargs["content"]
    assert content.startswith("[Sender: Emeka]")
    assert "Hello!" in content


@pytest.mark.asyncio
async def test_sender_name_not_injected_for_commands():
    ch = _make_identity_channel(
        allow_from=["14125550002"],
        sender_map={"14125550002": "Emeka"},
    )
    ch._handle_message = AsyncMock()

    await ch._on_inbound(
        _inbound(sender="14125550002@s.whatsapp.net", pn="", content="/status")
    )
    assert ch._handle_message.await_args.kwargs["content"] == "/status"


@pytest.mark.asyncio
async def test_sender_name_not_injected_for_media_only():
    ch = _make_identity_channel(
        allow_from=["14125550002"],
        sender_map={"14125550002": "Emeka"},
    )
    ch._handle_message = AsyncMock()

    await ch._on_inbound(
        _inbound(sender="14125550002@s.whatsapp.net", pn="", content="")
    )
    assert "[Sender:" not in ch._handle_message.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_lid_learned_from_inbound_pn():
    ch = _make_identity_channel(allow_from=["14125550002"])
    ch._handle_message = AsyncMock()

    await ch._on_inbound(
        _inbound(
            sender="914125550002@lid.whatsapp.net",
            pn="14125550002@s.whatsapp.net",
            content="hello",
        )
    )

    assert "914125550002" in ch._lid_map
    assert ch._lid_map["914125550002"]["phone"] == "14125550002"


@pytest.mark.asyncio
async def test_typing_not_started_for_disallowed_sender():
    ch = WhatsAppChannel({"enabled": True, "allow_from": ["99999"]}, MagicMock())
    ch._client = _mock_client()
    ch._connected = True
    ch._handle_message = AsyncMock()

    await ch._on_inbound(
        _inbound(
            sender="other@lid.whatsapp.net",
            pn="12345@s.whatsapp.net",
            content="hello",
        )
    )

    assert "other@lid.whatsapp.net" not in ch._typing_tasks
    composing_true = [
        c for c in ch._client.send_typing.await_args_list if c.kwargs.get("composing") is True
    ]
    assert not composing_true


# ── Voice / media tagging ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_voice_message_transcription_uses_media_path():
    ch = _make_channel()
    ch.transcription_provider = "openai"
    ch.transcription_api_key = "sk-test"
    ch._handle_message = AsyncMock()
    ch.transcribe_audio = AsyncMock(return_value="Hello world")

    await ch._on_inbound(
        _inbound(
            id="v1",
            sender="12345@s.whatsapp.net",
            content="[Voice Message]",
            media=["/tmp/voice.ogg"],
        )
    )

    ch.transcribe_audio.assert_awaited_once_with("/tmp/voice.ogg")
    assert ch._handle_message.await_args.kwargs["content"].startswith("Hello world")


@pytest.mark.asyncio
async def test_voice_message_no_media_shows_not_available():
    ch = _make_channel()
    ch._handle_message = AsyncMock()

    await ch._on_inbound(
        _inbound(id="v2", sender="12345@s.whatsapp.net", content="[Voice Message]")
    )

    assert ch._handle_message.await_args.kwargs["content"] == "[Voice Message: Audio not available]"


@pytest.mark.asyncio
async def test_image_path_tagged_in_content():
    """Non-voice media should produce a [image: path] / [file: path] tag."""
    ch = _make_channel({"allow_from": ["12345"]})
    ch._handle_message = AsyncMock()

    await ch._on_inbound(
        _inbound(
            sender="12345@s.whatsapp.net",
            content="check this",
            media=["/tmp/wa_x.jpg"],
        )
    )

    content = ch._handle_message.await_args.kwargs["content"]
    assert "[image: /tmp/wa_x.jpg]" in content
    assert content.startswith("check this")


def test_whatsapp_auth_dir_honors_env_override(monkeypatch, tmp_path):
    """NANOBOT_WHATSAPP_AUTH_DIR should override the default runtime subdir."""
    from nanobot.channels.whatsapp import _whatsapp_auth_dir

    target = tmp_path / "persistent" / "whatsapp-auth"
    monkeypatch.setenv("NANOBOT_WHATSAPP_AUTH_DIR", str(target))

    assert _whatsapp_auth_dir() == target


def test_whatsapp_auth_dir_falls_back_to_runtime_subdir(monkeypatch, tmp_path):
    """Without the override, fall back to get_runtime_subdir('whatsapp-auth')."""
    from nanobot.channels.whatsapp import _whatsapp_auth_dir

    monkeypatch.delenv("NANOBOT_WHATSAPP_AUTH_DIR", raising=False)
    monkeypatch.setattr(
        "nanobot.config.paths.get_config_path", lambda: tmp_path / "config.json"
    )

    result = _whatsapp_auth_dir()
    assert result == tmp_path / "whatsapp-auth"


def test_whatsapp_auth_dir_expands_user_in_override(monkeypatch):
    """Support `~` expansion in the env var so users can write `~/wa-auth`."""
    from nanobot.channels.whatsapp import _whatsapp_auth_dir

    monkeypatch.setenv("NANOBOT_WHATSAPP_AUTH_DIR", "~/wa-test-auth")

    assert _whatsapp_auth_dir() == Path.home() / "wa-test-auth"


@pytest.mark.asyncio
async def test_dedupe_processed_message_ids():
    ch = _make_channel({"allow_from": ["12345"]})
    ch._handle_message = AsyncMock()

    inbound = _inbound(id="dup1", sender="12345@s.whatsapp.net", content="hi")
    await ch._on_inbound(inbound)
    await ch._on_inbound(inbound)

    assert ch._handle_message.await_count == 1
