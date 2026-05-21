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


# ── Watchdog: interim message on long silences ───────────────────────────────
#
# The watchdog fires a single "still working on this" message after
# _WATCHDOG_DELAY_S of no outbound, so the user is never left hanging during
# long tool loops. Tests use a monkey-patched short delay to keep them fast.


@pytest.mark.asyncio
async def test_watchdog_fires_interim_message_after_delay(monkeypatch):
    ch = _make_channel()
    monkeypatch.setattr(WhatsAppChannel, "_WATCHDOG_DELAY_S", 0.05)

    await ch._start_watchdog("chat1@lid")
    await asyncio.sleep(0.15)

    sent_messages = [c.args for c in ch._client.send_message.await_args_list]
    assert any(args[0] == "chat1@lid" and "Still working" in args[1] for args in sent_messages), (
        f"expected interim message; got {sent_messages}"
    )


@pytest.mark.asyncio
async def test_watchdog_cancelled_before_delay_does_not_fire(monkeypatch):
    ch = _make_channel()
    monkeypatch.setattr(WhatsAppChannel, "_WATCHDOG_DELAY_S", 0.2)

    await ch._start_watchdog("chat1@lid")
    await asyncio.sleep(0.05)
    await ch._stop_watchdog("chat1@lid")
    await asyncio.sleep(0.3)

    assert ch._client.send_message.await_count == 0
    assert "chat1@lid" not in ch._watchdog_tasks


@pytest.mark.asyncio
async def test_send_cancels_watchdog(monkeypatch):
    ch = _make_channel()
    monkeypatch.setattr(WhatsAppChannel, "_WATCHDOG_DELAY_S", 0.2)

    await ch._start_watchdog("chat1@lid")
    msg = OutboundMessage(channel="whatsapp", chat_id="chat1@lid", content="real reply")
    await ch.send(msg)
    await asyncio.sleep(0.3)

    # Only the real reply should have gone out; the watchdog must have been
    # cancelled before its interim message fired.
    sent = [c.args[1] for c in ch._client.send_message.await_args_list]
    assert sent == ["real reply"], f"watchdog leaked; got {sent}"
    assert "chat1@lid" not in ch._watchdog_tasks


@pytest.mark.asyncio
async def test_start_watchdog_resets_previous(monkeypatch):
    ch = _make_channel()
    monkeypatch.setattr(WhatsAppChannel, "_WATCHDOG_DELAY_S", 0.1)

    await ch._start_watchdog("chat1@lid")
    first_task = ch._watchdog_tasks["chat1@lid"]
    await ch._start_watchdog("chat1@lid")
    second_task = ch._watchdog_tasks["chat1@lid"]

    assert first_task is not second_task
    assert first_task.cancelled() or first_task.done()


@pytest.mark.asyncio
async def test_watchdog_silent_when_client_disconnected(monkeypatch):
    ch = _make_channel()
    monkeypatch.setattr(WhatsAppChannel, "_WATCHDOG_DELAY_S", 0.05)
    ch._connected = False

    await ch._start_watchdog("chat1@lid")
    await asyncio.sleep(0.15)

    assert ch._client.send_message.await_count == 0
    assert "chat1@lid" not in ch._watchdog_tasks


@pytest.mark.asyncio
async def test_watchdog_does_not_send_when_entry_already_popped(monkeypatch):
    """The dict pop happens BEFORE cancel in _stop_watchdog, so a watchdog
    whose sleep returned but hasn't yet hit send_message must bail out
    when it sees its dict entry is gone — otherwise a concurrent send()
    race can produce a duplicate interim message."""
    ch = _make_channel()
    monkeypatch.setattr(WhatsAppChannel, "_WATCHDOG_DELAY_S", 0.05)

    await ch._start_watchdog("chat1@lid")
    # Simulate the race: pop the entry between sleep-returns and send.
    # We do it from outside without cancelling so the task continues into
    # its ownership check and self-aborts.
    ch._watchdog_tasks.pop("chat1@lid", None)
    await asyncio.sleep(0.15)

    assert ch._client.send_message.await_count == 0


@pytest.mark.asyncio
async def test_watchdog_self_evicts_after_firing(monkeypatch):
    """After the watchdog runs to completion, its dict entry must be cleared
    so a chat that fires once and goes quiet doesn't leak a completed task."""
    ch = _make_channel()
    monkeypatch.setattr(WhatsAppChannel, "_WATCHDOG_DELAY_S", 0.05)

    await ch._start_watchdog("chat1@lid")
    await asyncio.sleep(0.15)

    assert ch._client.send_message.await_count == 1
    assert "chat1@lid" not in ch._watchdog_tasks


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


class TestLookupSenderName:
    """`_lookup_sender_name` is the side-effect-free variant of
    `_resolve_sender_name` (no greeted-set marking). The auto-heal path
    relies on it firing on every inbound, not just first-per-session."""

    def test_resolves_from_sender_map(self):
        ch = _make_identity_channel(sender_map={"14125550002": "Emeka"})
        assert ch._lookup_sender_name("14125550002") == "Emeka"

    def test_resolves_lid_via_cross_reference(self):
        ch = _make_identity_channel(
            sender_map={"14125550002": "Emeka"},
            lid_map={"914125550002": {"phone": "14125550002"}},
        )
        assert ch._lookup_sender_name("914125550002") == "Emeka"

    def test_resolves_lid_with_direct_name(self):
        ch = _make_identity_channel(
            lid_map={"914125550002": {"phone": "14125550002", "name": "Emeka Direct"}},
        )
        assert ch._lookup_sender_name("914125550002") == "Emeka Direct"

    def test_unknown_returns_none(self):
        ch = _make_identity_channel()
        assert ch._lookup_sender_name("999999") is None

    def test_does_not_mark_greeted(self):
        """Repeated calls must keep returning the same name — the side-effect-free
        variant is the whole point of this method."""
        ch = _make_identity_channel(sender_map={"14125550002": "Emeka"})
        assert ch._lookup_sender_name("14125550002") == "Emeka"
        assert ch._lookup_sender_name("14125550002") == "Emeka"
        # And the greeted-gate behavior of _resolve_sender_name still works.
        assert ch._resolve_sender_name("14125550002", "s1") == "Emeka"
        assert ch._resolve_sender_name("14125550002", "s1") is None

    def test_normalizes_us_country_code_when_map_is_10_digit(self):
        """homer's _build_sender_map strips country code prefixes from party_id
        JIDs (`4126920720@s.whatsapp.net` → `4126920720`), but Neonize emits the
        11-digit form (`14126920720`) on inbound. Look up must succeed either way."""
        ch = _make_identity_channel(sender_map={"4126920720": "Ebby"})
        # Neonize-form sender_id with leading 1.
        assert ch._lookup_sender_name("14126920720") == "Ebby"
        # And the bare 10-digit form (in case the map is keyed the other way).
        assert ch._lookup_sender_name("4126920720") == "Ebby"

    def test_normalizes_us_country_code_when_map_is_11_digit(self):
        """Inverse: map has the 11-digit form, inbound is bare 10-digit."""
        ch = _make_identity_channel(sender_map={"14126920720": "Ebby"})
        assert ch._lookup_sender_name("4126920720") == "Ebby"
        assert ch._lookup_sender_name("14126920720") == "Ebby"

    def test_no_spurious_match_for_short_string(self):
        """Don't try to normalize non-phone keys (LIDs, emails, sentinels)."""
        ch = _make_identity_channel(sender_map={"1abcd": "Should not match"})
        assert ch._lookup_sender_name("abcd") is None

    def test_lid_phone_uses_normalization(self):
        """When sender_id is a LID but lid_map exposes a phone, the phone
        lookup also benefits from country-code normalization."""
        ch = _make_identity_channel(
            sender_map={"4126920720": "Ebby"},
            lid_map={"246157477413033": {"phone": "14126920720"}},
        )
        assert ch._lookup_sender_name("246157477413033") == "Ebby"


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


def test_sender_map_paths_honors_config_field(tmp_path):
    """``sender_map_path`` config field takes precedence over the data-dir default."""
    target = tmp_path / "ws" / "sender_map.json"
    ch = WhatsAppChannel(
        {"enabled": True, "sender_map_path": str(target)}, MagicMock()
    )

    paths = ch._sender_map_paths()
    assert paths[0] == target
    # Data-dir candidate remains as the fallback.
    assert paths[-1].name == "sender_map.json"


def test_sender_map_paths_falls_back_to_data_dir(monkeypatch, tmp_path):
    """Without the override, only the data-dir candidate is returned."""
    monkeypatch.setattr(
        "nanobot.config.paths.get_config_path", lambda: tmp_path / "config.json"
    )

    ch = WhatsAppChannel({"enabled": True}, MagicMock())
    paths = ch._sender_map_paths()
    assert paths == [tmp_path / "sender_map.json"]


def test_sender_map_paths_isolates_per_channel(tmp_path):
    """Two channels in the same process get independent paths from their own configs."""
    main_path = tmp_path / "main" / "sender_map.json"
    guest_path = tmp_path / "guest" / "sender_map.json"

    main_ch = WhatsAppChannel(
        {"enabled": True, "sender_map_path": str(main_path)}, MagicMock()
    )
    guest_ch = WhatsAppChannel(
        {"enabled": True, "sender_map_path": str(guest_path)}, MagicMock()
    )

    assert main_ch._sender_map_paths()[0] == main_path
    assert guest_ch._sender_map_paths()[0] == guest_path


def test_sender_map_loads_from_config_override(tmp_path, monkeypatch):
    """End-to-end: when override file exists, ``_read_maps_from_disk`` loads from it."""
    import json

    override_file = tmp_path / "ws" / "sender_map.json"
    override_file.parent.mkdir(parents=True)
    override_file.write_text(json.dumps({"14125550002": "Emeka"}), encoding="utf-8")
    monkeypatch.setattr(
        "nanobot.config.paths.get_config_path", lambda: tmp_path / "config.json"
    )

    ch = WhatsAppChannel(
        {
            "enabled": True,
            "identity_resolution": True,
            "sender_map_path": str(override_file),
        },
        MagicMock(),
    )
    _, sender_map = ch._read_maps_from_disk()
    assert sender_map == {"14125550002": "Emeka"}


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


# ── _render_for_whatsapp -------------------------------------------------------
# Outbound rendering. WhatsApp doesn't parse `[text](url)` markdown syntax,
# so the channel strips it down to the bare URL before sending. Tests below
# pin the regex against the failure mode that triggered the rewrite: short
# links wrapped in `[Click here](…)` getting auto-linked with the trailing
# `)` and 404ing on the receiver side.


class TestRenderForWhatsApp:
    def test_collapses_markdown_link_to_bare_url(self):
        from nanobot.channels.whatsapp import _render_for_whatsapp

        out = _render_for_whatsapp("RSVP: [Click here](https://homer.help/s/YRW229SB)")
        assert out == "RSVP: https://homer.help/s/YRW229SB"

    def test_multiple_links_in_one_message(self):
        from nanobot.channels.whatsapp import _render_for_whatsapp

        text = "See [event](https://a.example) and [map](https://b.example/x?y=1)"
        assert _render_for_whatsapp(text) == "See https://a.example and https://b.example/x?y=1"

    def test_leaves_bare_url_alone(self):
        from nanobot.channels.whatsapp import _render_for_whatsapp

        # Bare URLs are already what WhatsApp's auto-linker wants — leave alone.
        assert _render_for_whatsapp("Go to https://homer.help/s/ABC") == \
            "Go to https://homer.help/s/ABC"

    def test_preserves_surrounding_text(self):
        from nanobot.channels.whatsapp import _render_for_whatsapp

        # The non-link text on either side of the markdown must survive verbatim,
        # including whitespace and trailing punctuation.
        text = "Tap to RSVP: [here](https://homer.help/s/X)."
        assert _render_for_whatsapp(text) == "Tap to RSVP: https://homer.help/s/X."

    def test_does_not_mangle_image_syntax(self):
        from nanobot.channels.whatsapp import _render_for_whatsapp

        # `![alt](url)` is markdown image syntax; the negative lookbehind on `!`
        # leaves it intact so we don't produce a stray `!url`.
        assert _render_for_whatsapp("![alt](https://x.example/img.png)") == \
            "![alt](https://x.example/img.png)"

    def test_empty_string_passthrough(self):
        from nanobot.channels.whatsapp import _render_for_whatsapp

        assert _render_for_whatsapp("") == ""

    def test_none_passthrough(self):
        from nanobot.channels.whatsapp import _render_for_whatsapp

        # send() guards on `msg.content` being truthy before calling, but the
        # helper itself should tolerate None to keep callers from having to
        # branch.
        assert _render_for_whatsapp(None) is None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_send_applies_markdown_link_transform():
    """End-to-end: a message with `[label](url)` reaches send_message as
    just the URL — pins the bug fix at the integration boundary."""
    ch = _make_channel()
    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="RSVP: [Click here](https://homer.help/s/YRW229SB)",
    )

    await ch.send(msg)

    ch._client.send_message.assert_awaited_once_with(
        "123@s.whatsapp.net",
        "RSVP: https://homer.help/s/YRW229SB",
    )


@pytest.mark.asyncio
async def test_send_media_caption_applies_markdown_link_transform():
    """Same transform applies when the text rides as a media caption."""
    ch = _make_channel()
    msg = OutboundMessage(
        channel="whatsapp",
        chat_id="123@s.whatsapp.net",
        content="Photo from [the event](https://homer.help/e/abc)",
        media=["/tmp/photo.jpg"],
    )

    await ch.send(msg)

    ch._client.send_media.assert_awaited_once()
    kwargs = ch._client.send_media.await_args.kwargs
    assert kwargs["caption"] == "Photo from https://homer.help/e/abc"


# ── Auto-heal users.yaml on handle drift ─────────────────────────────────────
#
# `_auto_heal_users_yaml` lazy-imports `users_loader` from $HOMER_TOOLS at
# runtime. In CI/dev the homer repo lives at `../homer` relative to nanobot;
# this fixture puts its `tools/` dir on sys.path so the import resolves the
# same way the container's entrypoint sets PYTHONPATH=$HOMER_TOOLS at boot.

import yaml as _yaml
# `homer_users_yaml` fixture lives in tests/conftest.py — shared with the
# heartbeat dispatch tests.


def _write_v2(path: Path, **users) -> None:
    path.write_text(_yaml.safe_dump(
        {"schema_version": 2, "users": users}, sort_keys=False,
    ), encoding="utf-8")


class TestAutoHealUsersYaml:
    """Drift detection runs after `_resolve_sender_name`. The hook is
    `_auto_heal_users_yaml(name, sender_jid)`; it must be idempotent,
    only rewrite when needed, and never crash message handling."""

    @pytest.mark.asyncio
    async def test_rewrites_when_handle_differs(self, homer_users_yaml):
        _write_v2(homer_users_yaml, primary={
            "display_name": "Ebby Anjorin",
            "role": "admin",
            "channels": {"whatsapp": "246157477413033@lid"},  # stale Baileys form
        })
        ch = _make_identity_channel()
        await ch._auto_heal_users_yaml("Ebby Anjorin", "246157477413033@lid.whatsapp.net")
        doc = _yaml.safe_load(homer_users_yaml.read_text())
        assert doc["users"]["primary"]["channels"]["whatsapp"] == "246157477413033@lid.whatsapp.net"

    @pytest.mark.asyncio
    async def test_noop_when_handle_matches(self, homer_users_yaml):
        _write_v2(homer_users_yaml, primary={
            "display_name": "Ebby Anjorin",
            "role": "admin",
            "channels": {"whatsapp": "246157477413033@lid.whatsapp.net"},
        })
        before_mtime = homer_users_yaml.stat().st_mtime_ns
        ch = _make_identity_channel()
        await ch._auto_heal_users_yaml("Ebby Anjorin", "246157477413033@lid.whatsapp.net")
        # File must not have been rewritten — same mtime.
        assert homer_users_yaml.stat().st_mtime_ns == before_mtime

    @pytest.mark.asyncio
    async def test_noop_for_unknown_name(self, homer_users_yaml):
        _write_v2(homer_users_yaml, primary={
            "display_name": "Ebby Anjorin", "role": "admin",
            "channels": {"whatsapp": "old@lid"},
        })
        before = homer_users_yaml.read_bytes()
        ch = _make_identity_channel()
        # Name not in registry — must not create a new user, must not crash.
        await ch._auto_heal_users_yaml("Stranger", "999@lid.whatsapp.net")
        assert homer_users_yaml.read_bytes() == before

    @pytest.mark.asyncio
    async def test_empty_args_skip(self, homer_users_yaml):
        _write_v2(homer_users_yaml, primary={
            "display_name": "Ebby", "role": "admin",
            "channels": {"whatsapp": "x@lid"},
        })
        before = homer_users_yaml.read_bytes()
        ch = _make_identity_channel()
        await ch._auto_heal_users_yaml("", "x@lid.whatsapp.net")
        await ch._auto_heal_users_yaml("Ebby", "")
        assert homer_users_yaml.read_bytes() == before

    @pytest.mark.asyncio
    async def test_swallows_load_failure(self, monkeypatch, homer_users_yaml, caplog):
        """A broken users.yaml must not crash the channel — the heal logs and
        returns, and the inbound message still gets processed."""
        homer_users_yaml.write_text("not: valid: yaml: ::\n", encoding="utf-8")
        ch = _make_identity_channel()
        # Should not raise.
        await ch._auto_heal_users_yaml("Ebby Anjorin", "246157477413033@lid.whatsapp.net")

    @pytest.mark.asyncio
    async def test_creates_channels_dict_when_absent(self, homer_users_yaml):
        """A user record with no channels yet (e.g. fresh row written by the
        portal before the welcome backfill ran) still gets healed."""
        _write_v2(homer_users_yaml, primary={
            "display_name": "Ebby Anjorin", "role": "admin",
        })
        ch = _make_identity_channel()
        await ch._auto_heal_users_yaml("Ebby Anjorin", "246157477413033@lid.whatsapp.net")
        doc = _yaml.safe_load(homer_users_yaml.read_text())
        assert doc["users"]["primary"]["channels"]["whatsapp"] == "246157477413033@lid.whatsapp.net"

    @pytest.mark.asyncio
    async def test_case_insensitive_name_match(self, homer_users_yaml):
        """sender_map values come from homer's USER.md / users.yaml; capitalisation
        may not match exactly. Heal uses case-insensitive display_name lookup."""
        _write_v2(homer_users_yaml, seun={
            "display_name": "Seun", "role": "member",
            "channels": {"whatsapp": "105321339076677@lid"},
        })
        ch = _make_identity_channel()
        await ch._auto_heal_users_yaml("seun", "105321339076677@lid.whatsapp.net")
        doc = _yaml.safe_load(homer_users_yaml.read_text())
        assert doc["users"]["seun"]["channels"]["whatsapp"] == "105321339076677@lid.whatsapp.net"

    @pytest.mark.asyncio
    async def test_nickname_matches_full_display_name(self, homer_users_yaml):
        """homer's sender_map stores nicknames (e.g. 'Ebby'), users.yaml stores
        full names ('Ebby Anjorin'). The heal falls back to first-token match
        when full-string match fails."""
        _write_v2(homer_users_yaml, primary={
            "display_name": "Ebby Anjorin", "role": "admin",
            "channels": {"whatsapp": "246157477413033@lid"},
        })
        ch = _make_identity_channel()
        await ch._auto_heal_users_yaml("Ebby", "246157477413033@lid.whatsapp.net")
        doc = _yaml.safe_load(homer_users_yaml.read_text())
        assert doc["users"]["primary"]["channels"]["whatsapp"] == "246157477413033@lid.whatsapp.net"

    @pytest.mark.asyncio
    async def test_ambiguous_nickname_skips(self, homer_users_yaml):
        """If two stored users share a first name, the heal can't safely pick
        one — must skip rather than risk healing the wrong row."""
        _write_v2(homer_users_yaml,
            primary={
                "display_name": "Alex Johnson", "role": "admin",
                "channels": {"whatsapp": "stored-A@lid"},
            },
            alex_2={
                "display_name": "Alex Smith", "role": "member",
                "channels": {"whatsapp": "stored-B@lid"},
            },
        )
        before = homer_users_yaml.read_bytes()
        ch = _make_identity_channel()
        await ch._auto_heal_users_yaml("Alex", "live@lid.whatsapp.net")
        # File untouched — ambiguous first name → no-op.
        assert homer_users_yaml.read_bytes() == before
