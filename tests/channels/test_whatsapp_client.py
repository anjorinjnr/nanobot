"""Wrapper-level tests for the neonize-backed WhatsApp client.

These exercise the protocol-translation layer directly — JID round-trips,
proto-message decoding (text/media/voice/contact), mention detection's
lazy self-identity read, and the legacy-format renderer. The neonize Go
runtime isn't involved; we feed proto fixtures straight in.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from neonize.proto.Neonize_pb2 import JID
from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import (
    AudioMessage,
    ContactMessage,
    ContextInfo,
    DocumentMessage,
    ExtendedTextMessage,
    ImageMessage,
    Message,
    VideoMessage,
)

from nanobot.channels.whatsapp_client import (
    WhatsAppClient,
    WhatsAppClientOptions,
    _legacy_jid_string,
    _parse_legacy_jid,
)


# ── JID round-trip ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "legacy",
    [
        "1234567890@s.whatsapp.net",
        "9876543210@lid.whatsapp.net",
        "120363012345@g.us",
    ],
)
def test_jid_round_trip(legacy: str) -> None:
    """Parsing a legacy string and re-rendering it should yield the same value."""
    parsed = _parse_legacy_jid(legacy)
    assert _legacy_jid_string(parsed) == legacy


def test_jid_strips_device_suffix_on_parse() -> None:
    """Device-suffixed inputs (`12345:42@s.whatsapp.net`) drop the device on parse."""
    parsed = _parse_legacy_jid("12345:42@s.whatsapp.net")
    assert _legacy_jid_string(parsed) == "12345@s.whatsapp.net"


def test_jid_string_without_at_treated_as_user() -> None:
    """Bare numeric input is parsed as user with default s.whatsapp.net server."""
    parsed = _parse_legacy_jid("12345")
    assert parsed.User == "12345"
    assert parsed.Server == "s.whatsapp.net"


# ── _extract_text_and_media_kind ─────────────────────────────────────────────


def _wrap() -> WhatsAppClient:
    """Build a WhatsAppClient instance without invoking neonize.connect()."""
    from pathlib import Path

    return WhatsAppClient(
        WhatsAppClientOptions(
            auth_dir=Path("/tmp/_wa_test"),
            on_message=lambda _: None,  # type: ignore[arg-type]
            on_qr=lambda _: None,  # type: ignore[arg-type]
            on_status=lambda _: None,  # type: ignore[arg-type]
        )
    )


def test_extract_plain_text() -> None:
    msg = Message(conversation="hello world")
    text, fallback, is_audio, media = _wrap()._extract_text_and_media_kind(msg)
    assert text == "hello world"
    assert fallback is None and is_audio is False and media is None


def test_extract_extended_text() -> None:
    msg = Message(extendedTextMessage=ExtendedTextMessage(text="reply with link"))
    text, fallback, is_audio, media = _wrap()._extract_text_and_media_kind(msg)
    assert text == "reply with link"
    assert fallback is None and is_audio is False and media is None


def test_extract_image_with_caption() -> None:
    msg = Message(imageMessage=ImageMessage(caption="look at this", mimetype="image/jpeg"))
    text, fallback, is_audio, media = _wrap()._extract_text_and_media_kind(msg)
    assert text == "look at this"
    assert fallback == "[Image]"
    assert is_audio is False
    assert media is msg.imageMessage


def test_extract_image_without_caption() -> None:
    msg = Message(imageMessage=ImageMessage(mimetype="image/png"))
    text, fallback, is_audio, media = _wrap()._extract_text_and_media_kind(msg)
    assert text == ""
    assert fallback == "[Image]"
    assert media is msg.imageMessage


def test_extract_video() -> None:
    msg = Message(videoMessage=VideoMessage(caption="clip", mimetype="video/mp4"))
    text, fallback, _, media = _wrap()._extract_text_and_media_kind(msg)
    assert text == "clip"
    assert fallback == "[Video]"
    assert media is msg.videoMessage


def test_extract_document() -> None:
    msg = Message(
        documentMessage=DocumentMessage(
            caption="see attached", fileName="invoice.pdf", mimetype="application/pdf"
        )
    )
    text, fallback, _, media = _wrap()._extract_text_and_media_kind(msg)
    assert text == "see attached"
    assert fallback == "[Document]"
    assert media is msg.documentMessage


def test_extract_audio_voice_returns_sentinel() -> None:
    """Voice notes must surface text=None + fallback='[Voice Message]' so the
    channel triggers Whisper transcription."""
    msg = Message(audioMessage=AudioMessage(mimetype="audio/ogg; codecs=opus", PTT=True))
    text, fallback, is_audio, media = _wrap()._extract_text_and_media_kind(msg)
    assert text is None
    assert fallback == "[Voice Message]"
    assert is_audio is True
    assert media is msg.audioMessage


def test_extract_contact() -> None:
    msg = Message(
        contactMessage=ContactMessage(displayName="Jane Doe", vcard="BEGIN:VCARD\nEND:VCARD")
    )
    text, fallback, _, media = _wrap()._extract_text_and_media_kind(msg)
    assert text is not None and "Jane Doe" in text
    assert "BEGIN:VCARD" in text
    assert media is None  # Contacts aren't downloaded; text carries everything


def test_extract_unknown_type_returns_nones() -> None:
    msg = Message()
    text, fallback, is_audio, media = _wrap()._extract_text_and_media_kind(msg)
    assert (text, fallback, is_audio, media) == (None, None, False, None)


# ── _was_self_mentioned: lazy self-identity read ─────────────────────────────


def _msg_with_mentions(mentions: list[str]) -> Message:
    return Message(
        extendedTextMessage=ExtendedTextMessage(
            text="hey @bot",
            contextInfo=ContextInfo(mentionedJID=mentions),
        )
    )


def test_mention_detection_returns_false_when_client_me_unset() -> None:
    """Race: if the 'Me' event hasn't fired yet, we can't decide — return False
    rather than crash. Future arrivals after `client.me` populates will succeed."""
    w = _wrap()
    # Simulate the early window where _client exists but `me` is None.
    w._client = SimpleNamespace(me=None)
    msg = _msg_with_mentions(["111@s.whatsapp.net", "222@lid"])
    assert w._was_self_mentioned(msg, is_group=True) is False


def test_mention_detection_matches_self_phone() -> None:
    w = _wrap()
    w._client = SimpleNamespace(me=SimpleNamespace(User="14125550002"))
    msg = _msg_with_mentions(["14125550002@s.whatsapp.net", "999@s.whatsapp.net"])
    assert w._was_self_mentioned(msg, is_group=True) is True


def test_mention_detection_matches_self_lid() -> None:
    w = _wrap()
    # `client.me.User` is just the bare identity string — could be phone OR LID
    # depending on whatsmeow's pairing mode. Either way a matching mentionedJID
    # bare prefix is a hit.
    w._client = SimpleNamespace(me=SimpleNamespace(User="914125550002"))
    msg = _msg_with_mentions(["914125550002@lid", "111@s.whatsapp.net"])
    assert w._was_self_mentioned(msg, is_group=True) is True


def test_mention_detection_strips_device_suffix() -> None:
    w = _wrap()
    w._client = SimpleNamespace(me=SimpleNamespace(User="14125550002"))
    msg = _msg_with_mentions(["14125550002:42@s.whatsapp.net"])
    assert w._was_self_mentioned(msg, is_group=True) is True


def test_mention_detection_is_false_for_1to1() -> None:
    """The mention path is only meaningful in groups — DM mentions are no-ops."""
    w = _wrap()
    w._client = SimpleNamespace(me=SimpleNamespace(User="14125550002"))
    msg = _msg_with_mentions(["14125550002@s.whatsapp.net"])
    assert w._was_self_mentioned(msg, is_group=False) is False


def test_mention_detection_handles_other_message_types() -> None:
    """ContextInfo lives on imageMessage/videoMessage/etc. too — caption mentions count."""
    w = _wrap()
    w._client = SimpleNamespace(me=SimpleNamespace(User="14125550002"))
    msg = Message(
        imageMessage=ImageMessage(
            caption="hi @bot",
            contextInfo=ContextInfo(mentionedJID=["14125550002@s.whatsapp.net"]),
        )
    )
    assert w._was_self_mentioned(msg, is_group=True) is True


def test_mention_detection_no_mentions_returns_false() -> None:
    w = _wrap()
    w._client = SimpleNamespace(me=SimpleNamespace(User="14125550002"))
    msg = Message(extendedTextMessage=ExtendedTextMessage(text="just text, no @bot tag"))
    assert w._was_self_mentioned(msg, is_group=True) is False


# ── _legacy_jid_string special cases ─────────────────────────────────────────


def test_legacy_jid_string_renders_status_broadcast() -> None:
    """status@broadcast must round-trip; the channel keys off the prefix to drop it."""
    jid = JID(User="status", Server="broadcast")
    assert _legacy_jid_string(jid).startswith("status@")


# ── _suffix_for media filename heuristic ─────────────────────────────────────


@pytest.mark.parametrize(
    "mime,is_audio,expected_prefix",
    [
        ("image/png", False, "."),
        ("video/mp4", False, "."),
        ("application/pdf", False, "."),
        ("audio/ogg; codecs=opus", True, "."),
    ],
)
def test_suffix_for_known_mimetypes(mime: str, is_audio: bool, expected_prefix: str) -> None:
    suffix = WhatsAppClient._suffix_for(mime, "", is_audio=is_audio)
    assert suffix.startswith(expected_prefix) and len(suffix) > 1


def test_suffix_for_named_file_drops_extension() -> None:
    """If the original filename carries an extension, don't add another one."""
    assert WhatsAppClient._suffix_for("application/pdf", "invoice.pdf", is_audio=False) == ""


def test_suffix_for_audio_unknown_mime_falls_back_to_ogg() -> None:
    """Voice notes default to .ogg — Whisper sniffs the format anyway."""
    assert WhatsAppClient._suffix_for("", "", is_audio=True) == ".ogg"
