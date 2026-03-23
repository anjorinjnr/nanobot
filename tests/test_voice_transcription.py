"""Tests for voice message transcription — WhatsApp and Telegram channels."""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.whatsapp import WhatsAppChannel
from nanobot.config.schema import WhatsAppConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_whatsapp_channel(gemini_api_key: str = "test-key") -> tuple[WhatsAppChannel, list[dict]]:
    """Return a WhatsApp channel wired to a fake WebSocket."""
    config = WhatsAppConfig(enabled=True, bridge_url="ws://localhost:3001", allow_from=["*"])
    channel = WhatsAppChannel(config, MessageBus(), gemini_api_key=gemini_api_key)

    published: list[dict] = []

    async def _fake_publish(msg):
        published.append({"content": msg.content, "sender_id": msg.sender_id})

    channel.bus.publish_inbound = _fake_publish  # type: ignore[method-assign]

    sent: list[str] = []

    class _FakeWS:
        async def send(self, data: str) -> None:
            sent.append(data)

    channel._ws = _FakeWS()
    channel._connected = True
    return channel, published


def _audio_bridge_message(
    audio_bytes: bytes = b"\x00" * 100,
    mimetype: str = "audio/ogg; codecs=opus",
    duration: float | None = 10.0,
    sender: str = "123456789@s.whatsapp.net",
) -> str:
    """Build a fake bridge JSON message containing audio data."""
    return json.dumps({
        "type": "message",
        "id": "msg-001",
        "sender": sender,
        "pn": "",
        "content": "",
        "timestamp": 1700000000,
        "isGroup": False,
        "audio": {
            "data": base64.b64encode(audio_bytes).decode(),
            "mimetype": mimetype,
            **({"duration": duration} if duration is not None else {}),
        },
    })


# ── GeminiTranscriptionProvider unit tests ───────────────────────────────────

class TestGeminiTranscriptionProvider:
    @pytest.mark.asyncio
    async def test_returns_transcript_on_success(self) -> None:
        from nanobot.providers.transcription import GeminiTranscriptionProvider

        provider = GeminiTranscriptionProvider(api_key="fake-key", model="gemini-2.5-flash")

        fake_response = MagicMock()
        fake_response.text = "  Hello world  "

        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        with patch("google.genai.Client", return_value=fake_client):
            result = await provider.transcribe_bytes(b"\x00" * 50, mime_type="audio/ogg")

        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_returns_failure_sentinel_when_no_api_key(self) -> None:
        from nanobot.providers.transcription import GeminiTranscriptionProvider

        provider = GeminiTranscriptionProvider(api_key=None)
        # Ensure env var is absent
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("GEMINI_API_KEY", None)
            result = await provider.transcribe_bytes(b"\x00" * 50)

        assert result == "[Voice message - transcription failed]"

    @pytest.mark.asyncio
    async def test_returns_too_long_sentinel_when_over_5_minutes(self) -> None:
        from nanobot.providers.transcription import GeminiTranscriptionProvider

        provider = GeminiTranscriptionProvider(api_key="fake-key")
        result = await provider.transcribe_bytes(
            b"\x00" * 50, duration_seconds=301.0
        )

        assert result == "[Voice message too long - please type it out]"

    @pytest.mark.asyncio
    async def test_returns_failure_sentinel_on_sdk_error(self) -> None:
        from nanobot.providers.transcription import GeminiTranscriptionProvider

        provider = GeminiTranscriptionProvider(api_key="fake-key")

        with patch("google.genai.Client", side_effect=RuntimeError("API error")):
            result = await provider.transcribe_bytes(b"\x00" * 50)

        assert result == "[Voice message - transcription failed]"

    @pytest.mark.asyncio
    async def test_accepts_exactly_5_minutes(self) -> None:
        from nanobot.providers.transcription import GeminiTranscriptionProvider

        provider = GeminiTranscriptionProvider(api_key="fake-key", model="gemini-2.5-flash")

        fake_response = MagicMock()
        fake_response.text = "Exactly five minutes"

        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        with patch("google.genai.Client", return_value=fake_client):
            result = await provider.transcribe_bytes(b"\x00" * 50, duration_seconds=300.0)

        assert result == "Exactly five minutes"


# ── WhatsApp channel voice transcription integration tests ───────────────────

class TestWhatsAppVoiceTranscription:
    @pytest.mark.asyncio
    async def test_voice_message_replaced_with_transcript(self) -> None:
        channel, published = _make_whatsapp_channel()

        with patch(
            "nanobot.channels.whatsapp.WhatsAppChannel._transcribe_audio",
            new=AsyncMock(return_value="Hi, this is a test."),
        ):
            await channel._handle_bridge_message(_audio_bridge_message())

        assert len(published) == 1
        assert published[0]["content"] == "Hi, this is a test."

    @pytest.mark.asyncio
    async def test_voice_message_uses_fallback_on_transcription_error(self) -> None:
        channel, published = _make_whatsapp_channel()

        with patch(
            "nanobot.channels.whatsapp.WhatsAppChannel._transcribe_audio",
            new=AsyncMock(return_value="[Voice message - transcription failed]"),
        ):
            await channel._handle_bridge_message(_audio_bridge_message())

        assert len(published) == 1
        assert published[0]["content"] == "[Voice message - transcription failed]"

    @pytest.mark.asyncio
    async def test_normal_text_message_not_transcribed(self) -> None:
        channel, published = _make_whatsapp_channel()

        raw = json.dumps({
            "type": "message",
            "id": "msg-002",
            "sender": "123@s.whatsapp.net",
            "pn": "",
            "content": "Hello there",
            "timestamp": 1700000001,
            "isGroup": False,
        })

        with patch(
            "nanobot.channels.whatsapp.WhatsAppChannel._transcribe_audio",
            new=AsyncMock(side_effect=AssertionError("should not be called")),
        ):
            await channel._handle_bridge_message(raw)

        assert len(published) == 1
        assert published[0]["content"] == "Hello there"

    @pytest.mark.asyncio
    async def test_transcription_disabled_returns_voice_message_placeholder(self) -> None:
        channel, published = _make_whatsapp_channel()

        with patch.dict("os.environ", {"VOICE_TRANSCRIPTION_PROVIDER": "disabled"}):
            with patch(
                "nanobot.providers.transcription.GeminiTranscriptionProvider.transcribe_bytes",
                new=AsyncMock(side_effect=AssertionError("should not be called")),
            ):
                await channel._handle_bridge_message(_audio_bridge_message())

        assert len(published) == 1
        assert published[0]["content"] == "[Voice Message]"

    @pytest.mark.asyncio
    async def test_invalid_base64_returns_failure_sentinel(self) -> None:
        channel, published = _make_whatsapp_channel()

        raw = json.dumps({
            "type": "message",
            "id": "msg-003",
            "sender": "123@s.whatsapp.net",
            "pn": "",
            "content": "",
            "timestamp": 1700000002,
            "isGroup": False,
            "audio": {
                "data": "!!!invalid_base64!!!",
                "mimetype": "audio/ogg",
            },
        })

        await channel._handle_bridge_message(raw)

        assert len(published) == 1
        assert published[0]["content"] == "[Voice message - transcription failed]"

    @pytest.mark.asyncio
    async def test_too_long_audio_returns_sentinel(self) -> None:
        channel, published = _make_whatsapp_channel()

        with patch.dict("os.environ", {"VOICE_TRANSCRIPTION_PROVIDER": "gemini"}):
            with patch(
                "nanobot.providers.transcription.GeminiTranscriptionProvider.transcribe_bytes",
                new=AsyncMock(return_value="[Voice message too long - please type it out]"),
            ):
                await channel._handle_bridge_message(
                    _audio_bridge_message(duration=400.0)
                )

        assert published[0]["content"] == "[Voice message too long - please type it out]"
