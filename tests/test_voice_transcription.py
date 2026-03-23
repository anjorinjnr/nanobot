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
from nanobot.providers.transcription import (
    GeminiTranscriptionProvider,
    GroqTranscriptionProvider,
    TranscriptionProvider,
    create_transcription_provider,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_whatsapp_channel(
    transcription_provider: TranscriptionProvider | None = None,
) -> tuple[WhatsAppChannel, list[dict]]:
    """Return a WhatsApp channel wired to a fake WebSocket."""
    config = WhatsAppConfig(enabled=True, bridge_url="ws://localhost:3001", allow_from=["*"])
    channel = WhatsAppChannel(config, MessageBus(), transcription_provider=transcription_provider)

    published: list[dict] = []

    async def _fake_publish(msg):
        published.append({"content": msg.content, "sender_id": msg.sender_id})

    channel.bus.publish_inbound = _fake_publish  # type: ignore[method-assign]

    class _FakeWS:
        async def send(self, data: str) -> None:
            pass

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


# ── TranscriptionProvider protocol ────────────────────────────────────────────

class TestTranscriptionProviderProtocol:
    def test_gemini_satisfies_protocol(self) -> None:
        provider = GeminiTranscriptionProvider(api_key="key")
        assert isinstance(provider, TranscriptionProvider)

    def test_groq_satisfies_protocol(self) -> None:
        provider = GroqTranscriptionProvider(api_key="key")
        assert isinstance(provider, TranscriptionProvider)

    def test_custom_provider_satisfies_protocol(self) -> None:
        class MyProvider:
            async def transcribe(self, audio_bytes, mime_type="audio/ogg", duration_seconds=None):
                return "ok"

        assert isinstance(MyProvider(), TranscriptionProvider)


# ── create_transcription_provider factory ─────────────────────────────────────

class TestCreateTranscriptionProvider:
    def test_returns_gemini_by_default(self) -> None:
        with patch.dict("os.environ", {"VOICE_TRANSCRIPTION_PROVIDER": "gemini"}):
            p = create_transcription_provider(api_key="k")
        assert isinstance(p, GeminiTranscriptionProvider)

    def test_returns_groq_when_specified(self) -> None:
        p = create_transcription_provider(provider_name="groq", api_key="k")
        assert isinstance(p, GroqTranscriptionProvider)

    def test_returns_none_when_disabled(self) -> None:
        p = create_transcription_provider(provider_name="disabled")
        assert p is None

    def test_returns_none_for_unknown_provider(self) -> None:
        p = create_transcription_provider(provider_name="unknown_xyz")
        assert p is None

    def test_reads_provider_from_env(self) -> None:
        with patch.dict("os.environ", {"VOICE_TRANSCRIPTION_PROVIDER": "groq"}):
            p = create_transcription_provider(api_key="k")
        assert isinstance(p, GroqTranscriptionProvider)


# ── GeminiTranscriptionProvider unit tests ────────────────────────────────────

def _mock_genai(transcript: str = "Hello world"):
    """Return a sys.modules patch that stubs out google-genai."""
    import sys
    fake_part = MagicMock()
    fake_response = MagicMock()
    fake_response.text = transcript
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_types = MagicMock()
    fake_types.Part.from_bytes.return_value = fake_part
    fake_google = MagicMock()
    fake_google.genai = fake_genai
    return patch.dict(
        "sys.modules",
        {"google": fake_google, "google.genai": fake_genai, "google.genai.types": fake_types},
    )


class TestGeminiTranscriptionProvider:
    @pytest.mark.asyncio
    async def test_returns_transcript_on_success(self) -> None:
        provider = GeminiTranscriptionProvider(api_key="fake-key", model="gemini-2.5-flash")
        with _mock_genai("  Hello world  "):
            result = await provider.transcribe(b"\x00" * 50, mime_type="audio/ogg")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_returns_failure_sentinel_when_no_api_key(self) -> None:
        import os
        os.environ.pop("GEMINI_API_KEY", None)
        provider = GeminiTranscriptionProvider(api_key=None)
        result = await provider.transcribe(b"\x00" * 50)
        assert result == "[Voice message - transcription failed]"

    @pytest.mark.asyncio
    async def test_returns_too_long_sentinel_when_over_5_minutes(self) -> None:
        provider = GeminiTranscriptionProvider(api_key="fake-key")
        result = await provider.transcribe(b"\x00" * 50, duration_seconds=301.0)
        assert result == "[Voice message too long - please type it out]"

    @pytest.mark.asyncio
    async def test_accepts_exactly_5_minutes(self) -> None:
        provider = GeminiTranscriptionProvider(api_key="fake-key", model="gemini-2.5-flash")
        with _mock_genai("Exactly five minutes"):
            result = await provider.transcribe(b"\x00" * 50, duration_seconds=300.0)
        assert result == "Exactly five minutes"

    @pytest.mark.asyncio
    async def test_returns_failure_sentinel_on_sdk_error(self) -> None:
        provider = GeminiTranscriptionProvider(api_key="fake-key")
        import sys
        with patch.dict("sys.modules", {"google": MagicMock(genai=MagicMock(Client=MagicMock(side_effect=RuntimeError("API error")))), "google.genai": MagicMock(Client=MagicMock(side_effect=RuntimeError("API error"))), "google.genai.types": MagicMock()}):
            result = await provider.transcribe(b"\x00" * 50)
        assert result == "[Voice message - transcription failed]"


# ── GroqTranscriptionProvider unit tests ──────────────────────────────────────

class TestGroqTranscriptionProvider:
    @pytest.mark.asyncio
    async def test_returns_transcript_on_success(self) -> None:
        import httpx

        provider = GroqTranscriptionProvider(api_key="fake-groq-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "Groq transcript"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await provider.transcribe(b"\x00" * 50, mime_type="audio/ogg")

        assert result == "Groq transcript"

    @pytest.mark.asyncio
    async def test_returns_failure_sentinel_when_no_api_key(self) -> None:
        import os
        os.environ.pop("GROQ_API_KEY", None)
        provider = GroqTranscriptionProvider(api_key=None)
        result = await provider.transcribe(b"\x00" * 50)
        assert result == "[Voice message - transcription failed]"

    @pytest.mark.asyncio
    async def test_returns_too_long_sentinel_when_over_5_minutes(self) -> None:
        provider = GroqTranscriptionProvider(api_key="fake-key")
        result = await provider.transcribe(b"\x00" * 50, duration_seconds=301.0)
        assert result == "[Voice message too long - please type it out]"

    @pytest.mark.asyncio
    async def test_returns_failure_sentinel_on_http_error(self) -> None:
        import httpx

        provider = GroqTranscriptionProvider(api_key="fake-groq-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=Exception("network error"))
            mock_client_cls.return_value = mock_client

            result = await provider.transcribe(b"\x00" * 50)

        assert result == "[Voice message - transcription failed]"


# ── WhatsApp channel voice transcription integration tests ────────────────────

class TestWhatsAppVoiceTranscription:
    @pytest.mark.asyncio
    async def test_voice_message_replaced_with_transcript(self) -> None:
        mock_provider = AsyncMock(spec=TranscriptionProvider)
        mock_provider.transcribe = AsyncMock(return_value="Hi, this is a test.")
        channel, published = _make_whatsapp_channel(transcription_provider=mock_provider)

        await channel._handle_bridge_message(_audio_bridge_message())

        assert len(published) == 1
        assert published[0]["content"] == "Hi, this is a test."
        mock_provider.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_voice_message_uses_fallback_on_transcription_error(self) -> None:
        mock_provider = AsyncMock(spec=TranscriptionProvider)
        mock_provider.transcribe = AsyncMock(return_value="[Voice message - transcription failed]")
        channel, published = _make_whatsapp_channel(transcription_provider=mock_provider)

        await channel._handle_bridge_message(_audio_bridge_message())

        assert len(published) == 1
        assert published[0]["content"] == "[Voice message - transcription failed]"

    @pytest.mark.asyncio
    async def test_no_provider_returns_voice_message_placeholder(self) -> None:
        """When no transcription provider is injected, output is [Voice Message]."""
        channel, published = _make_whatsapp_channel(transcription_provider=None)

        await channel._handle_bridge_message(_audio_bridge_message())

        assert len(published) == 1
        assert published[0]["content"] == "[Voice Message]"

    @pytest.mark.asyncio
    async def test_normal_text_message_not_transcribed(self) -> None:
        mock_provider = AsyncMock(spec=TranscriptionProvider)
        mock_provider.transcribe = AsyncMock(side_effect=AssertionError("should not be called"))
        channel, published = _make_whatsapp_channel(transcription_provider=mock_provider)

        raw = json.dumps({
            "type": "message",
            "id": "msg-002",
            "sender": "123@s.whatsapp.net",
            "pn": "",
            "content": "Hello there",
            "timestamp": 1700000001,
            "isGroup": False,
        })

        await channel._handle_bridge_message(raw)

        assert len(published) == 1
        assert published[0]["content"] == "Hello there"

    @pytest.mark.asyncio
    async def test_invalid_base64_returns_failure_sentinel(self) -> None:
        mock_provider = AsyncMock(spec=TranscriptionProvider)
        channel, published = _make_whatsapp_channel(transcription_provider=mock_provider)

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
    async def test_provider_receives_correct_args(self) -> None:
        """Channel must pass mime_type and duration_seconds from bridge payload to provider."""
        mock_provider = AsyncMock(spec=TranscriptionProvider)
        mock_provider.transcribe = AsyncMock(return_value="ok")
        channel, _ = _make_whatsapp_channel(transcription_provider=mock_provider)

        await channel._handle_bridge_message(
            _audio_bridge_message(mimetype="audio/mp4", duration=42.5)
        )

        call_kwargs = mock_provider.transcribe.call_args.kwargs
        assert call_kwargs["mime_type"] == "audio/mp4"
        assert call_kwargs["duration_seconds"] == 42.5
