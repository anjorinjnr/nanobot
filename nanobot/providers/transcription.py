"""Voice transcription providers."""

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
from loguru import logger

# Maximum audio duration to attempt transcription (5 minutes in seconds)
MAX_AUDIO_DURATION_SECONDS = 5 * 60


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Standard interface for voice transcription providers.

    Channels depend only on this interface — they never import or instantiate
    a concrete provider directly.  The provider is constructed by the channel
    manager and injected at startup.
    """

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        duration_seconds: float | None = None,
    ) -> str:
        """Transcribe raw audio bytes.

        Args:
            audio_bytes: Raw audio data.
            mime_type: MIME type of the audio (e.g. "audio/ogg", "audio/mpeg").
            duration_seconds: Optional known duration for length-gating.

        Returns:
            Transcribed text, or a sentinel string on error/overflow.
        """
        ...


class GroqTranscriptionProvider:
    """Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        duration_seconds: float | None = None,
    ) -> str:
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return "[Voice message - transcription failed]"

        if duration_seconds is not None and duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            logger.info(
                "Voice message duration {}s exceeds {}s limit — skipping transcription",
                duration_seconds,
                MAX_AUDIO_DURATION_SECONDS,
            )
            return "[Voice message too long - please type it out]"

        # Derive a filename with the right extension for Groq's multipart upload
        ext = Path(mime_type.split(";")[0].split("/")[-1]).suffix or ".ogg"
        filename = f"audio{ext}"

        try:
            async with httpx.AsyncClient() as client:
                files = {
                    "file": (filename, audio_bytes, mime_type),
                    "model": (None, "whisper-large-v3"),
                }
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = await client.post(
                    self.api_url, headers=headers, files=files, timeout=60.0
                )
                response.raise_for_status()
                return response.json().get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return "[Voice message - transcription failed]"


class GeminiTranscriptionProvider:
    """Voice transcription provider using Google Gemini.

    Uses the google-genai SDK to send raw audio bytes directly to Gemini.
    Audio is never written to disk.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or os.environ.get("VOICE_TRANSCRIPTION_MODEL", "gemini-2.5-flash")

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        duration_seconds: float | None = None,
    ) -> str:
        if not self.api_key:
            logger.warning("Gemini API key not configured for transcription (GEMINI_API_KEY)")
            return "[Voice message - transcription failed]"

        if duration_seconds is not None and duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            logger.info(
                "Voice message duration {}s exceeds {}s limit — skipping transcription",
                duration_seconds,
                MAX_AUDIO_DURATION_SECONDS,
            )
            return "[Voice message too long - please type it out]"

        try:
            import asyncio

            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            prompt = (
                "Transcribe the following audio message exactly as spoken. "
                "Return only the transcript text with no additional commentary."
            )
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

            # google-genai is sync; run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.model,
                    contents=[prompt, audio_part],
                ),
            )
            return (response.text or "").strip()

        except Exception as e:
            logger.error("Gemini transcription error: {}", e)
            return "[Voice message - transcription failed]"


def create_transcription_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> TranscriptionProvider | None:
    """Factory: build a TranscriptionProvider from config/env.

    Args:
        provider_name: "gemini", "groq", or "disabled" (None → read VOICE_TRANSCRIPTION_PROVIDER).
        api_key: Override the provider's API key (None → provider reads from env).
        model: Override model name for Gemini (None → provider reads VOICE_TRANSCRIPTION_MODEL).

    Returns:
        A provider instance, or None if transcription is disabled / provider unknown.
    """
    name = (provider_name or os.environ.get("VOICE_TRANSCRIPTION_PROVIDER", "gemini")).lower()

    if name == "disabled":
        logger.info("Voice transcription disabled")
        return None

    if name == "gemini":
        return GeminiTranscriptionProvider(api_key=api_key, model=model)

    if name == "groq":
        return GroqTranscriptionProvider(api_key=api_key)

    logger.warning("Unknown VOICE_TRANSCRIPTION_PROVIDER '{}' — transcription disabled", name)
    return None
