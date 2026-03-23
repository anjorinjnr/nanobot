"""Voice transcription providers."""

import os
from pathlib import Path

import httpx
from loguru import logger

# Maximum audio duration to attempt transcription (5 minutes in seconds)
MAX_AUDIO_DURATION_SECONDS = 5 * 60


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""


class GeminiTranscriptionProvider:
    """
    Voice transcription provider using Google Gemini.

    Uses the google-genai SDK to send raw audio bytes directly to Gemini
    for transcription. Audio is never written to disk.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or os.environ.get("VOICE_TRANSCRIPTION_MODEL", "gemini-2.5-flash")

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        duration_seconds: float | None = None,
    ) -> str:
        """
        Transcribe raw audio bytes using Gemini.

        Audio is passed entirely in-memory — nothing is written to disk.

        Args:
            audio_bytes: Raw audio data.
            mime_type: MIME type of the audio (e.g. "audio/ogg", "audio/mpeg").
            duration_seconds: Optional known duration for length-gating.

        Returns:
            Transcribed text, or a sentinel string on error/overflow.
        """
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

            transcript = response.text or ""
            return transcript.strip()

        except Exception as e:
            logger.error("Gemini transcription error: {}", e)
            return "[Voice message - transcription failed]"
