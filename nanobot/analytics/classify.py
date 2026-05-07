"""Use-case classifier — Gemini Flash with LRU cache.

Produces a snake_case tag per message. The LLM picks from a preferred set
when a message fits, and otherwise generates its own descriptive tag.
"other" is not a valid output — if classification fails for a technical
reason (no API key, network error, malformed response) we return
"unclassified" so dashboard filters can distinguish "model couldn't decide"
from "pipeline broke".
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Tags the LLM is nudged toward. Not a whitelist — the LLM may return a
# different snake_case tag when none of these fit. Add here when a newly
# generated tag becomes common enough to stabilize.
PREFERRED_TAGS: tuple[str, ...] = (
    "calendar",
    "events",
    "meal_planning",
    "maintenance",
    "health",
    "finance",
    "email",
    "tasks_reminders",
    "research",
    "travel",
    "people",
    "admin",
    "chitchat",
)

_PROMPT_TEMPLATE = (
    "Classify this message to a household AI assistant. Return EXACTLY ONE "
    "lowercase snake_case tag. No punctuation, no quotes, no explanation.\n\n"
    "Prefer these tags when they fit:\n"
    "{preferred}\n\n"
    "If none fit, generate your own descriptive snake_case tag "
    "(1-3 words joined by _). Do NOT return 'other' — pick something "
    "specific instead.\n\n"
    'Message: "{text}"'
)

# Accept any snake_case token starting with a letter. Min length 3 rejects
# truncated outputs ("ch" from "chitchat" when max_tokens cuts mid-token).
# Upper bound keeps pathological LLM output (prompt injections, run-on
# sentences) out of the analytics stream.
_TAG_RE = re.compile(r"^[a-z][a-z0-9_]{2,29}$")
_FALLBACK = "unclassified"


class _LRUCache:
    """Simple ordered-dict LRU cache."""

    def __init__(self, maxsize: int = 500):
        self._data: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> str | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        else:
            if len(self._data) >= self._maxsize:
                self._data.popitem(last=False)
        self._data[key] = value


_cache = _LRUCache()


def _message_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _validate(raw: str) -> str:
    """Return a valid snake_case tag, or _FALLBACK if the input is malformed
    or is literally "other" (which we actively reject)."""
    tag = raw.strip().strip('"').strip("'").strip().lower()
    if tag == "other":
        return _FALLBACK
    if _TAG_RE.fullmatch(tag):
        return tag
    return _FALLBACK


async def classify_message_async(text: str) -> str:
    """Classify *text* into a snake_case use-case tag. Async version.

    Returns a preferred tag, a model-generated snake_case tag, or
    "unclassified" on technical failure. Results are cached by SHA-256 of
    the input text.
    """
    key = _message_hash(text)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        tag = await _call_gemini_async(text)
    except Exception:
        logger.debug("Classification failed", exc_info=True)
        tag = _FALLBACK
    _cache.put(key, tag)
    return tag


def _resolve_api_key() -> str:
    """Pick the API key for the analytics classifier.

    Hosted tenants set tenant-owned ``GEMINI_API_KEY`` for chat. We don't
    want to charge tenants for Homer's classifier, and not every tenant
    even uses Gemini for chat — when they don't, the classifier silently
    fails and every event ships as ``unclassified``. Prefer the
    Homer-owned ``HOMER_ANALYTICS_GEMINI_API_KEY`` injected by the portal,
    fall back to ``GEMINI_API_KEY`` so dev/local with a single key still
    works.
    """
    return (
        os.environ.get("HOMER_ANALYTICS_GEMINI_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )


async def _call_gemini_async(text: str) -> str:
    """Call Gemini Flash and return a validated tag.

    Emits one ``$ai_generation`` event tagged ``task_kind=tool_classifier``
    so spend on this side-channel call shows up correctly in PostHog
    dashboards. This path bypasses :class:`LLMProvider`, so we wire
    telemetry in directly. (#55)
    """
    api_key = _resolve_api_key()
    if not api_key:
        return _FALLBACK
    start = time.monotonic()
    input_tokens = 0
    output_tokens = 0
    is_error = False
    http_status: int | None = None
    try:
        import httpx

        prompt = _PROMPT_TEMPLATE.format(
            preferred=", ".join(PREFERRED_TAGS),
            text=text[:500],
        )
        # Gemini 2.5 Flash uses extended thinking by default. In OpenAI-compat
        # mode the thinking tokens count against `max_tokens`, so a low cap
        # eats the actual answer. `reasoning_effort: "none"` disables thinking
        # for this call (we don't need it for a one-tag classification), and
        # max_tokens is bumped to a comfortable margin for the longest tag.
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gemini-2.5-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0,
                    "reasoning_effort": "none",
                },
                timeout=5.0,
            )
            http_status = resp.status_code
            resp.raise_for_status()
            payload = resp.json()
            raw = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens", 0) or 0)
            output_tokens = int(usage.get("completion_tokens", 0) or 0)
            return _validate(raw)
    except Exception:
        is_error = True
        logger.debug("Gemini classification request failed", exc_info=True)
        return _FALLBACK
    finally:
        try:
            from nanobot.analytics.llm_telemetry import track_llm_generation

            track_llm_generation(
                model="gemini-2.5-flash",
                provider="gemini",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=time.monotonic() - start,
                task_kind="tool_classifier",
                is_error=is_error,
                http_status=http_status,
            )
        except Exception:
            logger.debug("classify telemetry emit failed", exc_info=True)
