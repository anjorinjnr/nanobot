"""Tests for the per-LLM-call ``$ai_generation`` telemetry hook.

These tests stub the PostHog client baked into :class:`AnalyticsHook` so we
can inspect every emitted event without touching the network. They cover:

* end-to-end emission via ``LLMProvider._safe_chat`` and ``_safe_chat_stream``
* error path (provider raises) → ``$ai_is_error=True``
* heartbeat task_kind context → ``heartbeat_system`` / ``heartbeat_user``
* synthetic flag honored
* PII regression: no email / 10-digit number / >200-char free-form props

Cost-table behavior is exercised in ``test_pricing.py``.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.analytics import hook as hook_module
from nanobot.analytics.llm_telemetry import (
    llm_telemetry_context,
    track_llm_generation,
)
from nanobot.providers.base import LLMProvider, LLMResponse


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip Homer-controlled env vars so each test starts deterministic."""
    for key in ("HOMER_MODEL_TIER", "HOMER_HOUSEHOLD_ID", "POSTHOG_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_hook(monkeypatch):
    """Stub ``get_analytics_hook()`` so emit lands on a MagicMock client."""
    fake = MagicMock()
    fake._initialized = True
    fake._client = MagicMock()
    fake._ensure_init = MagicMock(return_value=True)

    monkeypatch.setattr(hook_module, "_hook", fake)
    return fake


class _StubProvider(LLMProvider):
    """Minimal concrete provider for exercising _safe_chat / _safe_chat_stream."""

    def __init__(self, response: LLMResponse | None = None, raise_exc: Exception | None = None):
        super().__init__()
        self.provider_name = "anthropic"
        self.default_model = "claude-haiku-4-5-20251001"
        self._response = response
        self._raise_exc = raise_exc

    async def chat(self, *args, **kwargs):  # noqa: D401, ANN001 — match base
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response

    def get_default_model(self) -> str:
        return self.default_model


def _last_event(client: MagicMock) -> tuple[str, dict[str, Any]]:
    """Return (event_name, props) for the last capture() call."""
    assert client.capture.called, "expected at least one capture() call"
    args, kwargs = client.capture.call_args
    # capture(distinct_id, event_name, props)
    return args[1], args[2]


# ── direct track_llm_generation ───────────────────────────────────────────


def test_track_emits_event_with_full_schema(mock_hook, monkeypatch):
    monkeypatch.setenv("HOMER_MODEL_TIER", "tier1")
    monkeypatch.setenv("HOMER_HOUSEHOLD_ID", "hh-test")

    track_llm_generation(
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=500,
        latency_s=1.234567,
        task_kind="chat",
    )

    event, props = _last_event(mock_hook._client)
    assert event == "$ai_generation"
    assert props["$ai_model"] == "claude-haiku-4-5-20251001"
    assert props["$ai_provider"] == "anthropic"
    assert props["$ai_input_tokens"] == 1000
    assert props["$ai_output_tokens"] == 500
    assert props["$ai_cache_read_input_tokens"] == 0
    # 1000 in @ $1/MTok + 500 out @ $5/MTok = 0.001 + 0.0025 = 0.0035
    assert props["$ai_total_cost_usd"] == pytest.approx(0.0035, abs=1e-9)
    assert props["$ai_latency"] == pytest.approx(1.2346, abs=1e-9)
    assert props["$ai_is_error"] is False
    assert props["task_kind"] == "chat"
    assert props["tier"] == "tier1"
    assert props["household_id"] == "hh-test"
    assert props["is_synthetic"] is False


def test_track_default_tier_is_byok(mock_hook):
    track_llm_generation(
        model="x", provider="y", input_tokens=0, output_tokens=0,
        latency_s=0.1, task_kind="chat",
    )
    _, props = _last_event(mock_hook._client)
    assert props["tier"] == "byok"


def test_track_omits_household_when_unset(mock_hook):
    track_llm_generation(
        model="x", provider="y", input_tokens=0, output_tokens=0,
        latency_s=0.1, task_kind="chat",
    )
    _, props = _last_event(mock_hook._client)
    assert "household_id" not in props


def test_track_unknown_task_kind_tagged_not_dropped(mock_hook):
    track_llm_generation(
        model="x", provider="y", input_tokens=0, output_tokens=0,
        latency_s=0.1, task_kind="weird_kind",
    )
    _, props = _last_event(mock_hook._client)
    assert props["task_kind"] == "chat"
    assert props["unknown_task_kind"] is True


def test_track_skips_emit_when_posthog_disabled(monkeypatch):
    fake = MagicMock()
    fake._initialized = True
    fake._client = MagicMock()
    fake._ensure_init = MagicMock(return_value=False)
    monkeypatch.setattr(hook_module, "_hook", fake)

    track_llm_generation(
        model="x", provider="y", input_tokens=0, output_tokens=0,
        latency_s=0.1, task_kind="chat",
    )
    fake._client.capture.assert_not_called()


def test_track_swallows_emit_failure(monkeypatch):
    fake = MagicMock()
    fake._initialized = True
    fake._ensure_init = MagicMock(return_value=True)
    fake._client = MagicMock()
    fake._client.capture.side_effect = RuntimeError("posthog down")
    monkeypatch.setattr(hook_module, "_hook", fake)

    # Must not raise — observability never crashes the loop.
    track_llm_generation(
        model="x", provider="y", input_tokens=0, output_tokens=0,
        latency_s=0.1, task_kind="chat",
    )


# ── context propagation ───────────────────────────────────────────────────


def test_heartbeat_context_threads_task_kind(mock_hook):
    with llm_telemetry_context(task_kind="heartbeat_system", is_synthetic=True):
        track_llm_generation(
            model="x", provider="y", input_tokens=0, output_tokens=0,
            latency_s=0.1,
        )
    _, props = _last_event(mock_hook._client)
    assert props["task_kind"] == "heartbeat_system"
    assert props["is_synthetic"] is True


def test_explicit_task_kind_wins_over_context(mock_hook):
    with llm_telemetry_context(task_kind="heartbeat_user", is_synthetic=True):
        track_llm_generation(
            model="x", provider="y", input_tokens=0, output_tokens=0,
            latency_s=0.1, task_kind="tool_classifier",
        )
    _, props = _last_event(mock_hook._client)
    assert props["task_kind"] == "tool_classifier"


# ── provider integration ──────────────────────────────────────────────────


async def test_safe_chat_emits_one_event_on_success(mock_hook):
    response = LLMResponse(
        content="hi",
        finish_reason="stop",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "cached_tokens": 20},
    )
    provider = _StubProvider(response=response)

    await provider._safe_chat(model="claude-haiku-4-5-20251001")

    assert mock_hook._client.capture.call_count == 1
    event, props = _last_event(mock_hook._client)
    assert event == "$ai_generation"
    assert props["$ai_input_tokens"] == 100
    assert props["$ai_output_tokens"] == 50
    assert props["$ai_cache_read_input_tokens"] == 20
    assert props["$ai_provider"] == "anthropic"
    assert props["$ai_is_error"] is False
    assert props["task_kind"] == "chat"


async def test_safe_chat_emits_event_on_provider_exception(mock_hook):
    provider = _StubProvider(raise_exc=RuntimeError("upstream 500"))

    response = await provider._safe_chat(model="claude-haiku-4-5-20251001")

    assert response.finish_reason == "error"
    assert "upstream 500" in (response.content or "")
    event, props = _last_event(mock_hook._client)
    assert event == "$ai_generation"
    assert props["$ai_is_error"] is True
    # No usage on a thrown exception → counts default to 0.
    assert props["$ai_input_tokens"] == 0
    assert props["$ai_output_tokens"] == 0


async def test_safe_chat_stream_emits_event(mock_hook):
    response = LLMResponse(
        content="streamed",
        finish_reason="stop",
        usage={"prompt_tokens": 7, "completion_tokens": 3},
    )
    provider = _StubProvider(response=response)

    await provider._safe_chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-haiku-4-5-20251001",
    )

    assert mock_hook._client.capture.call_count == 1
    _, props = _last_event(mock_hook._client)
    assert props["$ai_input_tokens"] == 7
    assert props["$ai_output_tokens"] == 3


async def test_safe_chat_inside_heartbeat_context(mock_hook):
    response = LLMResponse(
        content="ok", finish_reason="stop",
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )
    provider = _StubProvider(response=response)

    with llm_telemetry_context(task_kind="heartbeat_system", is_synthetic=True):
        await provider._safe_chat(model="claude-haiku-4-5-20251001")

    _, props = _last_event(mock_hook._client)
    assert props["task_kind"] == "heartbeat_system"
    assert props["is_synthetic"] is True


async def test_safe_chat_records_http_status_on_structured_error(mock_hook):
    response = LLMResponse(
        content="rate limited",
        finish_reason="error",
        usage={},
        error_status_code=429,
    )
    provider = _StubProvider(response=response)

    await provider._safe_chat(model="claude-haiku-4-5-20251001")

    _, props = _last_event(mock_hook._client)
    assert props["$ai_is_error"] is True
    assert props["$ai_http_status"] == 429


# ── PII regression ────────────────────────────────────────────────────────


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b\d{10,}\b")


def _sweep_props(props: dict[str, Any]) -> list[str]:
    """Return all string-typed property values for inspection."""
    out: list[str] = []
    for v in props.values():
        if isinstance(v, str):
            out.append(v)
    return out


async def test_pii_regression_no_email_or_phone_or_long_strings(mock_hook, monkeypatch):
    """The shape MUST never carry free-form text. Only counts, IDs, flags.

    This mirrors homer's ``test_pii_regression_no_email_or_phone_in_props``
    and is the load-bearing privacy guarantee for $ai_generation.
    """
    # Plant tempting values everywhere that *could* leak.
    monkeypatch.setenv("HOMER_HOUSEHOLD_ID", "hh-1234")
    monkeypatch.setenv("HOMER_MODEL_TIER", "tier1")

    response = LLMResponse(
        content="user said: alice@example.com and 5551234567",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    provider = _StubProvider(response=response)

    await provider._safe_chat(model="claude-haiku-4-5-20251001")

    _, props = _last_event(mock_hook._client)

    for value in _sweep_props(props):
        assert not _EMAIL_RE.search(value), (
            f"email-shaped substring leaked into telemetry: {value!r}"
        )
        assert not _PHONE_RE.search(value), (
            f"10+ digit number leaked into telemetry: {value!r}"
        )
        assert len(value) <= 200, (
            f"free-form-looking string > 200 chars leaked: {value[:80]!r}..."
        )
