"""Per-call PostHog ``$ai_generation`` telemetry for the agent loop.

Every LLM completion (chat / chat_stream, all providers) emits one event
with the post-fallback model, provider, token counts, latency, and a USD
cost estimate. Properties match Homer's ``tools/analytics/llm_call.py``
schema (PostHog LLM Analytics ``$ai_*`` namespace) so both producers feed
the same dashboards.

Privacy contract: prompt and completion content never enter this module.
Only counts, model identifiers, latency, cost, and a small set of context
flags (task_kind, tier, household_id) are emitted.

Cost estimation lives in :mod:`nanobot.analytics.pricing`; this module
re-exports :func:`estimate_cost_usd` for back-compat. Update prices there
— Homer's ``tools/analytics/llm_call.py`` imports the same table.

Heartbeat / cron callers set ``task_kind`` via ``llm_telemetry_context()``
before invoking the agent loop; the LLM-call boundary in
:class:`LLMProvider` reads it via :func:`current_task_kind`. Inbound user
messages leave the contextvar unset so events default to ``chat``.
"""

from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from nanobot.analytics.pricing import estimate_cost_usd

logger = logging.getLogger(__name__)


_VALID_TASK_KINDS = {"chat", "heartbeat_system", "heartbeat_user", "tool_classifier"}


# ── task_kind / synthetic context ─────────────────────────────────────────


@contextmanager
def llm_telemetry_context(
    *, task_kind: str, is_synthetic: bool = False,
) -> Iterator[None]:
    """Tag every LLM call inside this block with a known task_kind.

    Heartbeat dispatch wraps each task group in this context manager so
    the per-call telemetry fires with ``heartbeat_system`` /
    ``heartbeat_user`` instead of the default ``chat``.
    """
    t1 = _CTX_TASK_KIND.set(task_kind)
    t2 = _CTX_IS_SYNTHETIC.set(bool(is_synthetic))
    try:
        yield
    finally:
        _CTX_IS_SYNTHETIC.reset(t2)
        _CTX_TASK_KIND.reset(t1)


_CTX_TASK_KIND: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "homer_llm_task_kind", default=None,
)
_CTX_IS_SYNTHETIC: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "homer_llm_is_synthetic", default=False,
)


def current_task_kind() -> str:
    """Return the active task_kind, defaulting to ``chat``."""
    return _CTX_TASK_KIND.get() or "chat"


def current_is_synthetic() -> bool:
    return _CTX_IS_SYNTHETIC.get()


# ── Event emission ────────────────────────────────────────────────────────


def _model_tier() -> str:
    return os.environ.get("HOMER_MODEL_TIER", "byok")


def _household_id() -> str:
    return os.environ.get("HOMER_HOUSEHOLD_ID", "")


def _distinct_id_for_call() -> str:
    """Per-LLM-call distinct id.

    Tool/heartbeat-side calls don't have an inbound user, so we attribute
    to the household. Mirrors how Homer's tools side does it.
    """
    hid = _household_id()
    return hid or "system"


def track_llm_generation(
    *,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    latency_s: float,
    task_kind: str | None = None,
    cache_read_tokens: int = 0,
    is_error: bool = False,
    http_status: int | None = None,
    trace_id: str | None = None,
    is_synthetic: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Fire one ``$ai_generation`` event. Fire-and-forget.

    Errors during emission are swallowed — observability MUST NOT crash
    the agent loop.
    """
    try:
        kind = task_kind if task_kind is not None else current_task_kind()
        if kind not in _VALID_TASK_KINDS:
            # Don't drop — tag and keep going so we can find drift in PostHog.
            extra = dict(extra or {})
            extra.setdefault("unknown_task_kind", True)
            kind = "chat"

        cost_usd = estimate_cost_usd(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
        )

        synthetic = current_is_synthetic() if is_synthetic is None else bool(is_synthetic)

        props: dict[str, Any] = {
            "$ai_model": model,
            "$ai_provider": provider,
            "$ai_input_tokens": int(input_tokens),
            "$ai_output_tokens": int(output_tokens),
            "$ai_cache_read_input_tokens": int(cache_read_tokens),
            "$ai_total_cost_usd": round(cost_usd, 8),
            "$ai_latency": round(float(latency_s), 4),
            "$ai_is_error": bool(is_error),
            "task_kind": kind,
            "tier": _model_tier(),
            "is_synthetic": synthetic,
        }

        if http_status is not None:
            props["$ai_http_status"] = int(http_status)
        if trace_id:
            props["$ai_trace_id"] = trace_id

        hid = _household_id()
        if hid:
            props["household_id"] = hid

        if extra:
            for k, v in extra.items():
                # PostHog reserves $-prefixed keys; never let callers leak
                # arbitrary $-keys past the small allowlist defined above.
                if not k.startswith("$") and k not in props:
                    props[k] = v

        from nanobot.analytics.hook import get_analytics_hook

        hook = get_analytics_hook()
        hook.capture("$ai_generation", props, distinct_id=_distinct_id_for_call())
        if hid and hook._client is not None:
            hook._client.group_identify("household", hid, {})
    except Exception as exc:  # noqa: BLE001 — observability must not crash callers
        logger.debug("track_llm_generation failed: %s", exc)


__all__ = [
    "estimate_cost_usd",
    "llm_telemetry_context",
    "current_task_kind",
    "current_is_synthetic",
    "track_llm_generation",
]
