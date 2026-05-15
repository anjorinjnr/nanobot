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

Retry aggregation: :func:`retry_burst_context` (used by
:class:`LLMProvider._run_with_retry`) buffers per-attempt event props so
the wrapper can ship ONE consolidated ``$ai_generation`` with
``retry_count: N`` instead of N independent events that would inflate
P95/P99 latency dashboards. See issue #52.
"""

from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from nanobot.analytics.pricing import estimate_cost_usd

logger = logging.getLogger(__name__)


_VALID_TASK_KINDS = {
    "chat",
    "heartbeat_system",
    "heartbeat_user",
    "tool_classifier",
    "cron",
}


# Provider prefixes that nanobot/litellm prepend to bare API model names
# (e.g. ``gemini-2.5-flash`` arrives as ``gemini/gemini-2.5-flash`` from
# config but Anthropic SDK returns the bare name from ``response.model``).
# We strip these on emit so PostHog doesn't split a single model into two
# rows. See issue #51.
_KNOWN_PROVIDER_PREFIXES: tuple[str, ...] = (
    "gemini/",
    "openrouter/",
    "cerebras/",
    "anthropic/",
    "openai/",
)


def canonicalize_for_telemetry(model: str) -> str:
    """Return the bare canonical model name for ``$ai_model`` emission.

    Strips any leading provider prefix (``gemini/``, ``openrouter/``,
    ``cerebras/``, ``anthropic/``, ``openai/``) so PostHog dashboards see a
    single dimension per model regardless of whether the caller passed the
    config form (``gemini/gemini-2.5-flash``) or the bare API form
    (``gemini-2.5-flash``). Provider attribution lives in the separate
    ``$ai_provider`` property, so this collapse is non-lossy.

    No-op for models without a known prefix.
    """
    if not model:
        return ""
    for prefix in _KNOWN_PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


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
# Retry-aggregation buffer. When :func:`retry_burst_context` is active,
# per-attempt :func:`track_llm_generation` calls append their props dict
# to this buffer instead of emitting; the retry wrapper emits one
# consolidated event with ``retry_count: N`` once the loop exits. See
# issue #52 — without this each retry attempt fires its own event, which
# inflates P95/P99 latency dashboards with retry-wait time.
_CTX_RETRY_BUFFER: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "homer_llm_retry_buffer", default=None,
)


def current_task_kind() -> str:
    """Return the active task_kind, defaulting to ``chat``."""
    return _CTX_TASK_KIND.get() or "chat"


def current_is_synthetic() -> bool:
    return _CTX_IS_SYNTHETIC.get()


@contextmanager
def retry_burst_context() -> Iterator[list[dict[str, Any]]]:
    """Buffer per-attempt events until the retry loop exits.

    Yields a list that ``track_llm_generation`` will append attempt props
    to (instead of firing them as their own events). The retry wrapper
    inspects the list afterwards to emit ONE consolidated
    ``$ai_generation`` event with ``retry_count`` set to ``len(buffer)``.

    See issue #52 — keeps dashboards reporting user-perceived latency, not
    sum-of-retries.
    """
    buf: list[dict[str, Any]] = []
    tok = _CTX_RETRY_BUFFER.set(buf)
    try:
        yield buf
    finally:
        _CTX_RETRY_BUFFER.reset(tok)


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
    model_served: str | None = None,
    cost_usd_served: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Fire one ``$ai_generation`` event. Fire-and-forget.

    Errors during emission are swallowed — observability MUST NOT crash
    the agent loop.

    Inside a :func:`retry_burst_context` window, the per-attempt props
    are buffered (not emitted) so the retry wrapper can ship ONE
    aggregated event with ``retry_count: N``.
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
            "$ai_model": canonicalize_for_telemetry(model),
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
        # `$ai_model` is what we asked for; `$ai_model_served` is what
        # the provider actually routed to (OpenRouter exposes this via
        # `response.model`). Diverges when OR's auto-router or a
        # fallback chain substitutes — the only way to answer "which
        # generation actually used GPT-class compute" after the fact.
        # Emit only when it actually differs to keep payloads tight.
        served_norm = canonicalize_for_telemetry(model_served) if model_served else None
        if served_norm and served_norm != props["$ai_model"]:
            props["$ai_model_served"] = served_norm
        # `$ai_cost_usd_served` is the provider's authoritative dollar
        # charge (OpenRouter populates `usage.cost`). Strictly better
        # than our pricing-table estimate when present — no maintenance,
        # accounts for promo credits / volume tiers / route price
        # changes. Keep both: `$ai_total_cost_usd` (estimate) is the
        # universal fallback for providers that don't report cost.
        if cost_usd_served is not None:
            props["$ai_cost_usd_served"] = round(float(cost_usd_served), 8)

        hid = _household_id()
        if hid:
            props["household_id"] = hid

        if extra:
            for k, v in extra.items():
                # PostHog reserves $-prefixed keys; never let callers leak
                # arbitrary $-keys past the small allowlist defined above.
                if not k.startswith("$") and k not in props:
                    props[k] = v

        # Retry-burst aggregation: if a buffer is active, stash this
        # attempt's props for the retry wrapper to consolidate. See #52.
        buf = _CTX_RETRY_BUFFER.get()
        if buf is not None:
            buf.append(props)
            return

        # Default emission path (no retry burst) — fire the event with
        # an explicit retry_count=1 so dashboards can sum cleanly across
        # both the default and retry-aggregated paths.
        props.setdefault("retry_count", 1)
        _emit_event(props)
    except Exception as exc:  # noqa: BLE001 — observability must not crash callers
        logger.debug("track_llm_generation failed: %s", exc)


def emit_retry_aggregate(props: dict[str, Any]) -> None:
    """Emit one ``$ai_generation`` for an aggregated retry burst.

    Used by :class:`LLMProvider._run_with_retry` to ship a single event
    after a buffer of per-attempt props has been collapsed into a final
    record. Caller is responsible for setting ``retry_count`` and any
    aggregated latency / error fields on ``props`` before calling.
    """
    try:
        _emit_event(props)
    except Exception as exc:  # noqa: BLE001 — observability never crashes the loop
        logger.debug("emit_retry_aggregate failed: %s", exc)


def _emit_event(props: dict[str, Any]) -> None:
    """Send one ``$ai_generation`` capture call.

    Centralized so the default emission path and the retry-aggregation
    path share the same posthog wiring (init guard, distinct_id, group
    identify).
    """
    try:
        hid = _household_id()
        from nanobot.analytics.hook import get_analytics_hook

        hook = get_analytics_hook()
        hook.capture("$ai_generation", props, distinct_id=_distinct_id_for_call())
        if hid and hook._client is not None:
            hook._client.group_identify("household", hid, {})
    except Exception as exc:  # noqa: BLE001 — observability must not crash callers
        logger.debug("_emit_event failed: %s", exc)


__all__ = [
    "estimate_cost_usd",
    "llm_telemetry_context",
    "current_task_kind",
    "current_is_synthetic",
    "track_llm_generation",
    "canonicalize_for_telemetry",
    "retry_burst_context",
    "emit_retry_aggregate",
]
