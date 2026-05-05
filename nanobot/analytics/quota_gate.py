"""Pre-turn weekly token-budget gate for Homer's *default* model tier.

Homer offers three tiers (env ``HOMER_MODEL_TIER``):

* ``byok`` — user supplies their own provider key, no quota.
* ``managed`` — internal/staff tier, no quota.
* ``default`` — Homer-paid hosted tier, has a weekly token budget.

For default-tier households, before each user-initiated turn we ask the
portal whether the household is still under budget. The portal owns the
ledger; this module is just a thin pre-turn HTTP probe.

Contract (returned by :func:`check_token_budget_before_turn`):

* ``None`` — proceed normally. The default for everything except a hard cap-hit.
* ``str`` — the agent loop must SKIP the turn and send this exact string back
  to the user via the channel-send helper. Used only on a confirmed cap-hit
  (HTTP 200, ``{"ok": false}`` payload).

Soft signal: when the household is over the warn threshold but still under
budget, we set ``turn_ctx["quota_warn_pct"] = <int>`` (None / unset = no
warn) and let the turn run; a post-turn hook
(:func:`maybe_append_quota_warn`) appends the warn appendix to the outgoing
reply so the user gets a single coherent message.

Failure mode is **fail-open**: any non-cap-hit outcome (timeout, network
error, unexpected response, missing env) returns ``None`` so a flaky portal
or misconfigured env never blocks legitimate user turns. Quota is a soft
control — the goal is graceful degradation, not gatekeeping.

Synthetic turns (heartbeat ticks, cron reminders, internal self-sends) are
exempt and bail before the network call so background work can never trip
the gate.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import time
from datetime import date, datetime, timezone
from typing import Any, MutableMapping

import httpx

logger = logging.getLogger(__name__)


# ── HTTP client ───────────────────────────────────────────────────────────

# Module-level client, reused across calls to avoid rebuilding the
# connection pool / TLS context on every turn. Lazily initialized so
# tests that monkeypatch ``_request_quota`` never need a live client.
_http_client: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client()
    return _http_client


def _request_quota(
    url: str, headers: dict[str, str], timeout: float,
) -> httpx.Response:
    """Thin internal wrapper around the module-level httpx client.

    Tests patch this function directly to stub the portal response
    without touching the global client. Honors the caller-supplied
    timeout (no per-client default).
    """
    return _get_http_client().get(url, headers=headers, timeout=timeout)


# ── Copy (module constants for easy editing) ──────────────────────────────

# Template used by :func:`format_cap_hit_reply`. The ``{friendly_reset}`` slot
# is filled with a phrase like ``"tomorrow"`` / ``"on Monday"`` / ``"next week"``.
_CAP_HIT_TEMPLATE = (
    "You've used this week's free Homer budget 🏠 It resets {friendly_reset}.\n"
    "To keep going now, add your own AI provider key — settings:\n"
    "https://homer.joybuild.ai/settings/ai-provider"
)

WARN_APPENDIX = (
    "\n\nFYI — you're at {pct}% of this week's free Homer budget. "
    "Settings → AI Provider to add your own key any time."
)


def _friendly_reset(reset_at: str | None) -> str:
    """Translate a ``reset_at`` ISO string into human-friendly cap-hit copy.

    Rules:
    * missing / empty → ``"next week"`` (preserves pre-follow-up tone)
    * unparseable → ``"soon"``
    * in the past → ``"soon"``
    * 1.0–1.999 days away → ``"tomorrow"``
    * 2.0–6.999 days away → ``"on <weekday>"``
    * 7.0+ days away → ``"in N days"`` (defensive — shouldn't happen)

    Distance is computed in UTC against ``datetime.now(timezone.utc)`` and
    floored to whole days, so a reset 1.5 days from now reads as
    ``"tomorrow"`` (not ``"in 2 days"``).

    Date-only strings (``"2026-05-11"``) are treated as midnight UTC.
    """
    if reset_at is None or reset_at == "":
        return "next week"

    parsed: datetime | None = None
    s = reset_at.strip()
    # Tolerate trailing 'Z' (Python 3.11+ fromisoformat handles it, but be defensive).
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(s)
    except ValueError:
        # Try date-only.
        try:
            d = date.fromisoformat(s)
            parsed = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        except ValueError:
            return "soon"

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    delta_days = (parsed - now).total_seconds() / 86400.0

    # Past or essentially now → "soon".
    if delta_days < 0:
        return "soon"

    # Floor to whole days so a 1.5-day delta reads as "tomorrow" (not
    # "in 2 days"). Banker's rounding via int(round(...)) used to push
    # 1.5d UP to 2 — off-by-one in the cap-hit message. Floor matches
    # the user mental model: the reset is N days away iff at least
    # N*24h still remain.
    days = math.floor(delta_days)
    if days <= 0:
        # Future but within ~24h — still "soon".
        return "soon"
    if days == 1:
        return "tomorrow"
    if 2 <= days <= 6:
        return f"on {parsed.strftime('%A')}"
    return f"in {days} days"


def format_cap_hit_reply(reset_at: str | None) -> str:
    """Build the user-facing cap-hit reply string.

    See :func:`_friendly_reset` for the reset-phrase rules.
    """
    return _CAP_HIT_TEMPLATE.format(friendly_reset=_friendly_reset(reset_at))


# The "no reset_at" rendering, exposed as a constant for callers that want
# the default copy without computing a reset phrase.
CAP_HIT_REPLY = format_cap_hit_reply(None)


# ── Config ────────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT_S = 1.5
_DEFAULT_WARN_PCT = 80
_HMAC_FRESHNESS_WINDOW_S = 60  # documented contract for the portal verifier


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _model_tier() -> str:
    return _env("HOMER_MODEL_TIER") or "byok"


def _warn_pct() -> int:
    raw = _env("HOMER_QUOTA_WARN_PCT")
    if not raw:
        return _DEFAULT_WARN_PCT
    try:
        v = int(raw)
    except ValueError:
        return _DEFAULT_WARN_PCT
    # Clamp to a sensible range to avoid foot-guns from misconfig.
    if v < 1:
        return 1
    if v > 100:
        return 100
    return v


# ── HMAC ──────────────────────────────────────────────────────────────────


def _build_hmac_headers(hid: str, key: str) -> dict[str, str]:
    """Return the ``X-Homer-Ts`` / ``X-Homer-Sig`` pair for a quota probe.

    The portal-side verifier uses the exact same canonical-string layout::

        GET\n/api/quotas/{hid}\n{ts}

    HMAC-SHA256, hex-digested. Freshness window enforced by the portal.

    The key itself MUST NOT appear in logs or error messages — see
    :func:`check_token_budget_before_turn` for the redaction discipline.
    """
    ts = str(int(time.time()))
    payload = f"GET\n/api/quotas/{hid}\n{ts}".encode("utf-8")
    sig = hmac.new(key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return {"X-Homer-Ts": ts, "X-Homer-Sig": sig}


# ── Pre-turn hook ─────────────────────────────────────────────────────────


def check_token_budget_before_turn(
    turn_ctx: MutableMapping[str, Any],
) -> str | None:
    """Probe the portal quota ledger before running a turn.

    Returns:
        * ``None`` — proceed with the turn (also the value when over warn
          threshold; in that case ``turn_ctx["quota_warn_pct"]`` is set
          to the integer pct).
        * ``str`` — the cap-hit reply string. Caller must SKIP the agent
          loop and send this back to the user via the channel.

    The function is safe to call multiple times in a turn (idempotent in
    the sense that it doesn't mutate global state and re-runs the probe);
    callers should avoid doing so to keep portal load down, but defensive
    callers won't break anything.
    """
    # 1. Bail on synthetic turns — heartbeat / cron / self-sends never count.
    if turn_ctx.get("is_synthetic"):
        return None

    # 2. Only the default tier has a budget. byok / managed / unset → skip.
    if _model_tier() != "default":
        return None

    portal = _env("PORTAL_BASE_URL").rstrip("/")
    hid = _env("HOMER_HOUSEHOLD_ID")
    key = _env("HOMER_QUOTA_HMAC_KEY")

    # 3. Fail-open on missing env. We log at debug — the absence of these
    #    on a default-tier container is a deploy bug worth surfacing, but
    #    not at the cost of blocking the user.
    if not portal or not hid:
        logger.debug(
            "quota_gate: missing PORTAL_BASE_URL or HOMER_HOUSEHOLD_ID — fail-open",
        )
        return None
    if not key:
        # Mention env name only — never the value.
        logger.warning(
            "quota_gate: HOMER_QUOTA_HMAC_KEY unset on default-tier container — fail-open",
        )
        return None

    url = f"{portal}/api/quotas/{hid}"
    try:
        headers = _build_hmac_headers(hid, key)
        resp = _request_quota(url, headers, _DEFAULT_TIMEOUT_S)
    except httpx.TimeoutException:
        logger.warning("quota_gate: portal timeout — fail-open")
        return None
    except httpx.HTTPError as exc:
        # Don't include the URL with hid+headers in the error stream.
        logger.warning("quota_gate: portal request failed (%s) — fail-open", type(exc).__name__)
        return None
    except Exception as exc:  # noqa: BLE001 — defence in depth, never crash the turn
        logger.warning("quota_gate: unexpected error (%s) — fail-open", type(exc).__name__)
        return None

    if resp.status_code != 200:
        logger.warning(
            "quota_gate: portal returned %d — fail-open", resp.status_code,
        )
        return None

    try:
        data = resp.json()
    except ValueError:
        logger.warning("quota_gate: portal returned non-JSON — fail-open")
        return None

    if not isinstance(data, dict):
        logger.warning("quota_gate: portal returned non-object payload — fail-open")
        return None

    ok = data.get("ok")
    if ok is False:
        # Hard cap-hit. The portal may also include used/budget for logging
        # and reset_at (ISO string) for the user-facing copy.
        used = data.get("used")
        budget = data.get("budget")
        reset_at = data.get("reset_at")
        if reset_at is not None and not isinstance(reset_at, str):
            # Defence-in-depth: anything non-string falls back to "next week".
            reset_at = None
        logger.info(
            "quota_gate: cap-hit for household=%s used=%s budget=%s reset_at=%s",
            hid, used, budget, reset_at,
        )
        # Best-effort PostHog beacon — never let it block the reply.
        try:
            _emit_blocked_event(hid=hid, used=used, budget=budget)
        except Exception:
            logger.debug("quota_gate: blocked-event emit failed", exc_info=True)
        return format_cap_hit_reply(reset_at)

    if ok is not True:
        # Schema drift — treat as fail-open.
        logger.warning("quota_gate: unexpected ok=%r — fail-open", ok)
        return None

    # ok=True branch: check warn threshold.
    used = data.get("used")
    budget = data.get("budget")
    try:
        used_n = float(used)
        budget_n = float(budget)
    except (TypeError, ValueError):
        return None
    if budget_n <= 0:
        return None

    pct_used = (used_n / budget_n) * 100.0
    threshold = _warn_pct()
    if pct_used >= threshold:
        # Single Optional[int] flag: None / unset = no warn,
        # int = warn pct (rounded for display). The post-turn hook reads
        # this directly and interpolates into WARN_APPENDIX.
        turn_ctx["quota_warn_pct"] = int(round(pct_used))
    return None


# ── Post-turn hook ────────────────────────────────────────────────────────


def maybe_append_quota_warn(
    turn_ctx: MutableMapping[str, Any], reply: str | None,
) -> str | None:
    """Append :data:`WARN_APPENDIX` to *reply* iff the pre-turn hook set the pct.

    Single-message UX: we keep the agent's natural reply intact and tack the
    nudge on at the end so the user reads one coherent message instead of
    two.

    No-ops if:

    * ``turn_ctx["quota_warn_pct"]`` is None / unset / not an int,
    * the reply is empty (no message to ride on),
    * the appendix is already present (defensive idempotency).
    """
    if reply is None:
        return reply
    pct = turn_ctx.get("quota_warn_pct")
    if not isinstance(pct, int):
        return reply
    appendix = WARN_APPENDIX.format(pct=pct)
    if appendix in reply:  # defensive idempotency
        return reply
    return reply + appendix


# ── PostHog beacon (best-effort) ──────────────────────────────────────────


def _emit_blocked_event(*, hid: str, used: Any, budget: Any) -> None:
    """Fire a ``quota_gate_blocked`` event for visibility.

    Lazy import + try/except guard mirrors the analytics-hook discipline
    elsewhere in this package: telemetry must never crash the agent loop.
    """
    from nanobot.analytics.hook import get_analytics_hook

    hook = get_analytics_hook()
    props: dict[str, Any] = {"household_id": hid}
    if used is not None:
        props["used"] = used
    if budget is not None:
        props["budget"] = budget
    hook.capture("quota_gate_blocked", props, distinct_id=hid or "system")
    if hid and hook._client is not None:
        hook._client.group_identify("household", hid, {})


__all__ = [
    "CAP_HIT_REPLY",
    "WARN_APPENDIX",
    "check_token_budget_before_turn",
    "format_cap_hit_reply",
    "maybe_append_quota_warn",
]
