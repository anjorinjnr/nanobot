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
budget, we set ``turn_ctx["quota_warn"] = True`` and let the turn run; a
post-turn hook (:func:`maybe_append_quota_warn`) appends the warn appendix
to the outgoing reply so the user gets a single coherent message.

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
import os
import time
from typing import Any, MutableMapping

import httpx

logger = logging.getLogger(__name__)


# ── Copy (module constants for easy editing) ──────────────────────────────

CAP_HIT_REPLY = (
    "You've used this week's free Homer budget 🏠 To keep going, add your "
    "own AI provider key — settings: https://homer.joybuild.ai/settings/ai-provider"
)

WARN_APPENDIX = (
    "\n\nFYI — you're at {pct}% of this week's free Homer budget. "
    "Settings → AI Provider to add your own key any time."
)


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
          threshold; in that case ``turn_ctx["quota_warn"]`` is set).
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
        resp = httpx.get(url, headers=headers, timeout=_DEFAULT_TIMEOUT_S)
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
        # Hard cap-hit. The portal may also include used/budget for logging.
        used = data.get("used")
        budget = data.get("budget")
        logger.info(
            "quota_gate: cap-hit for household=%s used=%s budget=%s",
            hid, used, budget,
        )
        # Best-effort PostHog beacon — never let it block the reply.
        try:
            _emit_blocked_event(hid=hid, used=used, budget=budget)
        except Exception:
            logger.debug("quota_gate: blocked-event emit failed", exc_info=True)
        return CAP_HIT_REPLY

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
        # Round to int for display; expose pct so the post-turn hook can
        # interpolate into WARN_APPENDIX without re-fetching.
        turn_ctx["quota_warn"] = True
        turn_ctx["quota_warn_pct"] = int(round(pct_used))
    return None


# ── Post-turn hook ────────────────────────────────────────────────────────


def maybe_append_quota_warn(
    turn_ctx: MutableMapping[str, Any], reply: str | None,
) -> str | None:
    """Append :data:`WARN_APPENDIX` to *reply* iff the pre-turn hook set the flag.

    Single-message UX: we keep the agent's natural reply intact and tack the
    nudge on at the end so the user reads one coherent message instead of
    two.

    No-ops if:

    * the warn flag isn't set,
    * the reply is empty (no message to ride on),
    * the appendix is already present (defensive idempotency).
    """
    if reply is None:
        return reply
    if not turn_ctx.get("quota_warn"):
        return reply
    pct = turn_ctx.get("quota_warn_pct")
    if not isinstance(pct, int):
        return reply
    appendix = WARN_APPENDIX.format(pct=pct)
    if appendix.strip() and appendix.strip() in reply:
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
    if not hook._ensure_init():
        return
    client = hook._client
    props: dict[str, Any] = {"household_id": hid}
    if used is not None:
        props["used"] = used
    if budget is not None:
        props["budget"] = budget
    client.capture(hid or "system", "quota_gate_blocked", props)
    if hid:
        client.group_identify("household", hid, {})


__all__ = [
    "CAP_HIT_REPLY",
    "WARN_APPENDIX",
    "check_token_budget_before_turn",
    "maybe_append_quota_warn",
]
