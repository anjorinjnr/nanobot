"""Tests for the pre-turn weekly-quota gate (Phase 3 / Phase 4 copy)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from nanobot.analytics import quota_gate
from nanobot.analytics.quota_gate import (
    CAP_HIT_REPLY,
    WARN_APPENDIX,
    check_token_budget_before_turn,
    format_cap_hit_reply,
    maybe_append_quota_warn,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────


@pytest.fixture
def default_env(monkeypatch):
    """Standard default-tier container env."""
    monkeypatch.setenv("HOMER_MODEL_TIER", "default")
    monkeypatch.setenv("HOMER_HOUSEHOLD_ID", "hh-test-1234")
    monkeypatch.setenv("HOMER_QUOTA_HMAC_KEY", "test-secret-key-32b-long-x" * 1)
    monkeypatch.setenv("PORTAL_BASE_URL", "https://portal.test")
    monkeypatch.delenv("HOMER_QUOTA_WARN_PCT", raising=False)


def _mk_resp(status: int, payload):
    """Build a stand-in httpx.Response object usable by the gate."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    if isinstance(payload, Exception):
        resp.json.side_effect = payload
    else:
        resp.json.return_value = payload
    return resp


# ── Bail-fast cases (no portal call) ──────────────────────────────────────


def test_synthetic_turn_skips_portal(default_env):
    with patch.object(quota_gate.httpx, "get") as mock_get:
        out = check_token_budget_before_turn({"is_synthetic": True})
    assert out is None
    mock_get.assert_not_called()


@pytest.mark.parametrize("tier", ["byok", "managed", ""])
def test_non_default_tier_skips_portal(monkeypatch, tier):
    monkeypatch.setenv("HOMER_MODEL_TIER", tier)
    monkeypatch.setenv("HOMER_HOUSEHOLD_ID", "hh-1")
    monkeypatch.setenv("HOMER_QUOTA_HMAC_KEY", "k")
    monkeypatch.setenv("PORTAL_BASE_URL", "https://portal.test")
    with patch.object(quota_gate.httpx, "get") as mock_get:
        out = check_token_budget_before_turn({})
    assert out is None
    mock_get.assert_not_called()


def test_unset_tier_skips_portal(monkeypatch):
    monkeypatch.delenv("HOMER_MODEL_TIER", raising=False)
    monkeypatch.setenv("HOMER_HOUSEHOLD_ID", "hh-1")
    monkeypatch.setenv("HOMER_QUOTA_HMAC_KEY", "k")
    monkeypatch.setenv("PORTAL_BASE_URL", "https://portal.test")
    with patch.object(quota_gate.httpx, "get") as mock_get:
        out = check_token_budget_before_turn({})
    assert out is None
    mock_get.assert_not_called()


def test_missing_portal_url_fails_open(monkeypatch):
    monkeypatch.setenv("HOMER_MODEL_TIER", "default")
    monkeypatch.setenv("HOMER_HOUSEHOLD_ID", "hh-1")
    monkeypatch.setenv("HOMER_QUOTA_HMAC_KEY", "k")
    monkeypatch.delenv("PORTAL_BASE_URL", raising=False)
    with patch.object(quota_gate.httpx, "get") as mock_get:
        out = check_token_budget_before_turn({})
    assert out is None
    mock_get.assert_not_called()


def test_missing_household_id_fails_open(monkeypatch):
    monkeypatch.setenv("HOMER_MODEL_TIER", "default")
    monkeypatch.delenv("HOMER_HOUSEHOLD_ID", raising=False)
    monkeypatch.setenv("HOMER_QUOTA_HMAC_KEY", "k")
    monkeypatch.setenv("PORTAL_BASE_URL", "https://portal.test")
    with patch.object(quota_gate.httpx, "get") as mock_get:
        out = check_token_budget_before_turn({})
    assert out is None
    mock_get.assert_not_called()


def test_missing_hmac_key_fails_open(monkeypatch):
    monkeypatch.setenv("HOMER_MODEL_TIER", "default")
    monkeypatch.setenv("HOMER_HOUSEHOLD_ID", "hh-1")
    monkeypatch.delenv("HOMER_QUOTA_HMAC_KEY", raising=False)
    monkeypatch.setenv("PORTAL_BASE_URL", "https://portal.test")
    with patch.object(quota_gate.httpx, "get") as mock_get:
        out = check_token_budget_before_turn({})
    assert out is None
    mock_get.assert_not_called()


# ── Cap-hit / warn / under-budget ─────────────────────────────────────────


def test_ok_false_returns_cap_hit_reply(default_env):
    resp = _mk_resp(200, {"ok": False, "used": 1_500_000, "budget": 1_000_000})
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn({})
    assert out == CAP_HIT_REPLY
    # Phase 4 copy expectations.
    assert "free Homer budget" in CAP_HIT_REPLY
    assert "settings" in CAP_HIT_REPLY.lower()


def test_ok_true_at_warn_threshold_sets_flag(default_env):
    resp = _mk_resp(200, {"ok": True, "used": 900_000, "budget": 1_000_000})
    ctx: dict = {}
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn(ctx)
    assert out is None
    assert ctx.get("quota_warn") is True
    assert ctx.get("quota_warn_pct") == 90


def test_ok_true_under_threshold_no_flag(default_env):
    resp = _mk_resp(200, {"ok": True, "used": 500_000, "budget": 1_000_000})
    ctx: dict = {}
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn(ctx)
    assert out is None
    assert "quota_warn" not in ctx
    assert "quota_warn_pct" not in ctx


def test_warn_pct_env_override(default_env, monkeypatch):
    monkeypatch.setenv("HOMER_QUOTA_WARN_PCT", "50")
    resp = _mk_resp(200, {"ok": True, "used": 600_000, "budget": 1_000_000})
    ctx: dict = {}
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn(ctx)
    assert out is None
    assert ctx.get("quota_warn") is True
    assert ctx.get("quota_warn_pct") == 60


# ── Failure modes (fail-open) ─────────────────────────────────────────────


def test_portal_timeout_fails_open(default_env, caplog):
    with patch.object(
        quota_gate.httpx, "get", side_effect=httpx.ConnectTimeout("slow"),
    ), caplog.at_level(logging.WARNING, logger="nanobot.analytics.quota_gate"):
        out = check_token_budget_before_turn({})
    assert out is None
    assert any("timeout" in r.message.lower() for r in caplog.records)


def test_portal_network_error_fails_open(default_env, caplog):
    with patch.object(
        quota_gate.httpx, "get", side_effect=httpx.ConnectError("dns"),
    ), caplog.at_level(logging.WARNING, logger="nanobot.analytics.quota_gate"):
        out = check_token_budget_before_turn({})
    assert out is None
    assert any("portal request failed" in r.message.lower() for r in caplog.records)


def test_portal_non_200_fails_open(default_env, caplog):
    resp = _mk_resp(503, {"err": "down"})
    with patch.object(quota_gate.httpx, "get", return_value=resp), \
            caplog.at_level(logging.WARNING, logger="nanobot.analytics.quota_gate"):
        out = check_token_budget_before_turn({})
    assert out is None
    assert any("503" in r.message for r in caplog.records)


def test_portal_non_json_fails_open(default_env):
    resp = _mk_resp(200, ValueError("bad json"))
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn({})
    assert out is None


def test_portal_non_object_payload_fails_open(default_env):
    resp = _mk_resp(200, ["not", "an", "object"])
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn({})
    assert out is None


def test_unexpected_ok_value_fails_open(default_env):
    resp = _mk_resp(200, {"ok": "maybe"})
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn({})
    assert out is None


def test_zero_budget_does_not_warn(default_env):
    resp = _mk_resp(200, {"ok": True, "used": 100, "budget": 0})
    ctx: dict = {}
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn(ctx)
    assert out is None
    assert "quota_warn" not in ctx


# ── HMAC headers ──────────────────────────────────────────────────────────


def test_hmac_headers_structure(default_env):
    resp = _mk_resp(200, {"ok": True, "used": 1, "budget": 1000})
    with patch.object(quota_gate.httpx, "get", return_value=resp) as mock_get:
        check_token_budget_before_turn({})
    assert mock_get.called
    _, kwargs = mock_get.call_args
    headers = kwargs["headers"]
    assert "X-Homer-Ts" in headers
    assert "X-Homer-Sig" in headers
    # ts is a unix-second integer string within ~60s of now.
    ts = int(headers["X-Homer-Ts"])
    assert abs(ts - int(time.time())) < 60
    # sig is sha256 hex (64 chars).
    sig = headers["X-Homer-Sig"]
    assert len(sig) == 64
    int(sig, 16)  # valid hex


def test_hmac_signature_matches_expected(default_env):
    """Lock the canonical-string layout so the portal verifier can match it."""
    import hashlib
    import hmac as _hmac

    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = dict(headers or {})
        return _mk_resp(200, {"ok": True, "used": 0, "budget": 1000})

    with patch.object(quota_gate.httpx, "get", side_effect=fake_get):
        check_token_budget_before_turn({})

    headers = captured["headers"]
    ts = headers["X-Homer-Ts"]
    sig = headers["X-Homer-Sig"]
    hid = "hh-test-1234"
    key = "test-secret-key-32b-long-x" * 1
    expected_payload = f"GET\n/api/quotas/{hid}\n{ts}".encode("utf-8")
    expected_sig = _hmac.new(
        key.encode("utf-8"), expected_payload, hashlib.sha256,
    ).hexdigest()
    assert sig == expected_sig
    assert captured["url"] == "https://portal.test/api/quotas/hh-test-1234"


def test_hmac_key_not_logged_on_failure(default_env, caplog):
    """The HMAC key must never appear in error/warning output."""
    with patch.object(
        quota_gate.httpx, "get", side_effect=httpx.ConnectError("dns"),
    ), caplog.at_level(logging.DEBUG, logger="nanobot.analytics.quota_gate"):
        check_token_budget_before_turn({})
    secret = "test-secret-key-32b-long-x"
    for r in caplog.records:
        assert secret not in r.getMessage()


# ── Idempotency ───────────────────────────────────────────────────────────


def test_repeated_calls_are_safe(default_env):
    resp = _mk_resp(200, {"ok": True, "used": 950_000, "budget": 1_000_000})
    ctx: dict = {}
    with patch.object(quota_gate.httpx, "get", return_value=resp) as mock_get:
        a = check_token_budget_before_turn(ctx)
        b = check_token_budget_before_turn(ctx)
    assert a is None and b is None
    assert ctx.get("quota_warn") is True
    assert ctx.get("quota_warn_pct") == 95
    # Each call is allowed to probe — but neither should crash nor leak state.
    assert mock_get.call_count == 2


def test_repeated_cap_hit_calls_safe(default_env):
    resp = _mk_resp(200, {"ok": False, "used": 2_000_000, "budget": 1_000_000})
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        a = check_token_budget_before_turn({})
        b = check_token_budget_before_turn({})
    assert a == CAP_HIT_REPLY
    assert b == CAP_HIT_REPLY


# ── Post-turn warn appendix ───────────────────────────────────────────────


def test_warn_appendix_appended_when_flag_set():
    ctx = {"quota_warn": True, "quota_warn_pct": 87}
    out = maybe_append_quota_warn(ctx, "Here's your weather forecast.")
    assert out is not None
    assert out.startswith("Here's your weather forecast.")
    assert "87%" in out
    assert "AI Provider" in out


def test_warn_appendix_noop_when_flag_unset():
    out = maybe_append_quota_warn({}, "Hello.")
    assert out == "Hello."


def test_warn_appendix_noop_on_none_reply():
    ctx = {"quota_warn": True, "quota_warn_pct": 90}
    assert maybe_append_quota_warn(ctx, None) is None


def test_warn_appendix_idempotent():
    ctx = {"quota_warn": True, "quota_warn_pct": 90}
    once = maybe_append_quota_warn(ctx, "hi")
    twice = maybe_append_quota_warn(ctx, once)
    assert once == twice


def test_warn_appendix_format_matches_template():
    ctx = {"quota_warn": True, "quota_warn_pct": 80}
    out = maybe_append_quota_warn(ctx, "x")
    assert out == "x" + WARN_APPENDIX.format(pct=80)


# ── format_cap_hit_reply: friendly reset phrase ───────────────────────────


def _iso_in_days(days: float, *, date_only: bool = False) -> str:
    """Build an ISO string `days` from now (UTC)."""
    when = datetime.now(timezone.utc) + timedelta(days=days)
    if date_only:
        return when.date().isoformat()
    return when.isoformat()


def test_format_cap_hit_reply_one_day_says_tomorrow():
    out = format_cap_hit_reply(_iso_in_days(1.0))
    assert "tomorrow" in out
    assert "free Homer budget" in out
    assert "https://homer.joybuild.ai/settings/ai-provider" in out


def test_format_cap_hit_reply_three_days_says_on_weekday():
    target = datetime.now(timezone.utc) + timedelta(days=3)
    out = format_cap_hit_reply(target.isoformat())
    weekday = target.strftime("%A")
    assert f"on {weekday}" in out


def test_format_cap_hit_reply_eight_days_says_in_n_days():
    out = format_cap_hit_reply(_iso_in_days(8.0))
    assert "in 8 days" in out


def test_format_cap_hit_reply_past_says_soon():
    out = format_cap_hit_reply(_iso_in_days(-2.0))
    assert "soon" in out


@pytest.mark.parametrize("bad", ["", None])
def test_format_cap_hit_reply_missing_says_next_week(bad):
    out = format_cap_hit_reply(bad)
    assert "next week" in out


def test_format_cap_hit_reply_malformed_says_soon():
    out = format_cap_hit_reply("not-a-date")
    assert "soon" in out


def test_format_cap_hit_reply_date_only_string():
    """Date-only ISO (e.g. '2026-05-11') is treated as midnight UTC."""
    out = format_cap_hit_reply(_iso_in_days(1.0, date_only=True))
    # Could be "tomorrow" or "in <N> days" depending on current UTC time-of-day
    # relative to midnight; either way the reply must render and reference the
    # budget.
    assert "free Homer budget" in out
    assert ("tomorrow" in out) or ("on " in out) or ("soon" in out)


def test_format_cap_hit_reply_handles_trailing_z():
    """ISO with trailing 'Z' (Zulu) parses cleanly."""
    when = (datetime.now(timezone.utc) + timedelta(days=1)).replace(microsecond=0)
    iso_z = when.isoformat().replace("+00:00", "Z")
    out = format_cap_hit_reply(iso_z)
    assert "tomorrow" in out


# ── End-to-end: portal payload → cap-hit reply ─────────────────────────────


def test_e2e_cap_hit_reply_uses_reset_at_from_portal(default_env):
    payload = {
        "ok": False,
        "used": 1_500_000,
        "budget": 1_000_000,
        "reset_at": _iso_in_days(1.0),
    }
    resp = _mk_resp(200, payload)
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn({})
    assert out is not None
    assert "tomorrow" in out
    assert "free Homer budget" in out


def test_e2e_cap_hit_reply_missing_reset_at_says_next_week(default_env):
    payload = {"ok": False, "used": 2_000_000, "budget": 1_000_000}
    resp = _mk_resp(200, payload)
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn({})
    assert out is not None
    assert "next week" in out


def test_e2e_cap_hit_reply_null_reset_at_says_next_week(default_env):
    payload = {
        "ok": False,
        "used": 2_000_000,
        "budget": 1_000_000,
        "reset_at": None,
    }
    resp = _mk_resp(200, payload)
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn({})
    assert out is not None
    assert "next week" in out


def test_e2e_cap_hit_reply_non_string_reset_at_falls_back(default_env):
    """A non-string reset_at (schema drift) must not crash; falls back to next week."""
    payload = {
        "ok": False,
        "used": 2_000_000,
        "budget": 1_000_000,
        "reset_at": 12345,  # numeric, not ISO string
    }
    resp = _mk_resp(200, payload)
    with patch.object(quota_gate.httpx, "get", return_value=resp):
        out = check_token_budget_before_turn({})
    assert out is not None
    assert "next week" in out
