"""Local per-call cost ledger (llm_telemetry._append_cost_ledger).

The tenant container can't read PostHog back, so the weekly report sums this
ledger instead of re-estimating from session logs.
"""

import json

import pytest

from nanobot.analytics import llm_telemetry as t


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "NANOBOT_COST_LEDGER", "HOMER_WORKSPACE",
        "HOMER_HOUSEHOLD_ID", "HOMER_MODEL_TIER", "POSTHOG_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


def _props(**over):
    p = {
        "$ai_model": "deepseek/deepseek-v4-flash",
        "$ai_provider": "openrouter",
        "$ai_input_tokens": 1000,
        "$ai_output_tokens": 200,
        "$ai_cache_read_input_tokens": 800,
        "$ai_total_cost_usd": 0.00012,
        "task_kind": "chat",
        "retry_count": 1,
        "$ai_is_error": False,
    }
    p.update(over)
    return p


# ── path resolution ────────────────────────────────────────────────────────

def test_path_prefers_explicit_override(monkeypatch, tmp_path):
    p = tmp_path / "x.jsonl"
    monkeypatch.setenv("NANOBOT_COST_LEDGER", str(p))
    monkeypatch.setenv("HOMER_WORKSPACE", str(tmp_path / "ws"))
    assert t._cost_ledger_path() == p


def test_path_derives_from_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("HOMER_WORKSPACE", str(tmp_path))
    assert t._cost_ledger_path() == tmp_path / "analytics" / "llm_ledger.jsonl"


def test_path_none_when_unconfigured():
    assert t._cost_ledger_path() is None


# ── append behavior ──────────────────────────────────────────────────────────

def test_append_writes_expected_row(monkeypatch, tmp_path):
    led = tmp_path / "led.jsonl"
    monkeypatch.setenv("NANOBOT_COST_LEDGER", str(led))
    t._append_cost_ledger(_props())
    rows = [json.loads(line) for line in led.read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "deepseek/deepseek-v4-flash"
    assert r["provider"] == "openrouter"
    assert (r["in"], r["out"], r["cache"]) == (1000, 200, 800)
    assert r["cost"] == pytest.approx(0.00012)
    assert r["task"] == "chat"
    assert r["err"] is False
    assert r["ts"].endswith("+00:00")  # UTC ISO
    assert "cost_served" not in r and "model_served" not in r


def test_append_includes_served_fields_when_present(monkeypatch, tmp_path):
    led = tmp_path / "led.jsonl"
    monkeypatch.setenv("NANOBOT_COST_LEDGER", str(led))
    t._append_cost_ledger(_props(**{
        "$ai_cost_usd_served": 0.00009,
        "$ai_model_served": "deepseek/deepseek-v4-flash:exact",
    }))
    r = json.loads(led.read_text().splitlines()[0])
    assert r["cost_served"] == pytest.approx(0.00009)
    assert r["model_served"] == "deepseek/deepseek-v4-flash:exact"


def test_append_appends_not_truncates(monkeypatch, tmp_path):
    led = tmp_path / "led.jsonl"
    monkeypatch.setenv("NANOBOT_COST_LEDGER", str(led))
    t._append_cost_ledger(_props())
    t._append_cost_ledger(_props(task_kind="heartbeat_system"))
    rows = led.read_text().splitlines()
    assert len(rows) == 2
    assert json.loads(rows[1])["task"] == "heartbeat_system"


def test_append_is_noop_and_silent_when_unconfigured():
    # No NANOBOT_COST_LEDGER / HOMER_WORKSPACE → no write, no raise.
    t._append_cost_ledger(_props())


def test_append_never_raises_on_unwritable_path(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")  # mkdir under a file → OSError, must be swallowed
    monkeypatch.setenv("NANOBOT_COST_LEDGER", str(blocker / "sub" / "led.jsonl"))
    t._append_cost_ledger(_props())  # must not raise


def test_emit_event_tees_to_ledger(monkeypatch, tmp_path):
    # _emit_event writes the ledger independent of PostHog availability.
    led = tmp_path / "led.jsonl"
    monkeypatch.setenv("NANOBOT_COST_LEDGER", str(led))
    t._emit_event(_props())
    assert led.exists() and led.read_text().strip()
