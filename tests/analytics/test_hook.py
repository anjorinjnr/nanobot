"""Tests for AnalyticsHook — persistence, dedup observability, event shape.

These tests exercise the PostHog analytics hook without actually contacting
PostHog: the posthog client is stubbed with a MagicMock so we can inspect
every capture/identify call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.analytics import identity as identity_module
from nanobot.analytics.hook import AnalyticsHook
from nanobot.analytics.identity import _hash_identity_key


@pytest.fixture(autouse=True)
def _clear_identity_cache():
    """Identity map cache is process-global; reset between tests."""
    identity_module._load_identity_map_cached.cache_clear()
    yield
    identity_module._load_identity_map_cached.cache_clear()


# ── helpers ──────────────────────────────────────────────────────────────


class _AsyncReturn:
    """An awaitable factory: calling returns a coroutine that yields `value`."""
    def __init__(self, value): self.value = value
    async def __call__(self, *a, **kw): return self.value


def _make_hook(state_dir: Path, household_id: str = "hh-1") -> AnalyticsHook:
    """Return an AnalyticsHook whose state persists under `state_dir`,
    with a mocked PostHog client so captures are inspectable.
    """
    hook = AnalyticsHook()
    # Stub the posthog client + skip real init. _ensure_init is called
    # lazily; shortcut it by flipping flags and injecting a mock.
    hook._initialized = True
    hook._client = MagicMock()
    hook._household_id = household_id
    hook._state_path = state_dir / "seen_users.json"
    hook._state_loaded = False  # force _load_state to run
    hook._load_state()
    return hook


def _receive(
    hook: AnalyticsHook,
    *,
    channel: str = "whatsapp",
    sender_id: str = "+15551234",
    content: str = "hi",
) -> dict:
    """Mimic on_message_received → build context."""
    return hook.on_message_received(
        channel=channel,
        sender_id=sender_id,
        content=content,
        media=[],
        timestamp=datetime.now(timezone.utc),
        is_guest=False,
    )


def _captured_events(client: MagicMock, event_name: str) -> list[dict]:
    """Return the props dicts of every `capture(distinct_id, event_name, props)` call."""
    out = []
    for call in client.capture.call_args_list:
        args, kwargs = call.args, call.kwargs
        # capture(distinct_id, event, props) — positional or mixed.
        if len(args) >= 2 and args[1] == event_name:
            out.append(args[2] if len(args) >= 3 else kwargs.get("properties", {}))
    return out


# ── persistence ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_user_onboarded_fires_once_then_persists(tmp_path, monkeypatch):
    """First message fires user_onboarded; a second hook instance loaded
    from the same state does NOT re-fire for the same distinct_id."""
    monkeypatch.setenv("HOMER_ANALYTICS_STATE_DIR", str(tmp_path))

    hook1 = _make_hook(tmp_path)
    ctx = _receive(hook1, channel="whatsapp", sender_id="+15551234")
    await hook1.on_response_sent(
        ctx, response_content="ok", tools_used=set(),
    )

    onboarded = _captured_events(hook1._client, "user_onboarded")
    assert len(onboarded) == 1
    assert onboarded[0]["is_new_household"] is True

    # Second hook instance simulates a process restart — it must see the
    # persisted distinct_id and skip user_onboarded this time.
    hook2 = _make_hook(tmp_path)
    ctx = _receive(hook2, channel="whatsapp", sender_id="+15551234")
    await hook2.on_response_sent(
        ctx, response_content="ok", tools_used=set(),
    )

    assert _captured_events(hook2._client, "user_onboarded") == []
    assert _captured_events(hook2._client, "household_member_added") == []


@pytest.mark.asyncio
async def test_second_distinct_id_fires_user_onboarded_not_member_added(tmp_path, monkeypatch):
    """Two distinct_ids in the same household → two user_onboarded events,
    but NO household_member_added — that event is now owned by homer's
    explicit add-member flow, not inferred from inbound messages."""
    monkeypatch.setenv("HOMER_ANALYTICS_STATE_DIR", str(tmp_path))

    hook1 = _make_hook(tmp_path)
    await hook1.on_response_sent(
        _receive(hook1, channel="whatsapp", sender_id="+15551234"),
        response_content="ok", tools_used=set(),
    )
    await hook1.on_response_sent(
        _receive(hook1, channel="email", sender_id="b@example.com"),
        response_content="ok", tools_used=set(),
    )

    onboarded = _captured_events(hook1._client, "user_onboarded")
    assert len(onboarded) == 2
    assert onboarded[0]["is_new_household"] is True
    assert onboarded[1]["is_new_household"] is False
    assert _captured_events(hook1._client, "household_member_added") == []

    # Restart: persistence still works — no re-fire of user_onboarded.
    hook2 = _make_hook(tmp_path)
    await hook2.on_response_sent(
        _receive(hook2, channel="whatsapp", sender_id="+15551234"),
        response_content="ok", tools_used=set(),
    )
    await hook2.on_response_sent(
        _receive(hook2, channel="email", sender_id="b@example.com"),
        response_content="ok", tools_used=set(),
    )
    assert _captured_events(hook2._client, "user_onboarded") == []
    assert _captured_events(hook2._client, "household_member_added") == []


@pytest.mark.asyncio
async def test_state_file_shape(tmp_path, monkeypatch):
    """Persisted state is a JSON dict with version, seen_users, first_user_ts."""
    monkeypatch.setenv("HOMER_ANALYTICS_STATE_DIR", str(tmp_path))

    hook = _make_hook(tmp_path)
    await hook.on_response_sent(
        _receive(hook, channel="whatsapp", sender_id="+15551234"),
        response_content="ok", tools_used=set(),
    )

    state_file = tmp_path / "seen_users.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["version"] == 1
    assert isinstance(data["seen_users"], list)
    assert len(data["seen_users"]) == 1
    assert isinstance(data["first_user_ts"], (int, float))


@pytest.mark.asyncio
async def test_corrupt_state_file_recovers(tmp_path, monkeypatch):
    """A garbled state file must not crash — hook starts with empty seen_users."""
    monkeypatch.setenv("HOMER_ANALYTICS_STATE_DIR", str(tmp_path))
    (tmp_path / "seen_users.json").write_text("not valid json {")

    hook = _make_hook(tmp_path)
    assert hook._seen_users == set()

    # First message should still fire user_onboarded normally.
    await hook.on_response_sent(
        _receive(hook, channel="whatsapp", sender_id="+15551234"),
        response_content="ok", tools_used=set(),
    )
    assert len(_captured_events(hook._client, "user_onboarded")) == 1


def test_no_state_path_when_env_missing_and_no_config(monkeypatch):
    """Without HOMER_ANALYTICS_STATE_DIR and without a loaded nanobot config,
    the hook runs in-memory only (state_path is None)."""
    monkeypatch.delenv("HOMER_ANALYTICS_STATE_DIR", raising=False)

    hook = AnalyticsHook()
    with patch(
        "nanobot.config.paths.get_runtime_subdir",
        side_effect=RuntimeError("no config loaded"),
    ):
        hook._load_state()

    assert hook._state_path is None
    assert hook._seen_users == set()


# ── main/guest isolation ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_per_config_state_isolation(tmp_path, monkeypatch):
    """Two processes with different `get_config_path()` values must not
    clobber each other's state. Simulates main + guest nanobot sharing a
    parent data dir but each calling set_config_path() separately."""
    from nanobot.config import loader as config_loader

    data_dir = tmp_path / "nanobot_data"
    data_dir.mkdir()
    main_config = data_dir / "config.json"
    guest_config = data_dir / "guest_config.json"
    main_config.write_text("{}")
    guest_config.write_text("{}")

    # Ensure HOMER_ANALYTICS_STATE_DIR does NOT override — we want to
    # exercise the config-stem-based path resolution.
    monkeypatch.delenv("HOMER_ANALYTICS_STATE_DIR", raising=False)

    def _make_process_hook(config_path: Path) -> AnalyticsHook:
        config_loader.set_config_path(config_path)
        hook = AnalyticsHook()
        hook._initialized = True
        hook._client = MagicMock()
        hook._household_id = "hh-1"
        hook._load_state()
        return hook

    try:
        # Main records its user.
        main_hook = _make_process_hook(main_config)
        await main_hook.on_response_sent(
            _receive(main_hook, channel="whatsapp", sender_id="+15551234"),
            response_content="ok", tools_used=set(),
        )
        # Guest records its user in a *separate* process's view.
        guest_hook = _make_process_hook(guest_config)
        await guest_hook.on_response_sent(
            _receive(guest_hook, channel="telegram", sender_id="guest-telegram-id"),
            response_content="ok", tools_used=set(),
        )

        # Each should have written to a different state file.
        main_state = main_hook._state_path
        guest_state = guest_hook._state_path
        assert main_state != guest_state
        assert main_state.exists() and guest_state.exists()

        # Reload main's state — must still contain only main's user,
        # untouched by guest's write.
        reloaded_main = _make_process_hook(main_config)
        assert len(reloaded_main._seen_users) == 1
    finally:
        # Reset module-global config path so later tests aren't polluted.
        config_loader.set_config_path(None)  # type: ignore[arg-type]


# ── canonical identity migration ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_household_member_added_for_channel_switch(tmp_path, monkeypatch):
    """Ebby messaging via whatsapp, then telegram, then email must not
    trigger household_member_added — the identity map collapses them."""
    monkeypatch.setenv("HOMER_ANALYTICS_STATE_DIR", str(tmp_path))
    map_path = tmp_path / "identity_map.json"
    map_path.write_text(json.dumps({
        "whatsapp:14127733949": "person:ebby",
        "telegram:1973156656": "person:ebby",
        "email:ebby@joybuild.ai": "person:ebby",
    }))
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(map_path))

    hook = _make_hook(tmp_path)
    await hook.on_response_sent(
        _receive(hook, channel="whatsapp", sender_id="14127733949"),
        response_content="ok", tools_used=set(),
    )
    await hook.on_response_sent(
        _receive(hook, channel="telegram", sender_id="1973156656"),
        response_content="ok", tools_used=set(),
    )
    await hook.on_response_sent(
        _receive(hook, channel="email", sender_id="ebby@joybuild.ai"),
        response_content="ok", tools_used=set(),
    )

    # Exactly one onboarding — not three.
    assert len(_captured_events(hook._client, "user_onboarded")) == 1
    # And zero household_member_added — same human, not new members.
    assert _captured_events(hook._client, "household_member_added") == []


@pytest.mark.asyncio
async def test_second_human_fires_user_onboarded_not_member_added(tmp_path, monkeypatch):
    """A second distinct canonical person fires a second user_onboarded
    but no household_member_added — that event is now fired explicitly
    from homer's add-member flow, not inferred from inbound traffic."""
    monkeypatch.setenv("HOMER_ANALYTICS_STATE_DIR", str(tmp_path))
    map_path = tmp_path / "identity_map.json"
    map_path.write_text(json.dumps({
        "whatsapp:111": "person:ebby",
        "whatsapp:222": "person:seun",
        "telegram:333": "person:seun",
    }))
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(map_path))

    hook = _make_hook(tmp_path)
    await hook.on_response_sent(
        _receive(hook, channel="whatsapp", sender_id="111"),
        response_content="ok", tools_used=set(),
    )
    await hook.on_response_sent(
        _receive(hook, channel="whatsapp", sender_id="222"),
        response_content="ok", tools_used=set(),
    )
    await hook.on_response_sent(
        _receive(hook, channel="telegram", sender_id="333"),
        response_content="ok", tools_used=set(),
    )

    assert len(_captured_events(hook._client, "user_onboarded")) == 2
    assert _captured_events(hook._client, "household_member_added") == []


@pytest.mark.asyncio
async def test_seen_users_migrated_on_identity_map_rollout(tmp_path, monkeypatch):
    """Pre-existing seen_users entries (channel-scoped hashes, from a
    deploy before the identity map existed) must be migrated to the
    canonical hash when the map first loads — otherwise a deploy re-fires
    user_onboarded for every known user."""
    monkeypatch.setenv("HOMER_ANALYTICS_STATE_DIR", str(tmp_path))

    # Simulate a pre-map deployment: seen_users contains channel-scoped
    # hashes produced by the *old* get_distinct_id logic (which hashed
    # "channel:identifier").
    pre_map_hash_wa = _hash_identity_key("whatsapp:14127733949")
    pre_map_hash_tg = _hash_identity_key("telegram:1973156656")
    state_path = tmp_path / "seen_users.json"
    state_path.write_text(json.dumps({
        "version": 1,
        "seen_users": [pre_map_hash_wa, pre_map_hash_tg],
        "first_user_ts": 1700000000.0,
    }))

    # Now turn on the identity map and boot the hook.
    map_path = tmp_path / "identity_map.json"
    map_path.write_text(json.dumps({
        "whatsapp:14127733949": "person:ebby",
        "telegram:1973156656": "person:ebby",
    }))
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(map_path))

    hook = _make_hook(tmp_path)

    # Ebby's canonical hash should now be in seen_users.
    assert _hash_identity_key("person:ebby") in hook._seen_users

    # And the next message from any channel does NOT fire user_onboarded.
    await hook.on_response_sent(
        _receive(hook, channel="whatsapp", sender_id="14127733949"),
        response_content="ok", tools_used=set(),
    )
    assert _captured_events(hook._client, "user_onboarded") == []


@pytest.mark.asyncio
async def test_no_migration_when_identity_map_missing(tmp_path, monkeypatch):
    """Without HOMER_IDENTITY_MAP, _migrate_seen_users_to_canonical is a no-op."""
    monkeypatch.setenv("HOMER_ANALYTICS_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("HOMER_IDENTITY_MAP", raising=False)

    pre_hash = _hash_identity_key("whatsapp:+15551234")
    (tmp_path / "seen_users.json").write_text(json.dumps({
        "version": 1, "seen_users": [pre_hash], "first_user_ts": 1700000000.0,
    }))

    hook = _make_hook(tmp_path)
    assert hook._seen_users == {pre_hash}


# ── turn_id observability ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_id_shared_across_all_events_in_one_turn(tmp_path, monkeypatch):
    """Every event fired from a single _process_message call must carry
    the same turn_id — that's how we'll spot duplicate firings in PostHog."""
    monkeypatch.setenv("HOMER_ANALYTICS_STATE_DIR", str(tmp_path))

    # schedule_background runs the coro inline so message_sent fires
    # before we inspect captures.
    import asyncio

    coros = []

    def run_bg(coro):
        coros.append(asyncio.ensure_future(coro))

    hook = _make_hook(tmp_path)
    # Patch the classifier so we don't call any LLM.
    with patch(
        "nanobot.analytics.classify.classify_message_async",
        new=_AsyncReturn("calendar"),
    ):
        ctx = _receive(hook, channel="whatsapp", sender_id="+15551234", content="great job!")
        await hook.on_response_sent(
            ctx,
            response_content="ok",
            tools_used={"weather"},
            schedule_background=run_bg,
        )
        # Drain the message_sent background coro.
        await asyncio.gather(*coros)

    # Collect turn_ids from every event type that fired.
    turn_ids: set[str] = set()
    for event in ("agent_responded", "user_onboarded", "message_sent"):
        for props in _captured_events(hook._client, event):
            assert "turn_id" in props, f"{event} missing turn_id: {props}"
            turn_ids.add(props["turn_id"])

    assert len(turn_ids) == 1, f"events span multiple turn_ids: {turn_ids}"
    # And it must match the ctx turn_id.
    assert turn_ids.pop() == ctx["turn_id"]


@pytest.mark.asyncio
async def test_turn_id_differs_across_turns(tmp_path, monkeypatch):
    """Two separate turns → two different turn_ids."""
    monkeypatch.setenv("HOMER_ANALYTICS_STATE_DIR", str(tmp_path))

    hook = _make_hook(tmp_path)
    ctx1 = _receive(hook, channel="whatsapp", sender_id="+15551234")
    await hook.on_response_sent(ctx1, response_content="r1", tools_used=set())
    ctx2 = _receive(hook, channel="whatsapp", sender_id="+15551234")
    await hook.on_response_sent(ctx2, response_content="r2", tools_used=set())

    responded = _captured_events(hook._client, "agent_responded")
    assert len(responded) == 2
    assert responded[0]["turn_id"] != responded[1]["turn_id"]


# ── Issue #49: AnalyticsHook.capture public helper ────────────────────────


def test_capture_helper_emits_with_explicit_distinct_id(tmp_path):
    hook = _make_hook(tmp_path, household_id="hh-cap")
    hook.capture("custom_event", {"k": "v"}, distinct_id="user-7")
    hook._client.capture.assert_called_once_with("user-7", "custom_event", {"k": "v"})


def test_capture_helper_defaults_to_household_id(tmp_path):
    hook = _make_hook(tmp_path, household_id="hh-default")
    hook.capture("custom_event", {"k": "v"})
    hook._client.capture.assert_called_once_with("hh-default", "custom_event", {"k": "v"})


def test_capture_helper_defaults_to_system_when_no_household(tmp_path):
    hook = _make_hook(tmp_path, household_id="")
    hook.capture("custom_event", {"k": "v"})
    hook._client.capture.assert_called_once_with("system", "custom_event", {"k": "v"})


def test_capture_helper_no_op_when_uninitialized():
    """A hook with no API key (real _ensure_init returns False) must silently no-op."""
    hook = AnalyticsHook()
    # _initialized=False + no client; _ensure_init runs the real init path,
    # which returns False without POSTHOG_API_KEY → capture is a no-op.
    hook.capture("custom_event", {"k": "v"})
    # No client was assigned, so nothing to assert beyond "no crash".
    assert hook._client is None
