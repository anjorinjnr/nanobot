"""Tests for ChatPersistHook — Supabase calls are stubbed via httpx mocks."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.analytics import chat_persist as chat_persist_module
from nanobot.analytics.chat_persist import ChatPersistHook, get_chat_persist_hook


# ── helpers ──────────────────────────────────────────────────────────────


def _make_hook(
    *,
    contributor_rows: list[dict] | None = None,
    insert_status: int = 201,
    lookup_status: int = 200,
    raise_on_get: Exception | None = None,
    raise_on_post: Exception | None = None,
) -> tuple[ChatPersistHook, MagicMock]:
    """Build a ChatPersistHook with init bypassed and a mocked httpx client."""
    hook = ChatPersistHook()
    hook._initialized = True
    hook._enabled = True
    hook._supabase_url = "https://example.supabase.co"
    hook._service_key = "tok_test"
    hook._household_id = "hh-1"

    client = MagicMock()

    def _get_response():
        r = MagicMock()
        r.status_code = lookup_status
        r.raise_for_status = MagicMock()
        if lookup_status >= 400:
            r.raise_for_status.side_effect = Exception(f"HTTP {lookup_status}")
        r.json.return_value = contributor_rows or []
        return r

    def _post_response():
        r = MagicMock()
        r.status_code = insert_status
        r.raise_for_status = MagicMock()
        if insert_status >= 400:
            r.raise_for_status.side_effect = Exception(f"HTTP {insert_status}")
        return r

    if raise_on_get is not None:
        client.get = AsyncMock(side_effect=raise_on_get)
    else:
        client.get = AsyncMock(return_value=_get_response())
    if raise_on_post is not None:
        client.post = AsyncMock(side_effect=raise_on_post)
    else:
        client.post = AsyncMock(return_value=_post_response())

    hook._client = client
    return hook, client


@pytest.fixture(autouse=True)
def _reset_singleton():
    """The module-level singleton must not leak between tests."""
    chat_persist_module._HOOK = None
    yield
    chat_persist_module._HOOK = None


# ── init / env gating ────────────────────────────────────────────────────


class TestInit:
    def test_disabled_when_flag_unset(self, monkeypatch):
        monkeypatch.delenv("HOMER_CHAT_PERSIST_ENABLED", raising=False)
        h = ChatPersistHook()
        assert h._ensure_init() is False

    def test_disabled_when_supabase_missing(self, monkeypatch):
        monkeypatch.setenv("HOMER_CHAT_PERSIST_ENABLED", "1")
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        h = ChatPersistHook()
        assert h._ensure_init() is False

    def test_disabled_when_household_missing(self, monkeypatch):
        monkeypatch.setenv("HOMER_CHAT_PERSIST_ENABLED", "true")
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "k")
        monkeypatch.delenv("HOMER_HOUSEHOLD_ID", raising=False)
        h = ChatPersistHook()
        assert h._ensure_init() is False

    def test_enabled_when_all_set(self, monkeypatch):
        monkeypatch.setenv("HOMER_CHAT_PERSIST_ENABLED", "yes")
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co/")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "k")
        monkeypatch.setenv("HOMER_HOUSEHOLD_ID", "hh-x")
        h = ChatPersistHook()
        assert h._ensure_init() is True
        # Trailing slash on URL is stripped.
        assert h._supabase_url == "https://example.supabase.co"
        # Memoized.
        assert h._ensure_init() is True


# ── singleton ────────────────────────────────────────────────────────────


class TestSingleton:
    def test_returns_same_instance(self):
        a = get_chat_persist_hook()
        b = get_chat_persist_hook()
        assert a is b


# ── normalization ────────────────────────────────────────────────────────


class TestNormalizeSender:
    def test_whatsapp_strips_non_digits(self):
        assert ChatPersistHook._normalize_sender("whatsapp", "+1 (412) 555-1234") == "14125551234"

    def test_whatsapp_already_digits(self):
        assert ChatPersistHook._normalize_sender("whatsapp", "14125551234") == "14125551234"

    def test_email_lowercased_and_stripped(self):
        assert ChatPersistHook._normalize_sender("email", "  Mom@EXAMPLE.com ") == "mom@example.com"

    def test_unknown_channel_pass_through(self):
        assert ChatPersistHook._normalize_sender("telegram", "  12345 ") == "12345"

    def test_none_safe(self):
        assert ChatPersistHook._normalize_sender("whatsapp", None) == ""


# ── on_message_received: identity resolution ─────────────────────────────


class TestOnMessageReceived:
    @pytest.mark.asyncio
    async def test_whatsapp_happy_path_inserts_user_row(self):
        hook, client = _make_hook(contributor_rows=[{"id": "c-1"}])
        ctx = await hook.on_message_received(
            channel="whatsapp",
            sender_id="+1 412 555-1234",
            content="hi",
            media=[],
            timestamp=datetime.now(timezone.utc),
        )
        assert ctx == {"contributor_id": "c-1", "channel": "whatsapp"}
        # One GET (lookup) + one POST (insert).
        assert client.get.await_count == 1
        get_kwargs = client.get.await_args.kwargs
        assert get_kwargs["params"]["phone"] == "eq.14125551234"
        assert get_kwargs["params"]["household_id"] == "eq.hh-1"

        assert client.post.await_count == 1
        post_body = client.post.await_args.kwargs["json"]
        assert post_body["role"] == "user"
        assert post_body["text"] == "hi"
        assert post_body["contributor_id"] == "c-1"
        assert post_body["household_id"] == "hh-1"

    @pytest.mark.asyncio
    async def test_email_uses_email_column(self):
        hook, client = _make_hook(contributor_rows=[{"id": "c-2"}])
        await hook.on_message_received(
            channel="email",
            sender_id="MOM@example.COM",
            content="hello",
            media=[],
            timestamp=datetime.now(timezone.utc),
        )
        get_kwargs = client.get.await_args.kwargs
        assert "email" in get_kwargs["params"]
        assert get_kwargs["params"]["email"] == "eq.mom@example.com"

    @pytest.mark.asyncio
    async def test_unknown_sender_drops_returns_none(self):
        hook, client = _make_hook(contributor_rows=[])
        ctx = await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125550000",
            content="hi",
            media=[],
            timestamp=datetime.now(timezone.utc),
        )
        assert ctx is None
        # Lookup happened, but no insert.
        assert client.get.await_count == 1
        assert client.post.await_count == 0

    @pytest.mark.asyncio
    async def test_unsupported_channel_drops_without_lookup(self):
        hook, client = _make_hook(contributor_rows=[{"id": "c-1"}])
        ctx = await hook.on_message_received(
            channel="telegram",
            sender_id="987654321",
            content="hi",
            media=[],
            timestamp=datetime.now(timezone.utc),
        )
        assert ctx is None
        # No HTTP at all — channel rejected up front.
        assert client.get.await_count == 0
        assert client.post.await_count == 0

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, monkeypatch):
        monkeypatch.delenv("HOMER_CHAT_PERSIST_ENABLED", raising=False)
        hook = ChatPersistHook()
        ctx = await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="hi",
            media=[],
            timestamp=datetime.now(timezone.utc),
        )
        assert ctx is None

    @pytest.mark.asyncio
    async def test_lookup_failure_returns_none_and_swallows(self):
        hook, client = _make_hook(raise_on_get=Exception("boom"))
        ctx = await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="hi",
            media=[],
            timestamp=datetime.now(timezone.utc),
        )
        assert ctx is None  # transient failures don't proceed

    @pytest.mark.asyncio
    async def test_insert_failure_still_returns_ctx(self):
        # User-row insert fails, but contributor was resolved — we still want
        # the assistant turn persisted, so ctx must come back.
        hook, client = _make_hook(
            contributor_rows=[{"id": "c-1"}], raise_on_post=Exception("boom"),
        )
        ctx = await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="hi",
            media=[],
            timestamp=datetime.now(timezone.utc),
        )
        assert ctx == {"contributor_id": "c-1", "channel": "whatsapp"}

    @pytest.mark.asyncio
    async def test_skips_insert_when_text_and_media_empty(self):
        hook, client = _make_hook(contributor_rows=[{"id": "c-1"}])
        ctx = await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="",
            media=[],
            timestamp=datetime.now(timezone.utc),
        )
        # Lookup ran; no insert because there's nothing to record.
        assert ctx == {"contributor_id": "c-1", "channel": "whatsapp"}
        assert client.post.await_count == 0


# ── contributor cache ────────────────────────────────────────────────────


class TestContributorCache:
    @pytest.mark.asyncio
    async def test_repeat_lookup_hits_cache(self):
        hook, client = _make_hook(contributor_rows=[{"id": "c-1"}])
        kwargs = dict(sender_id="14125551234", content="hi", media=[],
                      timestamp=datetime.now(timezone.utc))
        await hook.on_message_received(channel="whatsapp", **kwargs)
        await hook.on_message_received(channel="whatsapp", **kwargs)
        # Two messages — but only one lookup, two inserts.
        assert client.get.await_count == 1
        assert client.post.await_count == 2

    @pytest.mark.asyncio
    async def test_unknown_sender_cached_negative(self):
        hook, client = _make_hook(contributor_rows=[])
        kwargs = dict(sender_id="14125550000", content="hi", media=[],
                      timestamp=datetime.now(timezone.utc))
        await hook.on_message_received(channel="whatsapp", **kwargs)
        await hook.on_message_received(channel="whatsapp", **kwargs)
        # Single lookup, dropped both times.
        assert client.get.await_count == 1
        assert client.post.await_count == 0


# ── on_response_sent ─────────────────────────────────────────────────────


class TestOnResponseSent:
    @pytest.mark.asyncio
    async def test_inserts_assistant_row(self):
        hook, client = _make_hook(contributor_rows=[{"id": "c-1"}])
        ctx = {"contributor_id": "c-1", "channel": "whatsapp"}
        await hook.on_response_sent(ctx, response_content="hello back")
        post_body = client.post.await_args.kwargs["json"]
        assert post_body == {
            "household_id": "hh-1",
            "contributor_id": "c-1",
            "role": "assistant",
            "text": "hello back",
        }

    @pytest.mark.asyncio
    async def test_none_ctx_is_noop(self):
        hook, client = _make_hook()
        await hook.on_response_sent(None, response_content="hi")
        assert client.post.await_count == 0

    @pytest.mark.asyncio
    async def test_empty_response_skipped(self):
        hook, client = _make_hook()
        ctx = {"contributor_id": "c-1", "channel": "whatsapp"}
        await hook.on_response_sent(ctx, response_content="")
        await hook.on_response_sent(ctx, response_content="   ")
        await hook.on_response_sent(ctx, response_content=None)
        assert client.post.await_count == 0

    @pytest.mark.asyncio
    async def test_failure_swallowed(self):
        hook, _ = _make_hook(raise_on_post=Exception("network"))
        ctx = {"contributor_id": "c-1", "channel": "whatsapp"}
        # Must not raise.
        await hook.on_response_sent(ctx, response_content="hi")
