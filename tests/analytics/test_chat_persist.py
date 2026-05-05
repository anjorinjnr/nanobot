"""Tests for ChatPersistHook — Supabase calls are stubbed via httpx mocks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.analytics import chat_persist as chat_persist_module
from nanobot.analytics.chat_persist import (
    ChatPersistHook,
    _build_storage_path,
    _kind_from_mime,
    _safe_filename,
    get_chat_persist_hook,
)


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

    @pytest.mark.asyncio
    async def test_schedule_background_offloads_insert(self):
        # When schedule_background is provided, the insert isn't awaited
        # inline — the user-visible OutboundMessage return doesn't wait.
        hook, client = _make_hook(contributor_rows=[{"id": "c-1"}])
        ctx = {"contributor_id": "c-1", "channel": "whatsapp"}

        scheduled: list = []
        def _schedule(coro):
            scheduled.append(coro)

        await hook.on_response_sent(ctx, response_content="hi", schedule_background=_schedule)
        # Insert was scheduled, not awaited.
        assert client.post.await_count == 0
        assert len(scheduled) == 1
        # Drain the scheduled coroutine so the test doesn't leak unawaited tasks.
        await scheduled[0]
        assert client.post.await_count == 1


# ── helpers / pure functions ─────────────────────────────────────────────


class TestPureHelpers:
    def test_kind_from_mime_image(self):
        assert _kind_from_mime("image/jpeg") == "image"
        assert _kind_from_mime("image/png") == "image"

    def test_kind_from_mime_audio(self):
        assert _kind_from_mime("audio/ogg") == "audio"

    def test_kind_from_mime_video(self):
        assert _kind_from_mime("video/mp4") == "video"

    def test_kind_from_mime_unknown_returns_none(self):
        assert _kind_from_mime(None) is None
        assert _kind_from_mime("application/pdf") is None
        assert _kind_from_mime("") is None

    def test_safe_filename_strips_unsafe(self):
        assert _safe_filename("hello world!@#.jpg") == "hello_world___.jpg"

    def test_safe_filename_keeps_dotdash(self):
        assert _safe_filename("photo-2024.jpg") == "photo-2024.jpg"

    def test_safe_filename_truncates(self):
        out = _safe_filename("a" * 500 + ".jpg")
        assert len(out) <= 200

    def test_safe_filename_empty_input_falls_back(self):
        assert _safe_filename("") == "file"

    def test_safe_filename_replaces_unsafe_with_underscore(self):
        # Unsafe chars become "_"; underscores are themselves portal-safe so
        # there's no further fallback. (`_` is in [A-Za-z0-9._\-].)
        assert _safe_filename("###") == "___"

    def test_build_storage_path_matches_portal_regex(self):
        import re
        # Mirror of backend/routers/history.py:_STORAGE_PATH_RE
        pattern = re.compile(r"^\d{4}/\d{2}/[a-f0-9]{12}-[A-Za-z0-9._\-]{1,200}$")
        path = _build_storage_path("photo.jpg")
        assert pattern.match(path), f"{path!r} does not match portal regex"


# ── on_message_received with media ───────────────────────────────────────


class TestMediaUpload:
    @pytest.fixture
    def jpg_file(self, tmp_path):
        p = tmp_path / "photo.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0fake jpeg bytes")
        return p

    @pytest.fixture
    def ogg_file(self, tmp_path):
        p = tmp_path / "voice.ogg"
        p.write_bytes(b"OggS" + b"\x00" * 100)
        return p

    @pytest.fixture
    def patched_httpx(self, monkeypatch):
        """Patch httpx.AsyncClient inside chat_persist for storage calls.

        Returns a MagicMock with a `.post` AsyncMock you can configure.
        Successful by default.
        """
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=False)

        def _resp(status: int, body: dict | None = None, text: str = ""):
            r = MagicMock()
            r.status_code = status
            r.text = text
            r.json = MagicMock(return_value=body or {})
            return r

        cm.post = AsyncMock(return_value=_resp(201))
        cm._resp = _resp  # expose so tests can override

        monkeypatch.setattr(chat_persist_module.httpx, "AsyncClient", lambda **kw: cm)
        return cm

    @pytest.mark.asyncio
    async def test_image_upload_sets_pending_upload(self, jpg_file, patched_httpx):
        hook, rest_client = _make_hook(contributor_rows=[{"id": "c-1"}])
        ctx = await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="check this out",
            media=[str(jpg_file)],
            timestamp=datetime.now(timezone.utc),
        )
        assert ctx == {"contributor_id": "c-1", "channel": "whatsapp"}
        # Storage upload happened.
        assert patched_httpx.post.await_count >= 1
        upload_call = patched_httpx.post.await_args_list[0]
        # base_url=/storage/v1 is on the AsyncClient; the request path is relative.
        assert upload_call.args[0].startswith("/object/history-media-hh-1/")
        assert upload_call.kwargs["headers"]["Content-Type"] == "image/jpeg"
        assert upload_call.kwargs["content"] == jpg_file.read_bytes()
        # Insert body has pending_upload.
        post_body = rest_client.post.await_args.kwargs["json"]
        assert post_body["role"] == "user"
        assert post_body["text"] == "check this out"
        pu = post_body["pending_upload"]
        assert pu["filename"] == "photo.jpg"
        assert pu["mime"] == "image/jpeg"
        assert pu["kind"] == "image"
        assert pu["storage_path"].startswith("history-media-hh-1/")

    @pytest.mark.asyncio
    async def test_audio_upload_kind_audio(self, ogg_file, patched_httpx):
        hook, rest_client = _make_hook(contributor_rows=[{"id": "c-1"}])
        await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="",
            media=[str(ogg_file)],
            timestamp=datetime.now(timezone.utc),
        )
        post_body = rest_client.post.await_args.kwargs["json"]
        assert post_body["pending_upload"]["kind"] == "audio"
        assert post_body["pending_upload"]["mime"] == "audio/ogg"

    @pytest.mark.asyncio
    async def test_unsupported_mime_skips_upload_keeps_text(
        self, tmp_path, patched_httpx,
    ):
        # PDF (kind not in image/audio/video) → no upload, no pending_upload,
        # but the user row still records the text.
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        hook, rest_client = _make_hook(contributor_rows=[{"id": "c-1"}])
        await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="see attached",
            media=[str(pdf)],
            timestamp=datetime.now(timezone.utc),
        )
        # No storage upload.
        assert patched_httpx.post.await_count == 0
        # Insert went through with text, no pending_upload.
        post_body = rest_client.post.await_args.kwargs["json"]
        assert post_body["text"] == "see attached"
        assert "pending_upload" not in post_body

    @pytest.mark.asyncio
    async def test_missing_file_skips_upload(self, tmp_path, patched_httpx):
        ghost = str(tmp_path / "nope.jpg")
        hook, rest_client = _make_hook(contributor_rows=[{"id": "c-1"}])
        await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="hi",
            media=[ghost],
            timestamp=datetime.now(timezone.utc),
        )
        assert patched_httpx.post.await_count == 0
        post_body = rest_client.post.await_args.kwargs["json"]
        assert post_body["text"] == "hi"
        assert "pending_upload" not in post_body

    @pytest.mark.asyncio
    async def test_oversize_file_skipped(self, tmp_path, patched_httpx, monkeypatch):
        # Lower the cap so the test doesn't have to allocate 50MB.
        monkeypatch.setattr(chat_persist_module, "_MAX_UPLOAD_BYTES", 100)
        big = tmp_path / "big.jpg"
        big.write_bytes(b"x" * 500)
        hook, rest_client = _make_hook(contributor_rows=[{"id": "c-1"}])
        await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="huge",
            media=[str(big)],
            timestamp=datetime.now(timezone.utc),
        )
        assert patched_httpx.post.await_count == 0
        assert "pending_upload" not in rest_client.post.await_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_upload_failure_keeps_text(self, jpg_file, patched_httpx):
        # First POST returns 500 — bucket retry path is only taken on 400/404,
        # so a 500 means upload is unrecoverable. Row still persists w/o pending_upload.
        patched_httpx.post = AsyncMock(return_value=patched_httpx._resp(500, text="oops"))
        hook, rest_client = _make_hook(contributor_rows=[{"id": "c-1"}])
        await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="hi",
            media=[str(jpg_file)],
            timestamp=datetime.now(timezone.utc),
        )
        post_body = rest_client.post.await_args.kwargs["json"]
        assert post_body["text"] == "hi"
        assert "pending_upload" not in post_body

    @pytest.mark.asyncio
    async def test_bucket_missing_provisions_and_retries(self, jpg_file, patched_httpx):
        # POST sequence: 404 (bucket missing) → 201 (bucket create) → 201 (retry upload).
        sequence = [
            patched_httpx._resp(404, text="Bucket not found"),  # initial upload
            patched_httpx._resp(201),                            # bucket create
            patched_httpx._resp(201),                            # retry upload
        ]
        patched_httpx.post = AsyncMock(side_effect=sequence)
        hook, rest_client = _make_hook(contributor_rows=[{"id": "c-1"}])
        await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="hi",
            media=[str(jpg_file)],
            timestamp=datetime.now(timezone.utc),
        )
        # All three storage calls happened.
        assert patched_httpx.post.await_count == 3
        # Bucket-create call hit /bucket on the storage client (base_url=/storage/v1).
        bucket_call = patched_httpx.post.await_args_list[1]
        assert bucket_call.args[0] == "/bucket"
        assert bucket_call.kwargs["json"] == {
            "id": "history-media-hh-1",
            "name": "history-media-hh-1",
            "public": False,
        }
        # Final row carries pending_upload (upload succeeded after retry).
        post_body = rest_client.post.await_args.kwargs["json"]
        assert post_body["pending_upload"]["kind"] == "image"

    @pytest.mark.asyncio
    async def test_bucket_create_409_treated_as_success(self, jpg_file, patched_httpx):
        # 400 with statusCode=409 body == bucket already exists.
        sequence = [
            patched_httpx._resp(404),                                                       # upload
            patched_httpx._resp(400, body={"statusCode": "409", "error": "Duplicate"}),     # ensure
            patched_httpx._resp(201),                                                       # retry
        ]
        patched_httpx.post = AsyncMock(side_effect=sequence)
        hook, rest_client = _make_hook(contributor_rows=[{"id": "c-1"}])
        await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="hi",
            media=[str(jpg_file)],
            timestamp=datetime.now(timezone.utc),
        )
        # Retry succeeded → pending_upload present.
        assert "pending_upload" in rest_client.post.await_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_multi_attachment_keeps_first_only(
        self, tmp_path, jpg_file, patched_httpx,
    ):
        second = tmp_path / "extra.jpg"
        second.write_bytes(b"\xff\xd8")
        hook, rest_client = _make_hook(contributor_rows=[{"id": "c-1"}])
        await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="two photos",
            media=[str(jpg_file), str(second)],
            timestamp=datetime.now(timezone.utc),
        )
        # Exactly one storage upload — the first.
        assert patched_httpx.post.await_count == 1
        pu = rest_client.post.await_args.kwargs["json"]["pending_upload"]
        assert pu["filename"] == "photo.jpg"


# ── curator bootstrap (HOMER_ADMIN_PHONE) ────────────────────────────────


def _bootstrap_hook(
    *,
    initial_lookup_rows: list[dict] | None = None,
    curator_lookup_rows: list[dict] | None = None,
    admin_lookup_rows: list[dict] | None = None,
    insert_rows: list[dict] | None = None,
) -> tuple[ChatPersistHook, MagicMock]:
    """Build a hook whose mocked httpx client returns scripted GET/POST/PATCH
    responses suitable for the bootstrap flow.

    The bootstrap path can issue up to 3 GETs:
      1. initial contributor lookup by phone (always happens first)
      2. curator lookup by role='curator'
      3. household_members admin lookup
    Plus optionally a PATCH (existing curator phone=null) or POST (insert
    new curator).

    The fixture sequences GETs in that order and returns the last queued
    POST/PATCH response on every write.
    """
    hook = ChatPersistHook()
    hook._initialized = True
    hook._enabled = True
    hook._supabase_url = "https://example.supabase.co"
    hook._service_key = "tok_test"
    hook._household_id = "hh-1"

    def _resp(rows: list[dict] | None) -> MagicMock:
        r = MagicMock()
        r.status_code = 200
        r.raise_for_status = MagicMock()
        r.json.return_value = rows or []
        return r

    get_sequence = [
        _resp(initial_lookup_rows),
        _resp(curator_lookup_rows),
        _resp(admin_lookup_rows),
    ]
    write_resp = MagicMock()
    write_resp.status_code = 201
    write_resp.raise_for_status = MagicMock()
    write_resp.json.return_value = insert_rows or []

    client = MagicMock()
    client.get = AsyncMock(side_effect=get_sequence)
    client.post = AsyncMock(return_value=write_resp)
    client.patch = AsyncMock(return_value=write_resp)
    hook._client = client
    return hook, client


class TestCuratorBootstrap:
    @pytest.mark.asyncio
    async def test_bootstrap_skipped_when_admin_phone_unset(self, monkeypatch):
        monkeypatch.delenv("HOMER_ADMIN_PHONE", raising=False)
        hook, client = _bootstrap_hook(initial_lookup_rows=[])
        cid = await hook._resolve_contributor("whatsapp", "14125551234")
        assert cid is None
        # Only the initial contributor lookup happened — no bootstrap calls.
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_bootstrap_skipped_when_sender_doesnt_match_admin(self, monkeypatch):
        monkeypatch.setenv("HOMER_ADMIN_PHONE", "19998887777")
        hook, client = _bootstrap_hook(initial_lookup_rows=[])
        cid = await hook._resolve_contributor("whatsapp", "14125551234")
        assert cid is None
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_bootstrap_existing_curator_phone_match_returns_id(
        self, monkeypatch,
    ):
        # Curator row already has matching phone (race / no-op case) — just
        # return the existing id without a write.
        monkeypatch.setenv("HOMER_ADMIN_PHONE", "14125551234")
        hook, client = _bootstrap_hook(
            initial_lookup_rows=[],
            curator_lookup_rows=[{"id": "c-curator", "phone": "14125551234"}],
        )
        cid = await hook._resolve_contributor("whatsapp", "14125551234")
        assert cid == "c-curator"
        # No write — phone already matches.
        assert client.post.await_count == 0
        assert client.patch.await_count == 0

    @pytest.mark.asyncio
    async def test_bootstrap_existing_curator_phone_null_patches(self, monkeypatch):
        # The common case: portal-signup created the curator row with
        # phone=null, the curator's first WhatsApp message patches it.
        monkeypatch.setenv("HOMER_ADMIN_PHONE", "14125551234")
        hook, client = _bootstrap_hook(
            initial_lookup_rows=[],
            curator_lookup_rows=[{"id": "c-curator", "phone": None}],
        )
        cid = await hook._resolve_contributor("whatsapp", "14125551234")
        assert cid == "c-curator"
        # PATCH fired with the resolved phone.
        assert client.patch.await_count == 1
        body = client.patch.await_args.kwargs["json"]
        assert body == {"phone": "14125551234"}
        params = client.patch.await_args.kwargs["params"]
        assert params == {"id": "eq.c-curator"}

    @pytest.mark.asyncio
    async def test_bootstrap_existing_curator_different_phone_declines(
        self, monkeypatch,
    ):
        # Defends against silently overwriting a curator's previous phone
        # — could be a co-curator scenario or an admin who changed numbers
        # via portal. Manual reconciliation needed.
        monkeypatch.setenv("HOMER_ADMIN_PHONE", "14125551234")
        hook, client = _bootstrap_hook(
            initial_lookup_rows=[],
            curator_lookup_rows=[{"id": "c-curator", "phone": "19998887777"}],
        )
        cid = await hook._resolve_contributor("whatsapp", "14125551234")
        assert cid is None
        assert client.patch.await_count == 0
        assert client.post.await_count == 0

    @pytest.mark.asyncio
    async def test_bootstrap_no_curator_inserts_from_household_members(
        self, monkeypatch,
    ):
        # No curator row yet (household provisioned but no portal visit) —
        # look up the admin from household_members and INSERT.
        monkeypatch.setenv("HOMER_ADMIN_PHONE", "14125551234")
        hook, client = _bootstrap_hook(
            initial_lookup_rows=[],
            curator_lookup_rows=[],
            admin_lookup_rows=[{"user_id": "u-admin", "name": "Ebby"}],
            insert_rows=[{"id": "c-new"}],
        )
        cid = await hook._resolve_contributor("whatsapp", "14125551234")
        assert cid == "c-new"
        assert client.post.await_count == 1
        body = client.post.await_args.kwargs["json"]
        assert body["household_id"] == "hh-1"
        assert body["role"] == "curator"
        assert body["phone"] == "14125551234"
        assert body["auth_user_id"] == "u-admin"
        assert body["display_name"] == "Ebby"
        assert body["status"] == "active"

    @pytest.mark.asyncio
    async def test_bootstrap_no_curator_no_admin_returns_none(self, monkeypatch):
        # Genuinely empty household — refuse to invent a curator without
        # an underlying household_members admin to anchor the row to.
        monkeypatch.setenv("HOMER_ADMIN_PHONE", "14125551234")
        hook, client = _bootstrap_hook(
            initial_lookup_rows=[],
            curator_lookup_rows=[],
            admin_lookup_rows=[],
        )
        cid = await hook._resolve_contributor("whatsapp", "14125551234")
        assert cid is None
        assert client.post.await_count == 0

    @pytest.mark.asyncio
    async def test_bootstrap_falls_back_to_default_display_name(self, monkeypatch):
        # household_members.name is sometimes empty; insert should still
        # succeed with a sensible default rather than reject the bootstrap.
        monkeypatch.setenv("HOMER_ADMIN_PHONE", "14125551234")
        hook, client = _bootstrap_hook(
            initial_lookup_rows=[],
            curator_lookup_rows=[],
            admin_lookup_rows=[{"user_id": "u-admin", "name": ""}],
            insert_rows=[{"id": "c-new"}],
        )
        cid = await hook._resolve_contributor("whatsapp", "14125551234")
        assert cid == "c-new"
        body = client.post.await_args.kwargs["json"]
        assert body["display_name"] == "Curator"

    @pytest.mark.asyncio
    async def test_end_to_end_curator_first_message_persists(self, monkeypatch):
        # Real-world flow: curator's first WhatsApp message after
        # provisioning. Initial lookup misses, bootstrap creates curator,
        # then on_message_received persists the user row under the new id.
        monkeypatch.setenv("HOMER_ADMIN_PHONE", "14125551234")
        hook, client = _bootstrap_hook(
            initial_lookup_rows=[],
            curator_lookup_rows=[],
            admin_lookup_rows=[{"user_id": "u-admin", "name": "Ebby"}],
            insert_rows=[{"id": "c-new"}],
        )
        ctx = await hook.on_message_received(
            channel="whatsapp",
            sender_id="14125551234",
            content="hello, starting family history",
            media=[],
            timestamp=datetime.now(timezone.utc),
        )
        assert ctx == {"contributor_id": "c-new", "channel": "whatsapp"}
        # Two POSTs total: the curator INSERT + the chat message INSERT.
        assert client.post.await_count == 2
        chat_insert = client.post.await_args.kwargs["json"]
        assert chat_insert["role"] == "user"
        assert chat_insert["text"] == "hello, starting family history"
        assert chat_insert["contributor_id"] == "c-new"
