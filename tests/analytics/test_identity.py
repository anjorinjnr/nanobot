"""Tests for identity canonicalization via HOMER_IDENTITY_MAP."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.analytics import identity
from nanobot.analytics.identity import (
    _hash_identity_key,
    get_distinct_id,
    migrate_channel_hashes,
)


@pytest.fixture(autouse=True)
def _clear_identity_map_cache():
    """lru_cache on _load_identity_map_cached is keyed by (path, mtime) but
    persists across tests — make sure each test starts clean."""
    identity._load_identity_map_cached.cache_clear()
    yield
    identity._load_identity_map_cached.cache_clear()


def _write_map(tmp_path: Path, mapping: dict[str, str]) -> Path:
    path = tmp_path / "identity_map.json"
    path.write_text(json.dumps(mapping))
    return path


def test_channel_scoped_when_env_missing(monkeypatch):
    monkeypatch.delenv("HOMER_IDENTITY_MAP", raising=False)
    a = get_distinct_id("+15551234", "whatsapp")
    b = get_distinct_id("+15551234", "telegram")
    # Without a map, channel drives the hash — different channels, different ids.
    assert a != b


def test_canonical_collapses_channels(tmp_path, monkeypatch):
    """The same human across three channels must produce one distinct_id."""
    path = _write_map(tmp_path, {
        "whatsapp:14127733949": "person:ebby",
        "telegram:1973156656": "person:ebby",
        "email:ebby@joybuild.ai": "person:ebby",
    })
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(path))

    wa = get_distinct_id("14127733949", "whatsapp")
    tg = get_distinct_id("1973156656", "telegram")
    em = get_distinct_id("ebby@joybuild.ai", "email")
    assert wa == tg == em
    # And equal to the hash of the canonical key directly.
    assert wa == _hash_identity_key("person:ebby")


def test_unmapped_sender_falls_back_to_channel_hash(tmp_path, monkeypatch):
    path = _write_map(tmp_path, {"whatsapp:14127733949": "person:ebby"})
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(path))

    # Ebby on whatsapp → canonical
    ebby_wa = get_distinct_id("14127733949", "whatsapp")
    assert ebby_wa == _hash_identity_key("person:ebby")

    # Unknown number on whatsapp → channel-scoped, NOT the canonical.
    stranger = get_distinct_id("+19999999999", "whatsapp")
    assert stranger != ebby_wa
    assert stranger == _hash_identity_key("whatsapp:+19999999999")


def test_lookup_is_case_insensitive(tmp_path, monkeypatch):
    """Identity keys normalize to lowercase on write-read roundtrip — so
    the lookup works regardless of how homer formatted the keys."""
    path = _write_map(tmp_path, {"Email:EBBY@Joybuild.AI": "Person:Ebby"})
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(path))

    result = get_distinct_id("ebby@joybuild.ai", "email")
    assert result == _hash_identity_key("person:ebby")


def test_missing_file_is_silent(tmp_path, monkeypatch):
    """Env var points at a file that doesn't exist → fall through, don't crash."""
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(tmp_path / "nope.json"))
    # Should behave as if no map.
    result = get_distinct_id("+15551234", "whatsapp")
    assert result == _hash_identity_key("whatsapp:+15551234")


def test_corrupt_file_is_silent(tmp_path, monkeypatch, caplog):
    """Invalid JSON in map file → warn, fall through."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid")
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(bad))

    with caplog.at_level("WARNING"):
        result = get_distinct_id("+15551234", "whatsapp")
    assert result == _hash_identity_key("whatsapp:+15551234")
    assert any("identity map" in r.message.lower() for r in caplog.records)


def test_map_not_an_object_is_silent(tmp_path, monkeypatch):
    """A JSON list instead of an object is not a valid map."""
    bad = tmp_path / "list.json"
    bad.write_text('["not", "a", "dict"]')
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(bad))

    assert get_distinct_id("+15551234", "whatsapp") == _hash_identity_key("whatsapp:+15551234")


def test_map_rewrite_invalidates_cache(tmp_path, monkeypatch):
    """Rewriting the map file without a process restart must be picked up
    — mtime is part of the cache key."""
    path = _write_map(tmp_path, {"whatsapp:111": "person:a"})
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(path))

    before = get_distinct_id("111", "whatsapp")
    assert before == _hash_identity_key("person:a")

    # Rewrite: now 111 is person:b. Bump mtime explicitly so the assertion
    # doesn't depend on fs timestamp resolution.
    import os
    import time
    path.write_text(json.dumps({"whatsapp:111": "person:b"}))
    later = time.time() + 1
    os.utime(path, (later, later))

    after = get_distinct_id("111", "whatsapp")
    assert after == _hash_identity_key("person:b")
    assert before != after


def test_migrate_channel_hashes_adds_canonical(tmp_path, monkeypatch):
    """Channel-scoped hashes in seen_users → canonical hashes added in-place.
    Returns the count added so callers know whether to persist."""
    path = _write_map(tmp_path, {
        "whatsapp:1": "person:ebby",
        "telegram:2": "person:ebby",
    })
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(path))

    seen = {_hash_identity_key("whatsapp:1"), _hash_identity_key("telegram:2")}
    added = migrate_channel_hashes(seen)
    assert added == 1
    assert _hash_identity_key("person:ebby") in seen


def test_migrate_channel_hashes_is_idempotent(tmp_path, monkeypatch):
    path = _write_map(tmp_path, {"whatsapp:1": "person:ebby"})
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(path))

    seen = {_hash_identity_key("whatsapp:1"), _hash_identity_key("person:ebby")}
    assert migrate_channel_hashes(seen) == 0


def test_migrate_channel_hashes_noop_when_no_map(tmp_path, monkeypatch):
    monkeypatch.delenv("HOMER_IDENTITY_MAP", raising=False)
    seen = {_hash_identity_key("whatsapp:1")}
    before = set(seen)
    assert migrate_channel_hashes(seen) == 0
    assert seen == before


# ── channel-specific lookup normalization ────────────────────────────────


def test_telegram_username_suffix_is_stripped(tmp_path, monkeypatch):
    """python-telegram-bot produces '<id>|<username>' as sender_id when
    the user has a username set. users.yaml only records the id, so the
    lookup must strip the suffix."""
    path = _write_map(tmp_path, {"telegram:1973156656": "person:ebby_anjorin"})
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(path))

    with_suffix = get_distinct_id("1973156656|ebbyanj", "telegram")
    bare = get_distinct_id("1973156656", "telegram")
    assert with_suffix == bare == _hash_identity_key("person:ebby_anjorin")


def test_telegram_suffix_strip_does_not_leak_to_other_channels(tmp_path, monkeypatch):
    """The '|' strip is telegram-only — whatsapp/email identifiers with '|'
    (unlikely but not impossible) must hash as-is."""
    path = _write_map(tmp_path, {"whatsapp:1973156656": "person:e"})
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(path))

    wa_weird = get_distinct_id("1973156656|ebbyanj", "whatsapp")
    # No canonical hit on the pipe form; fall through to channel-scoped.
    assert wa_weird == _hash_identity_key("whatsapp:1973156656|ebbyanj")


def test_telegram_suffix_without_map_entry_falls_through(tmp_path, monkeypatch):
    """If neither the raw nor the bare form is in the map, fall through
    to channel-scoped hash of the original input — no silent swap to bare."""
    path = _write_map(tmp_path, {"telegram:999": "person:other"})
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(path))

    result = get_distinct_id("1973156656|ebbyanj", "telegram")
    assert result == _hash_identity_key("telegram:1973156656|ebbyanj")


def test_telegram_empty_id_before_pipe_falls_through(tmp_path, monkeypatch):
    """Malformed '|username' with no id must not trigger a bogus
    lookup of `telegram:`."""
    path = _write_map(tmp_path, {"telegram:": "person:anyone"})
    monkeypatch.setenv("HOMER_IDENTITY_MAP", str(path))

    result = get_distinct_id("|ghost", "telegram")
    # Falls through to channel-scoped hash of the raw identifier —
    # does NOT match the `telegram:` entry.
    assert result == _hash_identity_key("telegram:|ghost")
