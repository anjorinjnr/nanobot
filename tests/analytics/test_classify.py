"""Tests for nanobot/analytics/classify.py — validator + LRU cache.

The async Gemini call itself is exercised in production; here we focus on
the pure-Python pieces (validator regex, cache eviction) that protect us
from regressions like the 2-char "ch" truncation we hit when Gemini 2.5's
extended thinking ate the response budget.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nanobot.analytics.classify import (
    _FALLBACK,
    _LRUCache,
    PREFERRED_TAGS,
    _validate,
    classify_message_async,
    _cache,
)


class TestValidate:
    def test_accepts_preferred_tag(self):
        assert _validate("calendar") == "calendar"

    def test_accepts_generated_snake_case(self):
        assert _validate("birthday_planning") == "birthday_planning"

    def test_strips_quotes_and_whitespace(self):
        assert _validate('  "meal_planning"  ') == "meal_planning"

    def test_lowercases(self):
        assert _validate("Calendar") == "calendar"

    def test_rejects_literal_other(self):
        assert _validate("other") == _FALLBACK

    def test_rejects_multiword_with_space(self):
        assert _validate("meal planning") == _FALLBACK

    def test_rejects_hyphen(self):
        assert _validate("meal-planning") == _FALLBACK

    def test_rejects_empty(self):
        assert _validate("") == _FALLBACK

    def test_rejects_starts_with_digit(self):
        assert _validate("1st_task") == _FALLBACK

    def test_rejects_over_30_chars(self):
        assert _validate("a" * 31) == _FALLBACK

    def test_rejects_two_char_truncation(self):
        # "ch" appeared in production as a truncated "chitchat" — min length
        # 3 blocks these mid-token clips from polluting the analytics stream.
        assert _validate("ch") == _FALLBACK
        assert _validate("my") == _FALLBACK

    def test_accepts_three_char_tag(self):
        assert _validate("car") == "car"


class TestPreferredTags:
    def test_shape(self):
        assert isinstance(PREFERRED_TAGS, tuple)
        for tag in PREFERRED_TAGS:
            assert isinstance(tag, str)
            assert tag == tag.lower()

    def test_load_bearing_members_present(self):
        for must_exist in ("calendar", "events", "finance", "health", "email", "chitchat"):
            assert must_exist in PREFERRED_TAGS

    def test_other_is_not_in_preferred(self):
        # "other" is deliberately removed — provided no signal.
        assert "other" not in PREFERRED_TAGS


class TestLRUCache:
    def test_basic_put_get(self):
        cache = _LRUCache(maxsize=3)
        cache.put("a", "meal_planning")
        assert cache.get("a") == "meal_planning"

    def test_evicts_oldest(self):
        cache = _LRUCache(maxsize=2)
        cache.put("a", "v1")
        cache.put("b", "v2")
        cache.put("c", "v3")
        assert cache.get("a") is None

    def test_access_refreshes_order(self):
        cache = _LRUCache(maxsize=2)
        cache.put("a", "v1")
        cache.put("b", "v2")
        cache.get("a")
        cache.put("c", "v3")
        assert cache.get("b") is None
        assert cache.get("a") == "v1"


class TestClassifyAsync:
    @pytest.mark.asyncio
    async def test_cache_hit_avoids_api_call(self):
        _cache._data.clear()
        with patch("nanobot.analytics.classify._call_gemini_async", return_value="calendar") as mock:
            r1 = await classify_message_async("schedule a meeting")
            r2 = await classify_message_async("schedule a meeting")
            assert r1 == r2 == "calendar"
            assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_falls_back_on_exception(self):
        _cache._data.clear()
        with patch("nanobot.analytics.classify._call_gemini_async", side_effect=Exception("boom")):
            assert await classify_message_async("anything") == _FALLBACK
