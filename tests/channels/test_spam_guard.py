"""Tests for ChannelManager spam guard (dedup of repeated messages)."""
import time
from unittest.mock import MagicMock, patch

import pytest

from nanobot.bus.events import TASK_TAG_META_KEY, OutboundMessage
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import SpamGuardConfig


def _make_manager(enabled=True, window_s=300, max_repeats=2):
    """Create a ChannelManager with spam guard config (no actual channels)."""
    mgr = object.__new__(ChannelManager)
    mgr._dedup_log = {}
    mgr._dedup_last_cleanup = 0.0
    mgr.config = MagicMock()
    mgr.config.channels.spam_guard = SpamGuardConfig(
        enabled=enabled, window_s=window_s, max_repeats=max_repeats,
    )
    return mgr


def _msg(content="hello", channel="whatsapp", chat_id="user1", metadata=None):
    return OutboundMessage(
        channel=channel,
        chat_id=chat_id,
        content=content,
        metadata=metadata or {},
    )


class TestSpamGuard:
    def test_first_message_not_spam(self):
        mgr = _make_manager()
        assert mgr._is_spam(_msg()) is False

    def test_second_identical_message_not_spam(self):
        mgr = _make_manager()
        mgr._is_spam(_msg())
        assert mgr._is_spam(_msg()) is False

    def test_third_identical_message_is_spam(self):
        mgr = _make_manager()
        mgr._is_spam(_msg())
        mgr._is_spam(_msg())
        assert mgr._is_spam(_msg()) is True

    def test_different_content_not_spam(self):
        mgr = _make_manager()
        mgr._is_spam(_msg("hello"))
        mgr._is_spam(_msg("hello"))
        assert mgr._is_spam(_msg("world")) is False

    def test_different_recipient_not_spam(self):
        mgr = _make_manager()
        mgr._is_spam(_msg(chat_id="user1"))
        mgr._is_spam(_msg(chat_id="user1"))
        assert mgr._is_spam(_msg(chat_id="user2")) is False

    def test_different_channel_not_spam(self):
        mgr = _make_manager()
        mgr._is_spam(_msg(channel="whatsapp"))
        mgr._is_spam(_msg(channel="whatsapp"))
        assert mgr._is_spam(_msg(channel="telegram")) is False

    def test_stream_delta_exempt(self):
        mgr = _make_manager()
        delta = _msg(metadata={"_stream_delta": True})
        for _ in range(5):
            assert mgr._is_spam(delta) is False

    def test_stream_end_exempt(self):
        mgr = _make_manager()
        end = _msg(metadata={"_stream_end": True})
        for _ in range(5):
            assert mgr._is_spam(end) is False

    def test_progress_exempt(self):
        mgr = _make_manager()
        progress = _msg(metadata={"_progress": True})
        for _ in range(5):
            assert mgr._is_spam(progress) is False

    def test_empty_content_exempt(self):
        mgr = _make_manager()
        empty = _msg(content="")
        for _ in range(5):
            assert mgr._is_spam(empty) is False

    def test_whitespace_only_exempt(self):
        mgr = _make_manager()
        ws = _msg(content="   \n  ")
        for _ in range(5):
            assert mgr._is_spam(ws) is False

    def test_expires_after_window(self):
        mgr = _make_manager(window_s=300)
        mgr._is_spam(_msg())
        mgr._is_spam(_msg())
        assert mgr._is_spam(_msg()) is True

        with patch("time.monotonic", return_value=time.monotonic() + 301):
            assert mgr._is_spam(_msg()) is False

    def test_custom_max_repeats(self):
        mgr = _make_manager(max_repeats=5)
        for _ in range(5):
            assert mgr._is_spam(_msg()) is False
        assert mgr._is_spam(_msg()) is True

    def test_custom_window(self):
        mgr = _make_manager(window_s=10)
        mgr._is_spam(_msg())
        mgr._is_spam(_msg())
        assert mgr._is_spam(_msg()) is True

        with patch("time.monotonic", return_value=time.monotonic() + 11):
            assert mgr._is_spam(_msg()) is False

    def test_content_whitespace_normalized(self):
        mgr = _make_manager()
        mgr._is_spam(_msg(content="  hello world  "))
        mgr._is_spam(_msg(content="hello world"))
        assert mgr._is_spam(_msg(content="  hello world\n")) is True

    def test_digit_drift_caught_by_normalized_fingerprint(self):
        # Pre-fix the exact-content hash treated "May 18, 2026" and
        # "May 19, 2026" as distinct, so the same templated heartbeat output
        # could spam every minute as the date crept forward. Stripping digits
        # before hashing collapses these into one fingerprint. Wider wording
        # drift (filler words, reordering) is still caught by the heartbeat
        # task-tag path — see test_task_tag_supersedes_content_hash.
        mgr = _make_manager()
        a = "The Balance check task is scheduled for May 18, 2026."
        b = "The Balance check task is scheduled for May 19, 2026."
        c = "The Balance check task is scheduled for May 20, 2026."
        assert mgr._is_spam(_msg(content=a)) is False
        assert mgr._is_spam(_msg(content=b)) is False
        assert mgr._is_spam(_msg(content=c)) is True

    def test_task_tag_supersedes_content_hash(self):
        # When the heartbeat path tags messages with a task identifier, dedup
        # is per (recipient, task) — wording can vary arbitrarily and we
        # still suppress repeats from the same task to the same chat.
        mgr = _make_manager()
        meta = {TASK_TAG_META_KEY: "Balance check"}
        assert mgr._is_spam(_msg(content="apple", metadata=meta)) is False
        assert mgr._is_spam(_msg(content="banana", metadata=meta)) is False
        assert mgr._is_spam(_msg(content="cherry", metadata=meta)) is True

    def test_task_tag_isolated_per_recipient(self):
        mgr = _make_manager()
        meta = {TASK_TAG_META_KEY: "Balance check"}
        for _ in range(3):
            mgr._is_spam(_msg(chat_id="user1", metadata=meta))
        # Other recipient unaffected
        assert mgr._is_spam(_msg(chat_id="user2", metadata=meta)) is False

    def test_disabled_by_default(self):
        """When disabled, _is_spam should still work but dispatch won't call it."""
        mgr = _make_manager(enabled=False)
        # _is_spam itself still functions — it's the dispatch that checks enabled
        assert mgr._is_spam(_msg()) is False
