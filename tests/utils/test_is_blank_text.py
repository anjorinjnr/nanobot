"""Tests for is_blank_text — guards against invisible-content reply regressions.

Background: on 2026-05-12 a WhatsApp reply on Homer's hosted deployment was
emitted as just U+200B (zero-width space). It passed the prior
``not content.strip()`` check because .strip() does not remove zero-width
characters, so the user received an invisible message after a 22-minute loop.
These tests pin the fix: anything that renders as nothing in a messaging UI
must be classified as blank so the empty-final fallback fires.
"""

from nanobot.utils.runtime import is_blank_text


class TestStandardBlankCases:
    def test_none_is_blank(self):
        assert is_blank_text(None) is True

    def test_empty_string_is_blank(self):
        assert is_blank_text("") is True

    def test_whitespace_only_is_blank(self):
        assert is_blank_text("   ") is True

    def test_newlines_and_tabs_blank(self):
        assert is_blank_text("\n\t\r ") is True


class TestInvisibleCharsTreatedAsBlank:
    def test_zero_width_space_alone_is_blank(self):
        # The actual 2026-05-12 regression case.
        assert is_blank_text("​") is True

    def test_zero_width_non_joiner_is_blank(self):
        assert is_blank_text("‌") is True

    def test_zero_width_joiner_is_blank(self):
        assert is_blank_text("‍") is True

    def test_word_joiner_is_blank(self):
        assert is_blank_text("⁠") is True

    def test_bom_is_blank(self):
        assert is_blank_text("﻿") is True

    def test_all_invisibles_combined_is_blank(self):
        assert is_blank_text("​‌‍⁠﻿") is True

    def test_invisibles_sandwiched_with_whitespace_is_blank(self):
        # A pathological case: whitespace and zero-widths interleaved.
        assert is_blank_text(" ​ ​ ") is True


class TestRealContentNotBlank:
    def test_simple_text_not_blank(self):
        assert is_blank_text("hello") is False

    def test_single_char_not_blank(self):
        assert is_blank_text("h") is False

    def test_zero_widths_wrapping_real_content_not_blank(self):
        # Real content with invisible characters around it should NOT be
        # treated as blank — only the visible content matters for the check.
        assert is_blank_text("​ hello ​") is False

    def test_real_response_not_blank(self):
        assert is_blank_text("Awesome, you're all set!") is False
