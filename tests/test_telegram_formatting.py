"""Tests for aura.messaging.telegram_formatting."""
import pytest
from aura.messaging.telegram_formatting import (
    escape_mdv2,
    format_telegram_response,
    split_message,
    format_research_citations,
)


class TestEscapeMdv2:
    def test_escapes_underscore(self):
        assert escape_mdv2("hello_world") == r"hello\_world"

    def test_escapes_multiple_special_chars(self):
        assert escape_mdv2("a*b[c]d") == r"a\*b\[c\]d"

    def test_empty_string(self):
        assert escape_mdv2("") == ""

    def test_no_special_chars(self):
        assert escape_mdv2("hello world") == "hello world"

    def test_escapes_dot(self):
        assert escape_mdv2("3.14") == r"3\.14"

    def test_escapes_exclamation(self):
        assert escape_mdv2("wow!") == r"wow\!"

    def test_escapes_backslash(self):
        result = escape_mdv2("a\\b")
        assert "\\\\" in result

    def test_all_special_chars_present(self):
        # All MarkdownV2 specials: _ * [ ] ( ) ~ ` > # + - = | { } . ! \
        result = escape_mdv2("_*[]()~`>#+−=|{}.!\\")
        assert "\\_" in result
        assert "\\*" in result
        assert "\\[" in result
        assert "\\]" in result

    def test_preserves_alphanumeric(self):
        assert escape_mdv2("abc123") == "abc123"


class TestFormatTelegramResponse:
    def test_empty_string(self):
        result = format_telegram_response("")
        assert result == [""]

    def test_plain_text_returned(self):
        result = format_telegram_response("hello world")
        assert len(result) == 1
        assert "hello world" in result[0]

    def test_bold_converted(self):
        result = format_telegram_response("**bold text**")
        # MarkdownV2 bold uses *text*
        assert "*" in result[0]
        assert "bold text" in result[0]

    def test_italic_converted(self):
        result = format_telegram_response("*italic text*")
        # MarkdownV2 italic uses _text_
        assert "_" in result[0]
        assert "italic text" in result[0]

    def test_code_block_preserved(self):
        result = format_telegram_response("```python\nprint('hi')\n```")
        assert "```" in result[0]
        assert "print" in result[0]

    def test_inline_code_preserved(self):
        result = format_telegram_response("use `print()` here")
        assert "`print()`" in result[0]

    def test_link_preserved(self):
        result = format_telegram_response("[click here](https://example.com)")
        assert "https://example.com" in result[0]
        assert "click here" in result[0]

    def test_special_chars_escaped_in_plain_text(self):
        result = format_telegram_response("price is 10.50")
        # The dot should be escaped in MarkdownV2
        assert "\\." in result[0]

    def test_long_text_splits_into_multiple_chunks(self):
        # Create text that exceeds 4096 chars
        text = "Hello world.\n\n" * 300
        result = format_telegram_response(text)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 4096

    def test_returns_list(self):
        result = format_telegram_response("simple text")
        assert isinstance(result, list)


class TestSplitMessage:
    def test_short_message_not_split(self):
        assert split_message("hello", 100) == ["hello"]

    def test_exact_limit_not_split(self):
        text = "a" * 100
        assert split_message(text, 100) == [text]

    def test_splits_at_paragraph_break(self):
        text = "A" * 50 + "\n\n" + "B" * 50
        chunks = split_message(text, 60)
        assert len(chunks) == 2
        assert chunks[0].startswith("A")
        assert chunks[1].startswith("B")

    def test_splits_at_single_newline(self):
        text = "A" * 50 + "\n" + "B" * 50
        chunks = split_message(text, 60)
        assert len(chunks) == 2

    def test_splits_at_space(self):
        # 20 words of 5 chars each = 100 chars with spaces
        text = "word " * 20
        chunks = split_message(text, 50)
        assert len(chunks) >= 2
        # split_message only lstrips newlines, not spaces — first chunk should not
        # end mid-word (it ends at the space boundary via rstrip on the left part)
        assert not chunks[0].endswith(" ")

    def test_hard_cut_when_no_split_point(self):
        text = "A" * 200
        chunks = split_message(text, 100)
        assert len(chunks) == 2
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100

    def test_empty_string(self):
        assert split_message("", 100) == [""]

    def test_chunks_within_limit(self):
        text = "word " * 100
        chunks = split_message(text, 50)
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_three_paragraphs(self):
        text = ("Para one.\n\n" * 3) * 5
        chunks = split_message(text, 30)
        assert len(chunks) >= 3


class TestFormatResearchCitations:
    def test_no_sources_returns_original(self):
        assert format_research_citations("text", []) == "text"

    def test_with_single_source(self):
        sources = [{"url": "https://example.com", "title": "Example Site"}]
        result = format_research_citations("text", sources)
        assert "Sources:" in result
        assert "[1]" in result
        assert "Example Site" in result
        assert "https://example.com" in result

    def test_with_multiple_sources(self):
        sources = [
            {"url": "https://a.com", "title": "A"},
            {"url": "https://b.com", "title": "B"},
        ]
        result = format_research_citations("body", sources)
        assert "[1]" in result
        assert "[2]" in result
        assert "https://a.com" in result
        assert "https://b.com" in result

    def test_deduplicates_by_url(self):
        sources = [
            {"url": "https://example.com", "title": "First"},
            {"url": "https://example.com", "title": "Duplicate"},
        ]
        result = format_research_citations("text", sources)
        # Only one entry for the URL
        assert result.count("[1]") == 1
        assert "[2]" not in result

    def test_source_without_url_skipped(self):
        # Sources without a URL are filtered out by the dedup logic (url="" is falsy),
        # but the "Sources:" header is still appended. The source title is not included.
        sources = [{"title": "No URL Source"}]
        result = format_research_citations("text", sources)
        assert "No URL Source" not in result
        # The header is appended even if no valid sources remain
        assert result.startswith("text")

    def test_original_text_preserved(self):
        sources = [{"url": "https://x.com", "title": "X"}]
        result = format_research_citations("my report body", sources)
        assert result.startswith("my report body")

    def test_empty_sources_list(self):
        result = format_research_citations("only text", [])
        assert result == "only text"
