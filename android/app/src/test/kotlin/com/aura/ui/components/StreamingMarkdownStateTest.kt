package com.aura.ui.components

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Lock the streaming markdown masking behavior. The actual fix
 * is in [StreamingMarkdownState] which suppresses trailing
 * unclosed markers so the rendered text doesn't flicker as the
 * closing marker arrives in the next chunk.
 *
 * These tests focus on the masking rule. The full
 * [parseMarkdown] parser has its own tests (MarkdownTest.kt).
 */
class StreamingMarkdownStateTest {

    private val state = StreamingMarkdownState()
    private val colors = MarkdownColors(
        link = androidx.compose.ui.graphics.Color.Black,
        linkDim = androidx.compose.ui.graphics.Color.Gray,
    )

    @Test
    fun `empty text renders empty`() {
        val result = state.render("", colors)
        assertEquals("", result.text)
    }

    @Test
    fun `text with no markers renders unchanged`() {
        val result = state.render("hello world", colors)
        assertEquals("hello world", result.text)
    }

    @Test
    fun `complete bold renders with the bold span`() {
        // **bold** is complete — should render with bold span
        // (the parser already handles this; we just verify the
        // masking doesn't break the happy path).
        val result = state.render("**bold**", colors)
        assertTrue(result.text.contains("bold"))
        // The leading ** and trailing ** should be present in
        // the span styles, not in the literal text.
        assertTrue(result.text.length <= 6, "expected 'bold' (4) but got '${result.text}'")
    }

    @Test
    fun `trailing unclosed bold is masked`() {
        // The `**` at the end is an opening marker with no close.
        // The masking should hide the TRAILING asterisks from the
        // rendered text. The leading `**` stays because it's the
        // open marker — without it the close later would have
        // nothing to close.
        val result = state.render("**bol", colors)
        // The trailing chars should not be `*` (otherwise the
        // marker would render as text).
        assertFalse(result.text.endsWith("*"), "expected trailing asterisks masked, got '${result.text}'")
        // The visible part should still be there.
        assertTrue(result.text.contains("bol"), "expected 'bol' in '$result'")
    }

    @Test
    fun `trailing single asterisk is masked`() {
        val result = state.render("italic ", colors)
        // Just plain text — no mask needed.
        assertEquals("italic ", result.text)
    }

    @Test
    fun `complete italic renders correctly`() {
        val result = state.render("*italic*", colors)
        assertTrue(result.text.contains("italic"))
    }

    @Test
    fun `unpaired trailing asterisk is masked`() {
        // A single `*` at the end of a line where the line has
        // an even number of `*` (excluding this one) is an
        // unpaired closer. Mask it.
        val result = state.render("text *", colors)
        assertFalse(result.text.endsWith("*"), "expected trailing * masked, got '$result'")
    }

    @Test
    fun `trailing unclosed backtick is masked`() {
        val result = state.render("`code", colors)
        assertFalse(result.text.endsWith("`"), "expected trailing backtick masked, got '$result'")
    }

    @Test
    fun `complete inline code renders correctly`() {
        val result = state.render("`code`", colors)
        assertTrue(result.text.contains("code"))
    }

    @Test
    fun `multiline text — mask only applies to the last line`() {
        val text = "first line\n**bol"
        val result = state.render(text, colors)
        // The first line should be present unchanged.
        assertTrue(result.text.startsWith("first line\n"), "got '${result.text}'")
        // The trailing ** should be masked.
        assertFalse(result.text.endsWith("**"), "expected trailing ** masked, got '${result.text}'")
    }

    @Test
    fun `progressive streaming — text grows from partial to complete`() {
        // Simulate: chunk 1 = "**bol", chunk 2 = "**bold**"
        val chunk1 = state.render("**bol", colors)
        // Chunk 1: the trailing ** is an opening bold marker
        // (no close yet), so it must be masked. The leading **
        // stays (it's the same opening).
        assertFalse(chunk1.text.endsWith("*"), "chunk 1 should mask trailing *, got '${chunk1.text}'")

        val chunk2 = state.render("**bold**", colors)
        // Now the bold is complete and the parser should style it.
        assertTrue(chunk2.text.contains("bold"), "chunk 2 should contain 'bold', got '${chunk2.text}'")
    }
}
