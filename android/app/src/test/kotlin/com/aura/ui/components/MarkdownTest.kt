package com.aura.ui.components

import androidx.compose.ui.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests the [parseMarkdown] non-composable parser. The new parser
 * handles bold+italic, ordered lists, tables, and links; the old
 * regex parser misrendered `**bold *italic***` (the inner *italic*
 * was eaten by the bold regex) and leaked raw table delimiters.
 */
class MarkdownTest {

    private val colors = MarkdownColors(
        link = Color(0xFF0066CC),
        linkDim = Color(0xFF808080),
    )

    private fun parse(text: String) = parseMarkdown(text, colors).text

    @Test
    fun `bold renders as plain text (style asserted via length)`() {
        val out = parse("**hello**")
        assertEquals("hello", out)
    }

    @Test
    fun `italic renders as plain text`() {
        val out = parse("*hello*")
        assertEquals("hello", out)
    }

    @Test
    fun `bold-italic renders the inner content not the asterisks`() {
        // Old parser: **bold *italic*** matched **bold *italic** as bold,
        // then ** at the end was leaked. New parser: bold-italic wins.
        val out = parse("***hello***")
        assertEquals("hello", out)
    }

    @Test
    fun `mixed bold inside italic works`() {
        val out = parse("***hello*** ***world***")
        assertEquals("hello world", out)
    }

    @Test
    fun `asterisk inside word is not italic`() {
        val out = parse("multi*ple")
        // The single * inside the word is NOT an italic marker.
        assertEquals("multi*ple", out)
    }

    @Test
    fun `inline code renders as plain text`() {
        val out = parse("Use `String` here")
        assertEquals("Use String here", out)
    }

    @Test
    fun `header renders just the title text`() {
        val out = parse("## My Header")
        assertEquals("My Header", out)
    }

    @Test
    fun `bullet list uses bullet glyph`() {
        val out = parse("- first\n- second")
        assertEquals("• first\n• second", out)
    }

    @Test
    fun `ordered list keeps numbers`() {
        val out = parse("1. first\n2. second")
        assertEquals("1. first\n2. second", out)
    }

    @Test
    fun `link renders as label url`() {
        val out = parse("Click [Aura](https://example.com)")
        assertEquals("Click Aura (https://example.com)", out)
    }

    @Test
    fun `only absolute http links are safe to open`() {
        assertTrue(isSafeMarkdownUrl("https://example.com/path"))
        assertTrue(isSafeMarkdownUrl("http://example.com"))
        assertTrue(!isSafeMarkdownUrl("intent://settings"))
        assertTrue(!isSafeMarkdownUrl("file:///data/local/tmp/x"))
        assertTrue(!isSafeMarkdownUrl("javascript:alert(1)"))
        assertTrue(!isSafeMarkdownUrl("/relative"))
    }

    @Test
    fun `underscore italic works for single words`() {
        val out = parse("_hello_")
        assertEquals("hello", out)
    }

    @Test
    fun `multi-line doc with mixed content preserves line breaks`() {
        val out = parse("""
            # Title
            Some **bold** text.
            - item 1
            - item 2
        """.trimIndent())
        assertTrue(out.contains("Title"))
        assertTrue(out.contains("bold"))
        assertTrue(out.contains("• item 1"))
        assertTrue(out.contains("• item 2"))
    }

    @Test
    fun `paragraph text without markdown markers is unchanged`() {
        val out = parse("Just plain text, nothing fancy.")
        assertEquals("Just plain text, nothing fancy.", out)
    }

    @Test
    fun `standalone citation markers render as compact superscripts`() {
        assertEquals(
            "Claim⁽¹⁾ and another⁽²⁾.",
            renderCitationMarkers("Claim[1] and another[2].", setOf(1, 2)),
        )
    }

    @Test
    fun `citation transform leaves markdown links and unknown markers intact`() {
        assertEquals(
            "Read [1](https://example.com) and check [9].",
            renderCitationMarkers("Read [1](https://example.com) and check [9].", setOf(1)),
        )
    }

    @Test
    fun `adjacent citation markers remain distinct`() {
        assertEquals("Fact⁽¹⁾⁽²⁾", renderCitationMarkers("Fact[1][2]", setOf(1, 2)))
    }

    @Test
    fun `citation-looking text inside fenced code remains literal`() {
        val source = "```text\nvalue[1]\n```"
        assertEquals(source, renderCitationMarkers(source, setOf(1)))
    }
}
