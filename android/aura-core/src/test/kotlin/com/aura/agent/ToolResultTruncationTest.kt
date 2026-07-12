package com.aura.agent

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for [truncateToolResult], the safety net applied to every
 * tool result before it lands in the conversation history. Without
 * this, a single web-search or deep-research tool could blow the
 * context window. Per-tool truncation (Firecrawl 8k, DeepResearch
 * 6k) is the first line of defense; this catches everything else.
 */
class ToolResultTruncationTest {

    @Test
    fun `short result is returned unchanged`() {
        val raw = "small result, well under budget"
        assertEquals(raw, truncateToolResult(raw))
    }

    @Test
    fun `result exactly at budget is returned unchanged`() {
        val raw = "x".repeat(4_000)
        assertEquals(raw, truncateToolResult(raw))
    }

    @Test
    fun `result over budget is truncated and marked`() {
        val raw = "x".repeat(10_000)
        val out = truncateToolResult(raw)
        assertTrue(out.startsWith("x".repeat(4_000)),
            "should keep the first 4000 chars")
        assertTrue(out.contains("[...truncated"),
            "should include the truncation marker, got: ${out.takeLast(120)}")
        assertTrue(out.length < 10_000,
            "truncated output should be much shorter than the original")
    }

    @Test
    fun `empty result is returned as empty string`() {
        assertEquals("", truncateToolResult(""))
    }
}
