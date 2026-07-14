package com.aura.widget

import com.aura.agent.Turn
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class QuickAskSessionTest {
    @Test
    fun `widget prefix is isolated from the base compact-session instruction`() {
        val prompt = buildQuickAskSystemPrompt("Prefer bullet points")

        assertTrue(prompt.contains("compact Aura session"))
        assertTrue(prompt.contains("Widget instruction:\nPrefer bullet points"))
    }

    @Test
    fun `blank widget prefix does not emit an empty instruction section`() {
        assertFalse(buildQuickAskSystemPrompt("  ").contains("Widget instruction:"))
    }

    @Test
    fun `latest response skips blank assistant turns`() {
        val turns = listOf(
            Turn(user = "first", assistant = "first answer", timestamp = 1L),
            Turn(user = "tool", assistant = "", timestamp = 2L),
        )

        assertEquals("first answer", latestQuickAskResponse(turns))
    }
}
