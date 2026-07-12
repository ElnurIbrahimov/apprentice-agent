package com.aura.usage

import com.aura.providers.Usage
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class UsageTrackerTest {

    @Test
    fun `reported provider tokens override character estimate`() {
        val tracker = UsageTracker()

        tracker.recordLlmCall(
            modelId = "openai:test-model",
            inputChars = 4_000,
            outputChars = 2_000,
            reportedUsage = Usage(promptTokens = 100, completionTokens = 50, totalTokens = 150),
        )

        val snapshot = tracker.snapshot.value
        assertEquals(100, snapshot.promptTokens)
        assertEquals(50, snapshot.completionTokens)
        assertEquals(1, snapshot.calls)
        assertFalse(snapshot.models.single().estimated)
    }

    @Test
    fun `missing provider usage is estimated and accumulated by model`() {
        val tracker = UsageTracker()

        tracker.recordLlmCall("ollama:model-a", inputChars = 400, outputChars = 200)
        tracker.recordLlmCall("ollama:model-a", inputChars = 200, outputChars = 200)
        tracker.recordToolResult(800)

        val snapshot = tracker.snapshot.value
        assertEquals(150, snapshot.promptTokens)
        assertEquals(100, snapshot.completionTokens)
        assertEquals(2, snapshot.calls)
        assertEquals(800, snapshot.toolResultChars)
        assertTrue(snapshot.models.single().estimated)
    }

    @Test
    fun `reset clears persistent and session counters`() {
        val tracker = UsageTracker()
        tracker.recordLlmCall("gemini:model", 100, 100)

        tracker.reset()

        assertEquals(UsageSnapshot(), tracker.snapshot.value)
    }
}
