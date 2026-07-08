package com.aura.ui.components

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Lock the [estimateTokensPerSecond] heuristic. Aura Web shows
 * "42 tok/s" live next to the streaming cursor, computed the
 * same way: chars divided by 4 (the standard LLM
 * chars-per-token rule of thumb), then divided by elapsed
 * seconds.
 */
class StreamingTokensPerSecondTest {

    @Test
    fun `first half-second reports zero tokens to avoid flicker`() {
        // Half a second of streaming isn't enough to trust
        // the rate. The web also delays showing the badge
        // until a chunk has been received for >= 0.5s.
        val result = estimateTokensPerSecond(
            charCount = 80,
            startTimeMs = 1_000L,
            nowMs = 1_400L,
        )
        assertEquals(0, result)
    }

    @Test
    fun `400 chars in 1 second reports 100 tok_s`() {
        // 400 chars / 4 chars-per-token = 100 tokens in
        // 1 second = 100 tok/s.
        val result = estimateTokensPerSecond(
            charCount = 400,
            startTimeMs = 1_000L,
            nowMs = 2_000L,
        )
        assertEquals(100, result)
    }

    @Test
    fun `800 chars in 2 seconds reports 100 tok_s`() {
        val result = estimateTokensPerSecond(
            charCount = 800,
            startTimeMs = 1_000L,
            nowMs = 3_000L,
        )
        assertEquals(100, result)
    }

    @Test
    fun `empty stream reports zero tokens even after delay`() {
        // 0 chars → 0 tokens regardless of elapsed time.
        val result = estimateTokensPerSecond(
            charCount = 0,
            startTimeMs = 1_000L,
            nowMs = 5_000L,
        )
        assertEquals(0, result)
    }

    @Test
    fun `rate never goes below zero even if clock drifts backward`() {
        // If `nowMs` is somehow less than `startTimeMs` (e.g.
        // NTP step), the elapsed is clamped to 1ms. We don't
        // want negative rates leaking into the UI.
        val result = estimateTokensPerSecond(
            charCount = 4_000,
            startTimeMs = 5_000L,
            nowMs = 1_000L,
        )
        assertTrue(result >= 0)
    }
}
