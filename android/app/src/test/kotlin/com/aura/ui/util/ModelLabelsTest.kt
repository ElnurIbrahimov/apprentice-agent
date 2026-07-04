package com.aura.ui.util

import org.junit.Test
import kotlin.test.assertEquals

/**
 * Pins the model-display-name mapping that the History screen
 * uses to label each saved conversation. Centralizing this
 * prevents the well-known models from drifting out of sync with
 * the user-facing label shown in the chat header.
 */
class ModelLabelsTest {

    @Test
    fun `known model gets a friendly name plus the provider`() {
        assertEquals(
            "DeepSeek V4 Pro · ollama",
            modelDisplayName("ollama:deepseek-v4-pro:cloud"),
        )
    }

    @Test
    fun `anthropic model is human-readable`() {
        assertEquals(
            "Claude Sonnet 4.5 · anthropic",
            modelDisplayName("anthropic:claude-sonnet-4-5"),
        )
    }

    @Test
    fun `unknown model falls back to the raw id with the provider prefix`() {
        // Pattern: "<model-segment> · <provider>". A model that
        // isn't in the friendliness table still gets a useful
        // string instead of a blank or a crash.
        assertEquals(
            "some-future-model · ollama",
            modelDisplayName("ollama:some-future-model"),
        )
    }

    @Test
    fun `id with no colon falls back to the raw string`() {
        // Defensive: a malformed id like "garbage" shouldn't crash
        // the screen. With no colon, the only segment is used as
        // both provider and model — not pretty, but non-empty and
        // never throws.
        assertEquals("garbage · garbage", modelDisplayName("garbage"))
    }
}
