package com.aura.data

import org.junit.Test
import kotlin.test.assertEquals

/**
 * Pin the canonical default-model id.
 *
 * Background: the 2026-07-09 session verified directly against
 * https://ollama.com/v1/models that every Ollama Cloud model id has
 * the `:cloud` suffix on the wire. A bare id like
 * `ollama:deepseek-v4-pro` 404s. The picker refreshes from the live
 * `/v1/models` endpoint on open, so the canonical answer for any
 * unknown id is the live API — not a regex.
 */
class DefaultModelTest {

    @Test
    fun defaultModelIdIsKnownGoodOllamaCloudModel() {
        // Live id per https://ollama.com/v1/models on 2026-07-09.
        // Direct API call confirmed `ollama:deepseek-v4-pro` (no suffix)
        // returns 404; `ollama:deepseek-v4-pro:cloud` returns 200.
        assertEquals("ollama:deepseek-v4-pro:cloud", DEFAULT_MODEL)
    }

    @Test
    fun normalizeIsCurrentlyAnIdentityFunction() {
        // normalizeModelId is reserved as a single chokepoint for
        // future id canonicalization. Today it returns the input
        // unchanged — the live `/v1/models` API is the source of
        // truth, and stripping any suffix would actively make the
        // id wrong (see defaultModelIdIsKnownGoodOllamaCloudModel).
        assertEquals("ollama:deepseek-v4-pro:cloud", normalizeModelId("ollama:deepseek-v4-pro:cloud"))
        assertEquals("ollama:kimi-k2.6:cloud", normalizeModelId("ollama:kimi-k2.6:cloud"))
        assertEquals("ollama:minimax-m2.7:cloud", normalizeModelId("ollama:minimax-m2.7:cloud"))
        assertEquals("ollama:gemma4:31b-cloud", normalizeModelId("ollama:gemma4:31b-cloud"))
        assertEquals("anthropic:claude-sonnet-4-5", normalizeModelId("anthropic:claude-sonnet-4-5"))
        assertEquals("openai:gpt-5", normalizeModelId("openai:gpt-5"))
    }
}
