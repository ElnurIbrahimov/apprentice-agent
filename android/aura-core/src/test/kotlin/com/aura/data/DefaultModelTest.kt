package com.aura.data

import org.junit.Test
import kotlin.test.assertEquals

/**
 * Pin the canonical default-model id and the migration helper that
 * strips the legacy `:cloud` suffix from persisted values.
 *
 * Background: a pre-2026-07-09 install of Aura stored
 * `ollama:deepseek-v4-pro:cloud` as the default model. That id 404s
 * on Ollama Cloud (real ids are bare — `deepseek-v4-pro`). The
 * Settings screen was updated to write the correct ids, but the
 * DataStore value from a previous install persists until something
 * rewrites it. `normalizeModelId` is the self-heal: on every read
 * of [UserPreferences.defaultModel], the stored value is normalized
 * in place.
 */
class DefaultModelTest {

    @Test
    fun defaultModelIdIsKnownGoodOllamaCloudModel() {
        assertEquals("ollama:deepseek-v4-pro", DEFAULT_MODEL)
    }

    @Test
    fun normalizeStripsColonCloudSuffix() {
        assertEquals(
            "ollama:deepseek-v4-pro",
            normalizeModelId("ollama:deepseek-v4-pro:cloud")
        )
        assertEquals(
            "ollama:kimi-k2.7-code",
            normalizeModelId("ollama:kimi-k2.7-code:cloud")
        )
        assertEquals(
            "ollama:minimax-m2.7",
            normalizeModelId("ollama:minimax-m2.7:cloud")
        )
        assertEquals(
            "ollama:gemma4:31b",
            normalizeModelId("ollama:gemma4:31b:cloud")
        )
        assertEquals(
            "ollama:qwen3.5:397b",
            normalizeModelId("ollama:qwen3.5:397b:cloud")
        )
    }

    @Test
    fun normalizeLeavesBareIdsUnchanged() {
        // Already-correct ids from the live /v1/models endpoint.
        assertEquals("ollama:deepseek-v4-pro", normalizeModelId("ollama:deepseek-v4-pro"))
        assertEquals("ollama:kimi-k2.7-code", normalizeModelId("ollama:kimi-k2.7-code"))
        assertEquals("ollama:gemma4:31b", normalizeModelId("ollama:gemma4:31b"))
        // Internal colons must be preserved.
        assertEquals("ollama:qwen3.5:397b", normalizeModelId("ollama:qwen3.5:397b"))
    }

    @Test
    fun normalizeLeavesOtherProvidersUnchanged() {
        // Non-ollama providers with a ":cloud"-like suffix in their
        // real id must NOT be touched. (None exist today but the
        // check is cheap insurance.)
        assertEquals("anthropic:claude-sonnet-4.6", normalizeModelId("anthropic:claude-sonnet-4.6"))
        assertEquals("openai:gpt-5", normalizeModelId("openai:gpt-5"))
    }

    @Test
    fun normalizeHandlesEdgeCases() {
        // A bare "cloud" string (no leading colon segment) is ambiguous —
        // don't touch it, it might be a valid id on some provider.
        assertEquals("cloud", normalizeModelId("cloud"))
        // An id with just one colon shouldn't be touched either;
        // "provider:cloud" might be a real future id (e.g. a custom
        // deployment literally named "cloud").
        assertEquals("provider:cloud", normalizeModelId("provider:cloud"))
    }
}
