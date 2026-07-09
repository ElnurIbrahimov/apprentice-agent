package com.aura.data

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Pin the canonical default-model id so a future regression cannot
 * silently break the first chat again.
 *
 * 2026-07-09 bug: [DEFAULT_MODEL] was set to `"ollama:deepseek-v4-pro:cloud"`.
 * The `:cloud` suffix is not a real Ollama Cloud model tag (the
 * `/v1/models` endpoint returns `deepseek-v4-pro` with no suffix),
 * so the very first chat the user sent after onboarding returned a
 * 404 from Ollama Cloud and the user saw an opaque error.
 *
 * Verified against the live `https://ollama.com/v1/models` snapshot
 * on 2026-07-09: the bare id `deepseek-v4-pro` is present; the
 * `:cloud` variant is not.
 */
class DefaultModelTest {

    @Test
    fun defaultModelIdIsKnownGoodOllamaCloudModel() {
        assertEquals("ollama:deepseek-v4-pro", DEFAULT_MODEL)
    }

    @Test
    fun defaultModelIdDoesNotIncludePhantomCloudSuffix() {
        // Defensive: if a future contributor reverts to the `:cloud`
        // pattern, this catches it before it ships.
        assertFalse(
            DEFAULT_MODEL.endsWith(":cloud"),
            "DEFAULT_MODEL ends with ':cloud' but Ollama Cloud does not " +
                "serve models with a ':cloud' suffix. Verified against " +
                "https://ollama.com/v1/models on 2026-07-09.",
        )
    }

    @Test
    fun defaultModelIdHasOllamaPrefix() {
        assertTrue(
            DEFAULT_MODEL.startsWith("ollama:"),
            "DEFAULT_MODEL must use the ollama provider prefix so the " +
                "ProviderRegistry routes it to OllamaCloudProvider, not " +
                "MoAProvider or anything else.",
        )
    }
}
