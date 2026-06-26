package com.aura.providers

import io.mockk.every
import io.mockk.mockk
import okhttp3.OkHttpClient
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Unit tests for [OpenRouterProvider].
 *
 * These tests verify the provider contract without making real HTTP calls.
 * The chat() streaming path requires MockWebServer (not yet a dependency);
 * it will be added in a follow-up when cross-provider streaming tests are
 * introduced.
 */
class OpenRouterProviderTest {

    private fun createProvider(configured: Boolean): OpenRouterProvider {
        val keys = mockk<ProviderKeys> {
            every { keyFor("openrouter") } returns (if (configured) "test-api-key" else "")
            every { isConfigured("openrouter") } returns configured
        }
        return OpenRouterProvider(
            providerKeys = keys,
            httpClient = OkHttpClient(),
        )
    }

    @Test
    fun `isConfigured returns true when API key is set`() {
        val provider = createProvider(configured = true)
        assertTrue(provider.isConfigured())
    }

    @Test
    fun `isConfigured returns false when API key is not set`() {
        val provider = createProvider(configured = false)
        assertFalse(provider.isConfigured())
    }

    @Test
    fun `prefix and displayName are correct`() {
        val provider = createProvider(configured = false)
        assertEquals("openrouter", provider.prefix)
        assertEquals("OpenRouter", provider.displayName)
    }

    @Test
    fun `listModels returns hardcoded OpenRouter models`() = kotlinx.coroutines.runBlocking {
        val provider = createProvider(configured = true)
        val models = provider.listModels()
        assertEquals(
            listOf("gpt-4o", "claude-3.5-sonnet", "deepseek-v3"),
            models,
        )
    }

    @Test
    fun `cancel does not throw when no active call`() {
        val provider = createProvider(configured = true)
        // Should complete without exception
        kotlinx.coroutines.runBlocking {
            provider.cancel()
        }
    }
}
