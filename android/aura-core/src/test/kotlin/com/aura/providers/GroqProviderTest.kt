package com.aura.providers

import io.mockk.every
import io.mockk.mockk
import okhttp3.OkHttpClient
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Unit tests for [GroqProvider].
 *
 * These tests verify the provider contract without making real HTTP calls.
 * The chat() streaming path requires MockWebServer (not yet a dependency);
 * it will be added in a follow-up when cross-provider streaming tests are
 * introduced.
 */
class GroqProviderTest {

    private fun createProvider(configured: Boolean): GroqProvider {
        val keys = mockk<ProviderKeys> {
            every { keyFor("groq") } returns (if (configured) "test-api-key" else "")
            every { isConfigured("groq") } returns configured
        }
        return GroqProvider(
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
        assertEquals("groq", provider.prefix)
        assertEquals("Groq", provider.displayName)
    }

    @Test
    fun `listModels returns default Groq models`() = kotlinx.coroutines.runBlocking {
        val provider = createProvider(configured = true)
        val models = provider.listModels()
        assertEquals(
            listOf("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"),
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
