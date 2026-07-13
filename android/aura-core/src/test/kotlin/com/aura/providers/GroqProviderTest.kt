package com.aura.providers

import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.OkHttpClient
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * MockWebServer tests for [GroqProvider.listModels].
 *
 * GroqProvider inherits from OpenAiCompatProvider so the listModels contract
 * is identical. These tests verify that GroqProvider-specific construction
 * (default models, prefix, etc.) is correct.
 */
class GroqProviderTest {

    private lateinit var server: MockWebServer
    private lateinit var provider: GroqProvider

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
        val keys = mockk<ProviderKeys> {
            every { keyFor("groq") } returns "test-api-key"
            every { isConfigured("groq") } returns true
        }
        provider = GroqProvider(
            providerKeys = keys,
            httpClient = OkHttpClient(),
            baseUrl = server.url("/").toString().removeSuffix("/"),
        )
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun `isConfigured returns true when API key is set`() {
        assertTrue(provider.isConfigured())
    }

    @Test
    fun `isConfigured returns false when API key is not set`() {
        val unconfigured = GroqProvider(
            providerKeys = mockk {
                every { keyFor("groq") } returns null
                every { isConfigured("groq") } returns false
            },
            httpClient = OkHttpClient(),
        )
        assertFalse(unconfigured.isConfigured())
    }

    @Test
    fun `prefix and displayName are correct`() {
        assertEquals("groq", provider.prefix)
        assertEquals("Groq", provider.displayName)
    }

    @Test
    fun `listModels returns model IDs from valid response`() = runBlocking<Unit> {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("""{"data":[{"id":"llama-3.3-70b-versatile"},{"id":"mixtral-8x7b-32768"},{"id":"gemma2-9b-it"}]}""")
        )
        val models = provider.listModels()
        assertEquals(listOf("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"), models)
    }

    @Test
    fun `listModels throws AuthenticationException on 401`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(401))
        assertFailsWith<ProviderCatalogException.AuthenticationException> {
            provider.listModels()
        }
    }

    @Test
    fun `listModels throws RateLimitedException on 429`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(429))
        assertFailsWith<ProviderCatalogException.RateLimitedException> {
            provider.listModels()
        }
    }

    @Test
    fun `listModels throws NetworkException on 5xx`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(502))
        assertFailsWith<ProviderCatalogException.NetworkException> {
            provider.listModels()
        }
    }

    @Test
    fun `listModels throws EmptyCatalogException when data is empty`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(200).setBody("""{"data":[]}"""))
        assertFailsWith<ProviderCatalogException.EmptyCatalogException> {
            provider.listModels()
        }
    }

    @Test
    fun `cancel does not throw when no active call`() = runBlocking<Unit> {
        provider.cancel()
    }
}
