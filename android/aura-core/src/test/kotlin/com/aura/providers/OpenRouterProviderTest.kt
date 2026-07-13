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
 * MockWebServer tests for [OpenRouterProvider.listModels].
 *
 * OpenRouterProvider inherits from OpenAiCompatProvider so the listModels
 * contract is identical. These tests also verify the custom interceptor
 * headers (HTTP-Referer, X-Title) are present.
 */
class OpenRouterProviderTest {

    private lateinit var server: MockWebServer
    private lateinit var provider: OpenRouterProvider

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
        val keys = mockk<ProviderKeys> {
            every { keyFor("openrouter") } returns "test-api-key"
            every { isConfigured("openrouter") } returns true
        }
        provider = OpenRouterProvider(
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
        val unconfigured = OpenRouterProvider(
            providerKeys = mockk {
                every { keyFor("openrouter") } returns null
                every { isConfigured("openrouter") } returns false
            },
            httpClient = OkHttpClient(),
        )
        assertFalse(unconfigured.isConfigured())
    }

    @Test
    fun `prefix and displayName are correct`() {
        assertEquals("openrouter", provider.prefix)
        assertEquals("OpenRouter", provider.displayName)
    }

    @Test
    fun `listModels returns model IDs from valid response`() = runBlocking<Unit> {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("""{"data":[{"id":"openai/gpt-4o"},{"id":"anthropic/claude-sonnet-4"},{"id":"google/gemini-2.5-pro"}]}""")
        )
        val models = provider.listModels()
        assertEquals(
            listOf("openai/gpt-4o", "anthropic/claude-sonnet-4", "google/gemini-2.5-pro"),
            models,
        )
    }

    @Test
    fun `listModels sends HTTP-Referer and X-Title headers`() = runBlocking<Unit> {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("""{"data":[{"id":"model-a"}]}""")
        )
        provider.listModels()
        val recorded = server.takeRequest()
        assertEquals("https://aura-android", recorded.getHeader("HTTP-Referer"))
        assertEquals("Aura Android", recorded.getHeader("X-Title"))
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
        server.enqueue(MockResponse().setResponseCode(503))
        assertFailsWith<ProviderCatalogException.NetworkException> {
            provider.listModels()
        }
    }

    @Test
    fun `listModels throws MalformedResponseException on invalid JSON`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(200).setBody("broken"))
        assertFailsWith<ProviderCatalogException.MalformedResponseException> {
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
