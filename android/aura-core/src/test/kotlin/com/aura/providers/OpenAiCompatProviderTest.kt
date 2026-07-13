package com.aura.providers

import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.OkHttpClient
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * MockWebServer tests for [OpenAiCompatProvider.listModels].
 *
 * Every test uses a local MockWebServer so no real network calls are made.
 */
class OpenAiCompatProviderTest {

    private lateinit var server: MockWebServer
    private lateinit var provider: OpenAiCompatProvider

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
        val keys = mockk<ProviderKeys> {
            every { keyFor("test") } returns "test-api-key"
            every { isConfigured("test") } returns true
        }
        provider = OpenAiCompatProvider(
            prefix = "test",
            displayName = "Test Provider",
            baseUrl = server.url("/").toString().removeSuffix("/"),
            providerKeys = keys,
            httpClient = OkHttpClient(),
        )
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    // ── Success ──

    @Test
    fun `listModels returns model IDs from a valid response`() = runBlocking<Unit> {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("""{"data":[{"id":"model-a"},{"id":"model-b"}]}""")
        )
        val models = provider.listModels()
        assertEquals(listOf("model-a", "model-b"), models)
    }

    // ── 401 Unauthorised ──

    @Test
    fun `listModels throws AuthenticationException on 401`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(401))
        assertFailsWith<ProviderCatalogException.AuthenticationException> {
            provider.listModels()
        }
    }

    // ── 429 Rate Limited ──

    @Test
    fun `listModels throws RateLimitedException on 429`() = runBlocking<Unit> {
        server.enqueue(
            MockResponse()
                .setResponseCode(429)
                .addHeader("Retry-After", "30")
        )
        val ex = assertFailsWith<ProviderCatalogException.RateLimitedException> {
            provider.listModels()
        }
        assertEquals(30_000L, ex.retryAfterMs)
    }

    @Test
    fun `listModels throws RateLimitedException without Retry-After header`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(429))
        val ex = assertFailsWith<ProviderCatalogException.RateLimitedException> {
            provider.listModels()
        }
        assertEquals(null, ex.retryAfterMs)
    }

    // ── Other HTTP errors ──

    @Test
    fun `listModels throws NetworkException on 500`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(500))
        val ex = assertFailsWith<ProviderCatalogException.NetworkException> {
            provider.listModels()
        }
        assertEquals(500, ex.statusCode)
    }

    @Test
    fun `listModels throws NetworkException on 403`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(403))
        val ex = assertFailsWith<ProviderCatalogException.NetworkException> {
            provider.listModels()
        }
        assertEquals(403, ex.statusCode)
    }

    // ── Malformed response ──

    @Test
    fun `listModels throws MalformedResponseException on invalid JSON`() = runBlocking<Unit> {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("not-json")
        )
        assertFailsWith<ProviderCatalogException.MalformedResponseException> {
            provider.listModels()
        }
    }

    @Test
    fun `listModels throws MalformedResponseException when data field is missing`() = runBlocking<Unit> {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("""{"notData":[]}""")
        )
        val ex = assertFailsWith<ProviderCatalogException.MalformedResponseException> {
            provider.listModels()
        }
        assertTrue(ex.message?.contains("Missing data[]") == true)
    }

    @Test
    fun `listModels throws MalformedResponseException on empty body`() = runBlocking<Unit> {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("")
        )
        assertFailsWith<ProviderCatalogException.MalformedResponseException> {
            provider.listModels()
        }
    }

    // ── Empty catalog ──

    @Test
    fun `listModels throws EmptyCatalogException when data array is empty`() = runBlocking<Unit> {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("""{"data":[]}""")
        )
        assertFailsWith<ProviderCatalogException.EmptyCatalogException> {
            provider.listModels()
        }
    }

    // ── Network error ──

    @Test
    fun `listModels throws NetworkException when server is unreachable`() = runBlocking<Unit> {
        // Create a provider pointing at a port with no server
        val badProvider = OpenAiCompatProvider(
            prefix = "test",
            displayName = "Test",
            baseUrl = "http://localhost:1",
            providerKeys = mockk {
                every { keyFor("test") } returns "key"
                every { isConfigured("test") } returns true
            },
            httpClient = OkHttpClient.Builder()
                .connectTimeout(1, java.util.concurrent.TimeUnit.SECONDS)
                .build(),
        )
        assertFailsWith<ProviderCatalogException.NetworkException> {
            badProvider.listModels()
        }
    }

    // ── CancellationException rethrown ──

    @Test
    fun `CancellationException is rethrown by listModels`() = runBlocking<Unit> {
        // OpenAiCompatProvider is open, so we can override listModels
        val cancelling = object : OpenAiCompatProvider(
            prefix = "test", displayName = "Test",
            baseUrl = server.url("/").toString().removeSuffix("/"),
            providerKeys = mockk { every { keyFor("test") } returns "key"; every { isConfigured("test") } returns true },
            httpClient = OkHttpClient(),
        ) {
            override suspend fun listModels(): List<String> {
                throw CancellationException("cancelled by test")
            }
        }
        assertFailsWith<CancellationException> {
            cancelling.listModels()
        }
    }

    // ── Default models shortcut ──

    @Test
    fun `listModels returns defaultModels when set without calling network`() = runBlocking<Unit> {
        val prov = OpenAiCompatProvider(
            prefix = "test", displayName = "Test",
            baseUrl = "http://localhost:1",
            providerKeys = mockk { every { keyFor("test") } returns "key"; every { isConfigured("test") } returns true },
            httpClient = OkHttpClient(),
            defaultModels = listOf("built-in-model"),
        )
        assertEquals(listOf("built-in-model"), prov.listModels())
    }
}
