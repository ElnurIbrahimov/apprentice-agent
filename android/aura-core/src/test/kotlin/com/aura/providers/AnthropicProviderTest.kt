package com.aura.providers

import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.SocketPolicy
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * MockWebServer tests for [AnthropicProvider.listModels].
 *
 * Every test uses a local MockWebServer so no real network calls are made.
 * The hardcoded fallback catalog has been removed — only live API responses
 * or typed exceptions are acceptable.
 */
class AnthropicProviderTest {

    private lateinit var server: MockWebServer
    private lateinit var provider: AnthropicProvider

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
        val keys = mockk<ProviderKeys> {
            every { keyFor("anthropic") } returns "test-api-key"
            every { isConfigured("anthropic") } returns true
        }
        provider = AnthropicProvider(
            providerKeys = keys,
            httpClient = OkHttpClient(),
            baseUrl = server.url("/").toString().removeSuffix("/"),
        )
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    // ── Success ──

    @Test
    fun `listModels returns model IDs from valid Anthropic response`() = runBlocking<Unit> {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("""{"data":[{"id":"claude-sonnet-4-20250514","type":"model"},{"id":"claude-haiku-4-20250514","type":"model"}]}""")
        )
        val models = provider.listModels()
        assertEquals(listOf("claude-sonnet-4-20250514", "claude-haiku-4-20250514"), models)
    }

    // ── 401 ──

    @Test
    fun `listModels throws AuthenticationException on 401`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(401))
        assertFailsWith<ProviderCatalogException.AuthenticationException> {
            provider.listModels()
        }
    }

    // ── 429 ──

    @Test
    fun `listModels throws RateLimitedException on 429`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(429))
        assertFailsWith<ProviderCatalogException.RateLimitedException> {
            provider.listModels()
        }
    }

    // ── Other HTTP errors ──

    @Test
    fun `listModels throws NetworkException on 500`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(500))
        assertFailsWith<ProviderCatalogException.NetworkException> {
            provider.listModels()
        }
    }

    // ── Malformed response ──

    @Test
    fun `listModels throws MalformedResponseException on invalid JSON`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(200).setBody("not-json"))
        assertFailsWith<ProviderCatalogException.MalformedResponseException> {
            provider.listModels()
        }
    }

    @Test
    fun `listModels throws MalformedResponseException when data field is missing`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(200).setBody("""{"other":[]}"""))
        val ex = assertFailsWith<ProviderCatalogException.MalformedResponseException> {
            provider.listModels()
        }
        assertTrue(ex.message?.contains("Missing data[]") == true)
    }

    @Test
    fun `listModels throws EmptyCatalogException when data array is empty`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setResponseCode(200).setBody("""{"data":[]}"""))
        assertFailsWith<ProviderCatalogException.EmptyCatalogException> {
            provider.listModels()
        }
    }

    // ── Network error ──

    @Test
    fun `listModels throws NetworkException when server is unreachable`() = runBlocking<Unit> {
        val unavailable = MockWebServer()
        unavailable.start()
        val unavailableBaseUrl = unavailable.url("/").toString().removeSuffix("/")
        unavailable.shutdown()
        val offline = AnthropicProvider(
            providerKeys = mockk {
                every { keyFor("anthropic") } returns "key"
                every { isConfigured("anthropic") } returns true
            },
            httpClient = OkHttpClient.Builder()
                .connectTimeout(1, java.util.concurrent.TimeUnit.SECONDS)
                .build(),
            baseUrl = unavailableBaseUrl,
        )
        assertFailsWith<ProviderCatalogException.NetworkException> {
            offline.listModels()
        }
    }

    // ── CancellationException is rethrown by listModels (via CancellationException catch in impl) ──

    @Test
    fun `CancellationException is rethrown by listModels`() = runBlocking<Unit> {
        // The CancellationException handler is verified by the OpenAiCompatProvider
        // CancellationException test which uses the same pattern. Anthropic uses
        // the same catch blocks. Here we verify the catch exists by covering the
        // full method via MockWebServer + coroutine cancellation.
        server.enqueue(
            MockResponse().setResponseCode(200).setBody("incomplete".repeat(1000))
                .throttleBody(1, 1, java.util.concurrent.TimeUnit.SECONDS)
        )
        val job = launch {
            assertFailsWith<CancellationException> { provider.listModels() }
        }
        // Cancel before the slow response completes
        delay(50)
        job.cancel()
        job.join()
    }

    // ── Legacy contract: prefix, displayName, isConfigured, cancel ──

    @Test
    fun `isConfigured returns true when API key is set`() {
        assertTrue(provider.isConfigured())
    }

    @Test
    fun `isConfigured returns false when API key is not set`() {
        val unconfigured = AnthropicProvider(
            providerKeys = mockk {
                every { keyFor("anthropic") } returns null
                every { isConfigured("anthropic") } returns false
            },
            httpClient = OkHttpClient(),
        )
        assertFalse(unconfigured.isConfigured())
    }

    @Test
    fun `prefix and displayName are correct`() {
        assertEquals("anthropic", provider.prefix)
        assertEquals("Anthropic", provider.displayName)
    }

    @Test
    fun `cancel does not throw when no active call`() = runBlocking<Unit> {
        provider.cancel()
    }

    @Test
    fun `cancelling chat collection cancels the active HTTP call promptly`() = runBlocking<Unit> {
        server.enqueue(MockResponse().setSocketPolicy(SocketPolicy.NO_RESPONSE))
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val job = scope.launch {
            runCatching {
                provider.chat(
                    model = "claude-test",
                    messages = listOf(ProviderMessage(ProviderMessage.Role.user, "hello")),
                ).collect()
            }
        }

        assertTrue(server.takeRequest(2, TimeUnit.SECONDS) != null, "chat request did not start")
        job.cancel()
        val stoppedPromptly = withTimeoutOrNull(1_000L) {
            job.join()
            true
        } ?: false
        if (!stoppedPromptly) {
            provider.cancel()
            withTimeout(2_000L) { job.join() }
        }
        scope.cancel()

        assertTrue(stoppedPromptly, "cancelling the flow did not cancel the Anthropic call")
    }
}
