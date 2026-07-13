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
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Additional MockWebServer tests for [OpenAiCompatProvider.listModels] that
 * verify the structured exception contract: no secret/error body leakage.
 */
class OpenAiCompatProviderMockWebServerTest {

    private lateinit var server: MockWebServer
    private lateinit var provider: OpenAiCompatProvider

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
        val keys = mockk<ProviderKeys> {
            every { keyFor("test") } returns "sk-abc123-secret"
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

    @Test
    fun `error messages never include API key`() = runBlocking {
        // 401 body sometimes contains the word "key" or the actual key
        server.enqueue(
            MockResponse()
                .setResponseCode(401)
                .setBody("""{"error":{"message":"Invalid API key: sk-abc123-secret"}}""")
        )
        val ex = assertFailsWith<ProviderCatalogException.AuthenticationException> {
            provider.listModels()
        }
        // Message must NOT contain the raw API key
        assertTrue(ex.message?.contains("sk-abc123-secret") != true,
            "Exception message must not leak API key")
    }

    @Test
    fun `error messages never include raw response body`() = runBlocking {
        // 500 with a verbose body
        server.enqueue(
            MockResponse()
                .setResponseCode(500)
                .setBody("""{"error":"Internal server error: connection pool exhausted","details":["stacktrace line 1","stacktrace line 2"]}""")
        )
        val ex = assertFailsWith<ProviderCatalogException.NetworkException> {
            provider.listModels()
        }
        // Must not contain raw body fragments
        assertTrue(ex.message?.contains("stacktrace") != true,
            "Exception message must not include raw response body")
        assertTrue(ex.message?.contains("connection pool") != true,
            "Exception message must not include raw response body")
    }
}
