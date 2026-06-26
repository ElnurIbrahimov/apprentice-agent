package com.aura.memory

import com.aura.providers.ProviderKeys
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.test.runTest
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for [CloudEmbedder] — cloud embedding with local fallback.
 *
 * Uses mocked [OkHttpClient] and [ProviderKeys] so no real network is
 * required.
 */
class CloudEmbedderTest {

    private val jsonMediaType = "application/json".toMediaType()

    /** A realistic 384-dim embedding response from Ollama Cloud. */
    private val sampleEmbedding = FloatArray(384) { (it % 100) / 100f + 0.5f }

    // ─── helpers ──────────────────────────────────────────────────────────

    private fun mockHttp(responseBody: String): OkHttpClient {
        val response = Response.Builder()
            .request(Request.Builder().url("https://api.ollama.com/api/embeddings").build())
            .protocol(Protocol.HTTP_1_1)
            .code(200)
            .message("OK")
            .body(responseBody.toResponseBody(jsonMediaType))
            .build()

        val call = mockk<Call> {
            every { execute() } returns response
        }

        return mockk {
            every { newCall(any()) } returns call
        }
    }

    private fun mockHttpError(code: Int): OkHttpClient {
        val response = Response.Builder()
            .request(Request.Builder().url("https://api.ollama.com/api/embeddings").build())
            .protocol(Protocol.HTTP_1_1)
            .code(code)
            .message("Error")
            .body("{\"error\":\"failed\"}".toResponseBody(jsonMediaType))
            .build()

        val call = mockk<Call> {
            every { execute() } returns response
        }

        return mockk {
            every { newCall(any()) } returns call
        }
    }

    private fun mockHttpException(): OkHttpClient {
        val call = mockk<Call> {
            every { execute() } throws RuntimeException("Network error")
        }

        return mockk {
            every { newCall(any()) } returns call
        }
    }

    private fun providerKeys(apiKey: String?, model: String = "nomic-embed-text"): ProviderKeys {
        val pk = mockk<ProviderKeys>(relaxed = true)
        every { pk.keyFor("ollama") } returns apiKey
        every { pk.embeddingModel } returns model
        return pk
    }

    /** Elementwise assert on two FloatArrays with a tolerance. */
    private fun assertFloatArrayEquals(expected: FloatArray, actual: FloatArray, delta: Float = 0.0f) {
        assertEquals(expected.size, actual.size, "array size mismatch")
        for (i in expected.indices) {
            assertEquals(expected[i], actual[i], delta, "index $i")
        }
    }

    // ─── tests ────────────────────────────────────────────────────────────

    @Test
    fun `cloud embedding returns float array`() = runTest {
        val embedBody =
            """{"embedding":[${
                sampleEmbedding.joinToString(",") { it.toString() }
            }]}"""
        val httpClient = mockHttp(embedBody)
        val keys = providerKeys(apiKey = "sk-test-key")
        val local = mockk<LocalEmbedder>(relaxed = true)

        val sut = CloudEmbedder(local, keys, httpClient)
        val result = sut.embed("test text")

        assertEquals(384, result.size)
        assertFloatArrayEquals(sampleEmbedding, result, 0.001f)
        // Local fallback must NOT have been called
        coVerify(exactly = 0) { local.embed(any()) }
    }

    @Test
    fun `fallback to local when api key is blank`() = runTest {
        val httpClient = mockHttp("""{"embedding":[]}""")
        val keys = providerKeys(apiKey = null)
        val local = mockk<LocalEmbedder>(relaxed = true)
        coEvery { local.embed(any()) } returns FloatArray(384) { 1f }

        val sut = CloudEmbedder(local, keys, httpClient)
        val result = sut.embed("test text")

        assertEquals(384, result.size)
        assertEquals(1f, result[0])
        coVerify { local.embed("test text") }
    }

    @Test
    fun `fallback to local on http error`() = runTest {
        val httpClient = mockHttpError(500)
        val keys = providerKeys(apiKey = "sk-test-key")
        val local = mockk<LocalEmbedder>(relaxed = true)
        coEvery { local.embed(any()) } returns FloatArray(384) { 2f }

        val sut = CloudEmbedder(local, keys, httpClient)
        val result = sut.embed("test text")

        assertEquals(2f, result[0])
        coVerify { local.embed("test text") }
    }

    @Test
    fun `fallback to local on network exception`() = runTest {
        val httpClient = mockHttpException()
        val keys = providerKeys(apiKey = "sk-test-key")
        val local = mockk<LocalEmbedder>(relaxed = true)
        coEvery { local.embed(any()) } returns FloatArray(384) { 3f }

        val sut = CloudEmbedder(local, keys, httpClient)
        val result = sut.embed("test text")

        assertEquals(3f, result[0])
        coVerify { local.embed("test text") }
    }

    @Test
    fun `cache deduplicates identical text`() = runTest {
        val embedBody =
            """{"embedding":[${
                sampleEmbedding.joinToString(",") { it.toString() }
            }]}"""
        val httpClient = mockHttp(embedBody)
        val keys = providerKeys(apiKey = "sk-test-key")
        val local = mockk<LocalEmbedder>(relaxed = true)

        val sut = CloudEmbedder(local, keys, httpClient)

        // First call → goes to cloud
        val first = sut.embed("duplicate text")
        // Second call → should hit cache
        val second = sut.embed("duplicate text")

        assertFloatArrayEquals(first, second)
        // newCall must have been called exactly once (the second call hits the cache)
        verify(exactly = 1) { httpClient.newCall(any()) }
        coVerify(exactly = 0) { local.embed(any()) }
    }

    @Test
    fun `different texts produce different cache entries`() = runTest {
        val embedBodyA =
            """{"embedding":[${
                sampleEmbedding.joinToString(",") { it.toString() }
            }]}"""
        val embedBodyB =
            """{"embedding":[${
                FloatArray(384) { (it % 100 + 50) / 100f + 0.5f }.joinToString(",") { it.toString() }
            }]}"""

        // Return different responses for different request bodies by checking
        // the prompt field in a simple way — we mock sequential calls instead.
        val callA = mockk<Call> {
            every { execute() } returns Response.Builder()
                .request(Request.Builder().url("https://api.ollama.com/api/embeddings").build())
                .protocol(Protocol.HTTP_1_1)
                .code(200).message("OK")
                .body(embedBodyA.toResponseBody(jsonMediaType))
                .build()
        }
        val callB = mockk<Call> {
            every { execute() } returns Response.Builder()
                .request(Request.Builder().url("https://api.ollama.com/api/embeddings").build())
                .protocol(Protocol.HTTP_1_1)
                .code(200).message("OK")
                .body(embedBodyB.toResponseBody(jsonMediaType))
                .build()
        }

        val httpClient = mockk<OkHttpClient> {
            every { newCall(any()) } returnsMany listOf(callA, callB)
        }
        val keys = providerKeys(apiKey = "sk-test-key")
        val local = mockk<LocalEmbedder>(relaxed = true)

        val sut = CloudEmbedder(local, keys, httpClient)

        val a = sut.embed("text A")
        val b = sut.embed("text B")

        // Should be different vectors
        var diffs = 0
        for (i in a.indices) {
            if (a[i] != b[i]) diffs++
        }
        assertTrue(diffs > 0, "Different texts should produce different embeddings")
        verify(exactly = 2) { httpClient.newCall(any()) }
    }
}
