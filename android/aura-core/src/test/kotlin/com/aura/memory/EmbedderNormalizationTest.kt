package com.aura.memory

import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Regression test for the Embedder. The previous implementation added a
 * positional cosine offset AFTER normalization, which meant the returned
 * vector was no longer unit-normalized and cosine similarity against
 * it was silently bounded below 1.0 for identical inputs.
 */
class EmbedderNormalizationTest {

    @Test
    fun `embedder returns unit-normalized vector`() {
        val emb = Embedder(384)
        for (text in listOf("", "hello", "a longer sample sentence with more tokens", "x")) {
            val v = emb.embed(text)
            assertEquals(384, v.size, "vector should be 384-dim")
            var norm = 0f
            for (x in v) norm += x * x
            val len = sqrt(norm)
            if (text.isBlank()) {
                // Empty input legitimately returns the zero vector.
                assertEquals(0f, len, 0.0001f, "empty input should be all zeros")
            } else {
                assertTrue(abs(len - 1f) < 0.001f, "non-empty input should be unit-normalized, got length=$len for text='$text'")
            }
        }
    }

    @Test
    fun `identical inputs produce identical vectors`() {
        val emb = Embedder(384)
        val a = emb.embed("remember that I prefer dark mode")
        val b = emb.embed("remember that I prefer dark mode")
        for (i in a.indices) {
            assertEquals(a[i], b[i], 0.0001f, "vector at index $i differs between identical inputs")
        }
    }

    @Test
    fun `cosine similarity between identical inputs is one`() {
        val emb = Embedder(384)
        val v = emb.embed("user drinks coffee every morning")
        var dot = 0f
        for (i in v.indices) dot += v[i] * v[i]
        // v is unit-normalized, so dot product = 1.0
        assertTrue(abs(dot - 1f) < 0.001f, "cosine similarity of identical vector should be 1, got $dot")
    }
}
