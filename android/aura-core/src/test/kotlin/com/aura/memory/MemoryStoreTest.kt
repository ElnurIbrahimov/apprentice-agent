package com.aura.memory

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class MemoryStoreTest {

    @Test
    fun `Embedder produces normalized vector of correct dim`() {
        val emb = Embedder(384)
        val v = emb.embed("I prefer oat milk in my coffee")
        assertEquals(384, v.size)
        var norm = 0f
        for (x in v) norm += x * x
        assertTrue(kotlin.math.abs(kotlin.math.sqrt(norm) - 1f) < 0.01f, "vector should be normalized")
    }

    @Test
    fun `Embedder produces different vectors for different texts`() {
        val emb = Embedder(384)
        val a = emb.embed("hello world")
        val b = emb.embed("completely different topic")
        var same = 0
        for (i in a.indices) if (a[i] == b[i]) same++
        assertTrue(same < 384, "vectors should differ on at least some dims")
    }

    @Test
    fun `VectorIndex returns empty for empty input`() {
        val idx = VectorIndex(384)
        val hits = idx.search(FloatArray(384), emptyList())
        assertTrue(hits.isEmpty())
    }

    @Test
    fun `VectorIndex ranks closer vectors higher`() {
        val idx = VectorIndex(384)
        val emb = Embedder(384)
        val query = emb.embed("user likes coffee in the morning")
        val good = emb.embed("the user drinks coffee every morning")
        val bad = emb.embed("quantum chromodynamics experiments at CERN")
        val hits = idx.search(query, listOf("good" to good, "bad" to bad), topK = 2)
        assertEquals("good", hits.first().memoryId)
    }

    @Test
    fun `WriteGate classifies preferences correctly`() {
        val gate = WriteGate()
        val d = gate.evaluate("I prefer dark mode everywhere", "user")
        assertTrue(d.shouldStore)
        assertEquals("preference", d.category)
        assertTrue(d.importance >= 0.7f)
    }

    @Test
    fun `WriteGate rejects empty or short content`() {
        val gate = WriteGate()
        assertEquals(false, gate.evaluate("", "user").shouldStore)
        assertEquals(false, gate.evaluate("hi", "user").shouldStore)
    }

    @Test
    fun `FadeMem decays over simulated time`() {
        val now = System.currentTimeMillis()
        val created = now - 30L * 86_400_000  // 30 days ago
        val accessed = now - 30L * 86_400_000  // never touched
        val score = FadeMem.compute(created, accessed, now)
        // 30 days is 2.14 half-lives (14d half-life) → ~0.23
        assertTrue(score < 0.3f, "30-day-old untouched memory should decay to ~0.23, got $score")
        assertTrue(score > 0.1f, "30-day-old untouched memory should not be forgotten yet, got $score")
    }

    @Test
    fun `FadeMem keeps freshly accessed memories alive`() {
        val now = System.currentTimeMillis()
        val created = now - 365L * 86_400_000  // 1 year old
        val accessed = now  // just touched
        val score = FadeMem.compute(created, accessed, now)
        // Touched now → decay ~1.0, but floor kicks in (365/730 = 0.5)
        assertTrue(score > 0.4f, "freshly-touched 1-year-old memory should survive, got $score")
    }

    @Test
    fun `Embedder byte roundtrip preserves values`() {
        val emb = Embedder(384)
        val v = emb.embed("test roundtrip")
        val bytes = Embedder.toBytes(v)
        val v2 = Embedder.fromBytes(bytes)
        assertEquals(v.size, v2.size)
        for (i in v.indices) {
            assertEquals(v[i], v2[i], 1e-6f, "mismatch at index $i")
        }
    }
}
