package com.aura.memory

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class MemoryStoreTest {

    @Test
    fun `local embedder produces normalized vector of correct dim`() = runTest {
        val emb = LocalEmbedder(384)
        val v = emb.embed("I prefer oat milk in my coffee")
        assertEquals(384, v.size)
        var norm = 0f
        for (x in v) norm += x * x
        assertTrue(kotlin.math.abs(kotlin.math.sqrt(norm) - 1f) < 0.01f, "vector should be normalized")
    }

    @Test
    fun `local embedder produces different vectors for different texts`() = runTest {
        val emb = LocalEmbedder(384)
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
    fun `VectorIndex ranks closer vectors higher`() = runTest {
        val idx = VectorIndex(384)
        val emb = LocalEmbedder(384)
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
    fun `Embedder byte roundtrip preserves values`() = runTest {
        val emb = FakeEmbedder(384)
        val v = emb.embed("test roundtrip")
        val bytes = Embedder.toBytes(v)
        val v2 = Embedder.fromBytes(bytes)
        assertEquals(v.size, v2.size)
        for (i in v.indices) {
            assertEquals(v[i], v2[i], 1e-6f, "mismatch at index $i")
        }
    }

    @Test
    fun `update changes content and category, invalidates embedding`() = runTest {
        // The store gets a local mock; the class field `memoryDao` is
        // not used here. We re-bind via the captured slot pattern.
        val dao = mockk<MemoryDao>(relaxed = true)
        val store = MemoryStore(
            dao,
            FakeEmbedder(384),
            VectorIndex(384),
            WriteGate(),
        )
        // Real implementation: capture the entity written.
        val original = MemoryEntity(
            id = "m1", content = "old", source = "user", category = "fact",
            importance = 0.5f, embedding = byteArrayOf(1, 2, 3, 4),
            createdAt = 1L, accessedAt = 1L, accessCount = 0, decayScore = 1.0f,
            tags = "", metadata = "",
        )
        coEvery { dao.getById("m1") } returns original
        val captured = slot<MemoryEntity>()
        coEvery { dao.update(capture(captured)) } answers { Unit }

        store.update("m1", "new content", "preference")

        assertEquals("new content", captured.captured.content)
        assertEquals("preference", captured.captured.category)
        assertNull(captured.captured.embedding, "embedding should be invalidated so the next recall re-embeds")
        assertTrue(captured.captured.accessedAt > original.accessedAt, "accessedAt should be bumped")
        // Untouched fields preserved.
        assertEquals("m1", captured.captured.id)
        assertEquals("user", captured.captured.source)
        assertEquals(1L, captured.captured.createdAt)
    }

    @Test
    fun `update saves importance and tags`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val store = MemoryStore(dao, FakeEmbedder(384), VectorIndex(384), WriteGate())
        val original = MemoryEntity(
            id = "m1", content = "old", source = "user", category = "fact",
            importance = 0.3f, embedding = ByteArray(384 * 4),
            createdAt = 1L, accessedAt = 1L, accessCount = 0, decayScore = 1.0f,
            tags = "old", metadata = "",
        )
        coEvery { dao.getById("m1") } returns original
        val captured = slot<MemoryEntity>()
        coEvery { dao.update(capture(captured)) } answers { Unit }

        store.update("m1", "new content", "preference", 0.9f, "work,urgent")

        assertEquals(0.9f, captured.captured.importance)
        assertEquals("work,urgent", captured.captured.tags)
    }

    @Test
    fun `update is a no-op when the id does not exist`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val store = MemoryStore(
            dao,
            FakeEmbedder(384),
            VectorIndex(384),
            WriteGate(),
        )
        coEvery { dao.getById("missing") } returns null
        store.update("missing", "x", "fact")
        coVerify(exactly = 0) { dao.update(any()) }
    }

    @Test
    fun `rebuildEmbeddings re-embeds only rows with null embedding`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val store = MemoryStore(
            dao,
            FakeEmbedder(384),
            VectorIndex(384),
            WriteGate(),
        )
        // Two rows have null embeddings (fresh from a backup import);
        // one already has its embedding. The rebuild should touch only
        // the two null ones.
        val pending1 = MemoryEntity(id = "p1", content = "user likes coffee", source = "user", category = "preference", embedding = null)
        val pending2 = MemoryEntity(id = "p2", content = "user lives in Baku", source = "user", category = "fact", embedding = null)
        val ready = MemoryEntity(id = "r1", content = "user prefers dark mode", source = "user", category = "preference", embedding = byteArrayOf(1, 2, 3, 4))
        coEvery { dao.allForExport() } returns listOf(pending1, pending2, ready)

        val updated = mutableListOf<MemoryEntity>()
        coEvery { dao.update(capture(updated)) } answers { Unit }

        val rebuilt = store.rebuildEmbeddings()

        assertEquals(2, rebuilt, "should re-embed the two null rows")
        assertEquals(2, updated.size, "should write exactly two updates")
        assertEquals(setOf("p1", "p2"), updated.map { it.id }.toSet())
        // All updated rows have a non-null embedding now.
        assertTrue(updated.all { it.embedding != null }, "all updated rows should have a fresh embedding")
    }

    @Test
    fun `rebuildEmbeddings is a no-op when everything is already embedded`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val store = MemoryStore(
            dao,
            FakeEmbedder(384),
            VectorIndex(384),
            WriteGate(),
        )
        coEvery { dao.allForExport() } returns listOf(
            MemoryEntity(id = "r1", content = "a", source = "user", category = "fact", embedding = byteArrayOf(1, 2, 3, 4)),
            MemoryEntity(id = "r2", content = "b", source = "user", category = "fact", embedding = byteArrayOf(5, 6, 7, 8)),
        )

        val rebuilt = store.rebuildEmbeddings()

        assertEquals(0, rebuilt, "no rows to rebuild")
        coVerify(exactly = 0) { dao.update(any()) }
    }

    @Test
    fun `rebuildEmbeddings is a no-op on an empty table`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val store = MemoryStore(
            dao,
            FakeEmbedder(384),
            VectorIndex(384),
            WriteGate(),
        )
        coEvery { dao.allForExport() } returns emptyList()

        val rebuilt = store.rebuildEmbeddings()

        assertEquals(0, rebuilt)
        coVerify(exactly = 0) { dao.update(any()) }
    }

    @Test
    fun `rebuildEmbeddings continues past a per-row failure`() = runTest {
        // If one row's embedder call throws, the next row should still
        // be re-embedded. The count returned reflects only the
        // successes.
        val dao = mockk<MemoryDao>(relaxed = true)
        val embedder = mockk<Embedder>()
        val store = MemoryStore(
            dao,
            embedder,
            VectorIndex(384),
            WriteGate(),
        )
        val p1 = MemoryEntity(id = "p1", content = "a", source = "user", category = "fact", embedding = null)
        val p2 = MemoryEntity(id = "p2", content = "b", source = "user", category = "fact", embedding = null)
        val p3 = MemoryEntity(id = "p3", content = "c", source = "user", category = "fact", embedding = null)
        coEvery { dao.allForExport() } returns listOf(p1, p2, p3)
        coEvery { embedder.embed("a") } throws RuntimeException("network blip")
        coEvery { embedder.embed("b") } returns FloatArray(384) { 0f }
        coEvery { embedder.embed("c") } returns FloatArray(384) { 0f }
        coEvery { dao.update(any()) } answers { Unit }

        val rebuilt = store.rebuildEmbeddings()

        assertEquals(2, rebuilt, "two of three should succeed; the failure is contained")
        coVerify(exactly = 2) { dao.update(any()) }
    }
}
