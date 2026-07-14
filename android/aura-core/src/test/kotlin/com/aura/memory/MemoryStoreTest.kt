package com.aura.memory

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class MemoryStoreTest {
    private val memoryEditDao = mockk<MemoryEditDao>(relaxed = true)

    @Test
    fun `local embedder produces normalized vector of correct dim`() = runTest {
        val emb = LocalEmbedder(384)
        val v = emb.embed("I prefer oat milk in my coffee")
        assertEquals(384, v.size)
        // Check normalization: |v| ≈ 1
        val norm = kotlin.math.sqrt(v.fold(0f) { acc, x -> acc + x * x })
        assertTrue(kotlin.math.abs(norm - 1f) < 0.01f, "vector should be normalized, |v|=$norm")
    }

    @Test
    fun `local embedder produces different vectors for different texts`() = runTest {
        val emb = LocalEmbedder(384)
        val v1 = emb.embed("I prefer oat milk in my coffee")
        val v2 = emb.embed("The weather is nice today")
        // Vectors should not be identical
        var same = true
        for (i in v1.indices) {
            if (v1[i] != v2[i]) { same = false; break }
        }
        assertTrue(!same, "different texts should produce different vectors")
    }

    @Test
    fun `VectorIndex returns empty for empty input`() {
        val idx = VectorIndex(384)
        val results = idx.search(FloatArray(384), emptyList(), 5)
        assertEquals(0, results.size)
    }

    @Test
    fun `VectorIndex ranks closer vectors higher`() = runTest {
        val idx = VectorIndex(384)
        val emb = LocalEmbedder(384)
        val q = emb.embed("I love Kotlin")
        val v1 = emb.embed("I love Kotlin coroutines")
        val v2 = emb.embed("The weather is nice today")
        val results = idx.search(q, listOf("v1" to v1, "v2" to v2), 2)
        // At least one result should come back, and the first should
        // be the closer vector (v1). Unrelated text may fall below the
        // 0.05 cosine threshold, so we only assert on ordering.
        assertTrue(results.isNotEmpty(), "should have at least one result")
        assertEquals("v1", results[0].memoryId, "closer vector should be ranked first")
    }

    @Test
    fun `WriteGate classifies preferences correctly`() {
        val gate = WriteGate()
        val d = gate.evaluate("I prefer dark mode", "user")
        assertTrue(d.shouldStore)
        assertEquals("preference", d.category)
        assertEquals(0.8f, d.importance)
    }

    @Test
    fun `WriteGate rejects empty or short content`() {
        val gate = WriteGate()
        assertFalse(gate.evaluate("", "user").shouldStore)
        assertFalse(gate.evaluate("hi", "user").shouldStore)
        assertFalse(gate.evaluate("   ", "user").shouldStore)
    }

    @Test
    fun `FadeMem decays over simulated time`() {
        // 14 days without access → half-life → 0.5
        val now = 1_700_000_000_000L
        val createdAt = now - 14L * 86_400_000L
        val accessedAt = now - 14L * 86_400_000L
        val score = FadeMem.compute(createdAt, accessedAt, now)
        assertTrue(score < 0.55f, "14 days should be near half-life: $score")
        assertTrue(score > 0.35f, "14 days should be near half-life: $score")
    }

    @Test
    fun `FadeMem keeps freshly accessed memories alive`() {
        val now = 1_700_000_000_000L
        val createdAt = now - 60L * 86_400_000L // 60 days old
        val accessedAt = now // accessed just now
        val score = FadeMem.compute(createdAt, accessedAt, now)
        assertTrue(score > 0.9f, "freshly accessed memory should have high score: $score")
    }

    @Test
    fun `Embedder byte roundtrip preserves values`() = runTest {
        val original = FloatArray(384) { it.toFloat() / 384f }
        val bytes = Embedder.toBytes(original)
        val restored = Embedder.fromBytes(bytes)
        assertEquals(384, restored.size)
        for (i in original.indices) {
            assertTrue(kotlin.math.abs(original[i] - restored[i]) < 0.001f,
                "roundtrip mismatch at $i: ${original[i]} vs ${restored[i]}")
        }
    }

    @Test
    fun `update changes content and category, invalidates embedding`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val store = MemoryStore(
            dao,
            FakeEmbedder(384),
            VectorIndex(384),
            WriteGate(),
            memoryEditDao = memoryEditDao,
        )
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
        assertEquals("m1", captured.captured.id)
        assertEquals("user", captured.captured.source)
        assertEquals(1L, captured.captured.createdAt)
    }

    @Test
    fun `update saves importance and tags`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val store = MemoryStore(dao, FakeEmbedder(384), VectorIndex(384), WriteGate(), memoryEditDao)
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
            memoryEditDao = memoryEditDao,
        )
        coEvery { dao.getById("missing") } returns null
        store.update("missing", "x", "fact")
        coVerify(exactly = 0) { dao.update(any()) }
    }

    @Test
    fun `rebuildEmbeddings re-embeds only rows with null embedding`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val embedder = FakeEmbedder(384)
        val store = MemoryStore(
            dao,
            embedder,
            VectorIndex(384),
            WriteGate(),
            memoryEditDao = memoryEditDao,
        )
        val p1 = MemoryEntity(id = "p1", content = "a", source = "user", category = "fact", embedding = null)
        val p2 = MemoryEntity(id = "p2", content = "b", source = "user", category = "fact", embedding = null)
        val p3 = MemoryEntity(id = "p3", content = "c", source = "user", category = "fact", embedding = ByteArray(384 * 4))
        coEvery { dao.allForExport() } returns listOf(p1, p2, p3)
        coEvery { dao.update(any()) } answers { Unit }

        val rebuilt = store.rebuildEmbeddings()

        assertEquals(2, rebuilt)
        coVerify(exactly = 2) { dao.update(any()) }
    }

    @Test
    fun `rebuildEmbeddings is a no-op when everything is already embedded`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val store = MemoryStore(
            dao,
            FakeEmbedder(384),
            VectorIndex(384),
            WriteGate(),
            memoryEditDao = memoryEditDao,
        )
        val p = MemoryEntity(id = "p1", content = "a", source = "user", category = "fact", embedding = ByteArray(384 * 4))
        coEvery { dao.allForExport() } returns listOf(p)

        val rebuilt = store.rebuildEmbeddings()

        assertEquals(0, rebuilt)
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
            memoryEditDao = memoryEditDao,
        )
        coEvery { dao.allForExport() } returns emptyList()

        val rebuilt = store.rebuildEmbeddings()

        assertEquals(0, rebuilt)
    }

    @Test
    fun `rebuildEmbeddings continues past a per-row failure`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val embedder = mockk<Embedder>(relaxed = true)
        val store = MemoryStore(
            dao,
            embedder,
            VectorIndex(384),
            WriteGate(),
            memoryEditDao = memoryEditDao,
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

    @Test
    fun `storeIfAbsent inserts an exact marker only once`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val store = MemoryStore(dao, FakeEmbedder(384), VectorIndex(384), WriteGate(), memoryEditDao)
        val content = "This user started using Aura. They went through the onboarding."
        var exists = false
        coEvery { dao.existsByContent(content) } answers { if (exists) 1 else 0 }
        coEvery { dao.insert(any()) } answers { exists = true }

        val first = store.storeIfAbsent(
            content = content,
            source = "system",
            category = "episode",
            importance = 0.8f,
        )
        val second = store.storeIfAbsent(
            content = content,
            source = "system",
            category = "episode",
            importance = 0.8f,
        )

        assertNotNull(first)
        assertNull(second)
        coVerify(exactly = 1) { dao.insert(any()) }
    }

    @Test
    fun `storeIfAbsent skips embedding when marker already exists`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val embedder = mockk<Embedder>(relaxed = true)
        val store = MemoryStore(dao, embedder, VectorIndex(384), WriteGate(), memoryEditDao)
        coEvery { dao.existsByContent("marker") } returns 1

        val result = store.storeIfAbsent("marker", "system", "episode", 0.8f)

        assertNull(result)
        coVerify(exactly = 0) { embedder.embed(any()) }
        coVerify(exactly = 0) { dao.insert(any()) }
    }

    @Test
    fun `maybeStore deduplicates identical content`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val embedder = FakeEmbedder(384)
        val store = MemoryStore(dao, embedder, VectorIndex(384), WriteGate(), memoryEditDao)
        coEvery { dao.existsByContent("I prefer dark mode") } returns 1
        val result = store.maybeStore("I prefer dark mode", "user")
        assertNull(result)
        coVerify(exactly = 0) { dao.insert(any()) }
    }

    @Test
    fun `maybeStore stores new content when no duplicate exists`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val embedder = FakeEmbedder(384)
        val store = MemoryStore(dao, embedder, VectorIndex(384), WriteGate(), memoryEditDao)
        coEvery { dao.existsByContent("I prefer light mode") } returns 0
        val result = store.maybeStore("I prefer light mode", "user")
        assertNotNull(result)
        coVerify(exactly = 1) { dao.insert(any()) }
    }

    @Test
    fun `store persists exact conversation and turn provenance`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val captured = slot<MemoryEntity>()
        coEvery { dao.insert(capture(captured)) } answers { Unit }
        val store = MemoryStore(dao, FakeEmbedder(384), VectorIndex(384), WriteGate(), memoryEditDao)
        val provenance = com.aura.provenance.ConversationProvenance("conv-42", 1_700_000_123L)

        store.store(
            content = "The user is writing a solar-punk novel",
            source = "user",
            category = "project",
            importance = 0.9f,
            provenance = provenance,
        )

        assertEquals("conv-42", captured.captured.sourceConversationId)
        assertEquals(1_700_000_123L, captured.captured.sourceTurnTimestamp)
    }

    @Test
    fun `restore reinserts exact memory and its audit trail`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        val editDao = mockk<MemoryEditDao>(relaxed = true)
        val store = MemoryStore(dao, FakeEmbedder(384), VectorIndex(384), WriteGate(), editDao)
        val memory = MemoryEntity(
            id = "memory-original",
            content = "Original fact",
            source = "user",
            category = "fact",
            createdAt = 123L,
            accessedAt = 456L,
            accessCount = 7,
            sourceConversationId = "conv-1",
            sourceTurnTimestamp = 789L,
        )
        val edits = listOf(
            MemoryEditEntity(
                id = 41,
                memoryId = memory.id,
                oldContent = "Old fact",
                newContent = memory.content,
                oldCategory = "fact",
                newCategory = "fact",
                editedAt = 700L,
            ),
        )

        store.restore(memory, edits)

        coVerify(exactly = 1) { dao.insert(memory) }
        coVerify(exactly = 1) { editDao.insertAll(edits) }
    }
}