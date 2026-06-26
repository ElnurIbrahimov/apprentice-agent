package com.aura.memory

import com.aura.memory.MemoryDao
import com.aura.memory.MemoryEntity
import com.aura.memory.MemoryStore
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.coVerifyOrder
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Regression tests for the [MemoryStore.query] touch behavior.
 *
 * The original bug: `query()` returned memory hits but never called
 * `touch()` on them, so the access count and `accessedAt` timestamp
 * never moved, and the FadeMem decay score was always based on
 * `createdAt` only. The "frequently-recalled memories decay slower"
 * promise was structurally broken.
 */
class MemoryStoreTouchTest {

    private fun mem(id: String, content: String) = MemoryEntity(
        id = id,
        content = content,
        source = "user",
        category = "preference",
    )

    @Test
    fun `query touches each returned hit`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        coEvery { dao.searchByText("%dark%", any()) } returns listOf(
            mem("m1", "user prefers dark mode"),
            mem("m2", "user prefers dark theme"),
        )
        val embedder = mockk<Embedder>(relaxed = true)
        coEvery { embedder.embed(any()) } returns FloatArray(384) { 0f }
        val vectorIndex = mockk<VectorIndex>(relaxed = true)
        coEvery { vectorIndex.search(any(), any(), any()) } returns emptyList()

        val writeGate = mockk<WriteGate>(relaxed = true)
        val store = MemoryStore(dao, embedder, vectorIndex, writeGate)

        val results = store.query("dark", limit = 5)
        assertEquals(2, results.size, "query should return both hits")

        // The store's `touch()` ultimately delegates to `dao.touch(id)`. The
        // touch must fire for every returned hit, in any order. The DAO's
        // touch(id, now=...) takes a default now arg; use any() to match.
        coVerify(exactly = 1) { dao.touch("m1", any()) }
        coVerify(exactly = 1) { dao.touch("m2", any()) }
    }

    @Test
    fun `query does not touch on empty result`() = runTest {
        val dao = mockk<MemoryDao>(relaxed = true)
        coEvery { dao.searchByText("%nothing%", any()) } returns emptyList()
        val store = MemoryStore(
            dao,
            mockk<Embedder>(relaxed = true),
            mockk<VectorIndex>(relaxed = true),
            mockk<WriteGate>(relaxed = true),
        )

        val results = store.query("nothing", limit = 5)
        assertEquals(0, results.size)
        coVerify(exactly = 0) { dao.touch(any(), any()) }
    }

    @Test
    fun `listByCategory does not touch returned memories`() = runTest {
        // Category browsing is metadata, not a recall. Touching here would
        // mean simply scrolling through Memory tab counts as a recall.
        val dao = mockk<MemoryDao>(relaxed = true)
        coEvery { dao.byCategory("preference", any()) } returns listOf(
            mem("m1", "user prefers dark mode"),
        )
        val store = MemoryStore(
            dao,
            mockk<Embedder>(relaxed = true),
            mockk<VectorIndex>(relaxed = true),
            mockk<WriteGate>(relaxed = true),
        )

        store.listByCategory("preference", 10)
        coVerify(exactly = 0) { dao.touch(any(), any()) }
    }

    @Test
    fun `listByCategory returns the dao result unchanged`() = runTest {
        val hits = listOf(mem("a", "x"), mem("b", "y"))
        val dao = mockk<MemoryDao>(relaxed = true)
        coEvery { dao.byCategory("fact", any()) } returns hits
        val store = MemoryStore(
            dao,
            mockk<Embedder>(relaxed = true),
            mockk<VectorIndex>(relaxed = true),
            mockk<WriteGate>(relaxed = true),
        )

        val results = store.listByCategory("fact", 10)
        assertEquals(hits, results)
    }
}
