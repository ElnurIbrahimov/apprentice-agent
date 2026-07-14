package com.aura.documents

import com.aura.memory.MemoryStore
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DocumentRepositoryTest {
    private val dao = mockk<DocumentDao>(relaxed = true)
    private val memoryStore = mockk<MemoryStore>(relaxed = true)
    private val repository = DocumentRepository(dao, memoryStore)

    @Test
    fun `import stores searchable chunks and document metadata`() = runTest {
        val inserted = slot<DocumentEntity>()
        coEvery { dao.insert(capture(inserted)) } returns Unit
        val text = (1..500).joinToString(" ") { "worldbuilding-$it" }

        val result = repository.import(
            id = "sha256-id",
            name = "world-bible.pdf",
            mimeType = "application/pdf",
            sourceUri = "content://docs/world-bible",
            text = text,
        )

        assertTrue(result.chunkCount > 1)
        assertEquals("sha256-id", inserted.captured.id)
        assertEquals(result.chunkCount, inserted.captured.chunkCount)
        assertEquals(text.length, inserted.captured.characterCount)
        coVerify(exactly = result.chunkCount) {
            memoryStore.storeIfAbsent(
                content = any(),
                source = "document:sha256-id",
                category = "document",
                importance = 0.65f,
                tags = match { "document" in it && "world-bible" in it },
            )
        }
    }

    @Test
    fun `delete removes document chunks before metadata`() = runTest {
        repository.delete("doc-1")
        coVerify { memoryStore.deleteBySource("document:doc-1") }
        coVerify { dao.deleteById("doc-1") }
    }

    @Test
    fun `observeAll delegates to document dao`() {
        coEvery { dao.observeAll() } returns flowOf(emptyList())
        repository.observeAll()
        io.mockk.verify { dao.observeAll() }
    }
}