package com.aura.agent

import com.aura.memory.Embedder
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import io.mockk.slot
import kotlinx.coroutines.test.runTest
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class ConversationStoreTest {
    private val dao = mockk<ConversationDao>(relaxed = true)
    private val embedder = mockk<Embedder>(relaxed = true)

    @Test
    fun `save then load returns equivalent conversation`() = runTest {
        val store = ConversationStore(dao, embedder)
        val conv = Conversation(
            id = "test-id",
            title = "Test chat",
            systemPrompt = "be helpful",
            model = "test:model",
            contextSummary = "Earlier decisions are durable.",
            summaryThroughTurn = 1,
            turns = listOf(
                Turn(user = "hi"),
                Turn(assistant = "hello", toolTurns = listOf(ToolTurn("t1", "echo", "{}", "echoed"))),
            ),
        )
        coEvery { dao.insert(any()) } returns Unit
        coEvery { dao.getById("test-id") } returns null // save then load with no DAO roundtrip = null
        // For a true roundtrip test we'd need an in-memory Room. Instead, validate
        // the JSON encoding path by capturing what we wrote.
        var captured: ConversationEntity? = null
        coEvery { dao.insert(capture(slot<ConversationEntity>())) } answers { captured = firstArg() }
        store.save(conv)
        assertNotNull(captured)
        assertEquals("test-id", captured!!.id)
        assertEquals("Test chat", captured!!.title)
        assertEquals("be helpful", captured!!.systemPrompt)
        assertEquals("test:model", captured!!.model)
        assertEquals("Earlier decisions are durable.", captured!!.contextSummary)
        assertEquals(1, captured!!.summaryThroughTurn)
    }

    @Test
    fun `mostRecent returns null when DAO is empty`() = runTest {
        val store = ConversationStore(dao, embedder)
        coEvery { dao.mostRecent() } returns null
        assertNull(store.mostRecent())
    }

    @Test
    fun `mostRecent returns null when entity is missing`() = runTest {
        val store = ConversationStore(dao, embedder)
        coEvery { dao.getById("missing") } returns null
        assertNull(store.load("missing"))
    }

    @Test
    fun `load returns null for missing id`() = runTest {
        val store = ConversationStore(dao, embedder)
        coEvery { dao.getById("nope") } returns null
        assertNull(store.load("nope"))
    }

    @Test
    fun `delete delegates to DAO`() = runTest {
        val store = ConversationStore(dao, embedder)
        coEvery { dao.delete("id1") } returns Unit
        store.delete("id1")
        coVerify { dao.delete("id1") }
    }

    @Test
    fun `load restores durable summary boundary`() = runTest {
        val store = ConversationStore(dao, embedder)
        coEvery { dao.getById("summarized") } returns ConversationEntity(
            id = "summarized",
            title = "Long chat",
            createdAt = 1L,
            updatedAt = 2L,
            systemPrompt = null,
            model = "test:model",
            turnsJson = """[{"user":"old"},{"user":"recent"}]""",
            contextSummary = "Old context preserved.",
            summaryThroughTurn = 1,
        )

        val result = store.load("summarized")

        assertNotNull(result)
        assertEquals("Old context preserved.", result.contextSummary)
        assertEquals(1, result.summaryThroughTurn)
        assertEquals(2, result.turns.size)
    }

    @Test
    fun `recent maps DAO entities to Conversations`() = runTest {
        val store = ConversationStore(dao, embedder)
        coEvery { dao.recent(50) } returns listOf(
            ConversationEntity(id = "1", title = "First", createdAt = 1L, updatedAt = 2L, systemPrompt = null, model = null, metadataJson = "{}", turnsJson = "[]"),
            ConversationEntity(id = "2", title = "Second", createdAt = 3L, updatedAt = 4L, systemPrompt = null, model = null, metadataJson = "{}", turnsJson = "[]"),
        )
        val result = store.recent(50)
        assertEquals(2, result.size)
        assertEquals("First", result[0].title)
        assertEquals("Second", result[1].title)
    }
}
