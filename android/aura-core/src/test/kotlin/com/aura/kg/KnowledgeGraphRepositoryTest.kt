package com.aura.kg

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class KnowledgeGraphRepositoryTest {

    private val dao = mockk<KnowledgeGraphDao>(relaxed = true)
    private val repo = KnowledgeGraphRepository(dao)

    @Test
    fun `saveGraph inserts nodes and edges`() = runTest {
        val node = KgNode(id = "", label = "Kotlin", type = NodeType.SKILL)
        val edge = KgEdge(id = "", type = EdgeType.LEARNED_FROM, sourceId = "a", targetId = "b")
        coEvery { dao.insertNode(any()) } returns Unit
        coEvery { dao.insertEdge(any()) } returns Unit
        repo.saveGraph(listOf(node), listOf(edge), "turn-1")
        coVerify(exactly = 1) { dao.insertNode(any()) }
        coVerify(exactly = 1) { dao.insertEdge(any()) }
    }

    @Test
    fun `search returns mapped nodes`() = runTest {
        coEvery { dao.searchNodes("kotlin", 50) } returns listOf(
            NodeEntity(id = "1", label = "Kotlin", type = "skill"),
        )
        val result = repo.search("kotlin")
        assertEquals(1, result.size)
        assertEquals("Kotlin", result[0].label)
        assertEquals(NodeType.SKILL, result[0].type)
    }

    @Test
    fun `getNode increments access count`() = runTest {
        coEvery { dao.getNode("1") } returns NodeEntity(id = "1", label = "Elnur", type = "person", accessCount = 3)
        coEvery { dao.incrementAccessCount("1", any()) } returns Unit
        val result = repo.getNode("1")
        assertEquals("Elnur", result?.label)
        assertEquals(4, result?.accessCount)
        coVerify { dao.incrementAccessCount("1", any()) }
    }

    @Test
    fun `getNode returns null when missing`() = runTest {
        coEvery { dao.getNode("missing") } returns null
        assertNull(repo.getNode("missing"))
    }

    @Test
    fun `findPath returns direct path`() = runTest {
        coEvery { dao.edgesFrom("a") } returns listOf(
            EdgeEntity(id = "e1", type = "knows", sourceId = "a", targetId = "b"),
        )
        val path = repo.findPath("a", "b")
        assertEquals(listOf("a", "b"), path)
    }

    @Test
    fun `findPath returns empty when no path`() = runTest {
        coEvery { dao.edgesFrom("a") } returns emptyList()
        val path = repo.findPath("a", "b")
        assertTrue(path.isEmpty())
    }
}
