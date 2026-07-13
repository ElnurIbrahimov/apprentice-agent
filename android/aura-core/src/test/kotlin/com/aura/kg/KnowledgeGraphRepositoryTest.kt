package com.aura.kg

import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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

    @Test
    fun `updateNode preserves identity and provenance`() = runTest {
        val original = NodeEntity(
            id = "person-1",
            label = "Elnur",
            type = "person",
            properties = "{\"city\":\"Baku\"}",
            confidence = 0.7f,
            sourceTurnId = "turn-7",
            createdAt = 100L,
            updatedAt = 110L,
            accessCount = 4,
            lastAccessed = 120L,
        )
        coEvery { dao.getNode("person-1") } returns original

        val updated = repo.updateNode(
            id = "person-1",
            label = "Elnur Ibrahimov",
            type = NodeType.PERSON,
            properties = buildJsonObject { put("city", JsonPrimitive("Baku")) },
            now = 500L,
        )

        assertEquals("person-1", updated.id)
        assertEquals("Elnur Ibrahimov", updated.label)
        assertEquals("turn-7", updated.sourceTurnId)
        assertEquals(100L, updated.createdAt)
        assertEquals(500L, updated.updatedAt)
        coVerify(exactly = 1) {
            dao.updateNode(match {
                it.id == "person-1" &&
                    it.label == "Elnur Ibrahimov" &&
                    it.sourceTurnId == "turn-7" &&
                    it.createdAt == 100L &&
                    it.updatedAt == 500L
            })
        }
    }

    @Test
    fun `mergeNodes rewrites relations merges properties and removes source`() = runTest {
        val source = NodeEntity(
            id = "source",
            label = "Aura Android",
            type = "project",
            properties = "{\"platform\":\"Android\",\"owner\":\"source\"}",
            confidence = 0.8f,
            sourceTurnId = "turn-source",
            createdAt = 10L,
            updatedAt = 20L,
        )
        val target = NodeEntity(
            id = "target",
            label = "Aura",
            type = "project",
            properties = "{\"owner\":\"Elnur\"}",
            confidence = 0.9f,
            sourceTurnId = "turn-target",
            createdAt = 5L,
            updatedAt = 30L,
        )
        val incoming = EdgeEntity(
            id = "incoming",
            type = "created_by",
            sourceId = "person",
            targetId = "source",
        )
        val outgoing = EdgeEntity(
            id = "outgoing",
            type = "uses",
            sourceId = "source",
            targetId = "tool",
        )
        val becomesSelfEdge = EdgeEntity(
            id = "self",
            type = "relates_to",
            sourceId = "source",
            targetId = "target",
        )
        coEvery { dao.getNode("source") } returns source
        coEvery { dao.getNode("target") } returns target
        coEvery { dao.neighbors("source") } returns listOf(incoming, outgoing, becomesSelfEdge)

        val merged = repo.mergeNodes("source", "target", now = 700L)

        assertEquals("target", merged.id)
        assertEquals(JsonPrimitive("Android"), merged.properties["platform"])
        assertEquals(JsonPrimitive("Elnur"), merged.properties["owner"])
        assertEquals(0.9f, merged.confidence)
        coVerify(exactly = 1) {
            dao.mergeNodeRecords(
                sourceId = "source",
                target = match { it.id == "target" && it.updatedAt == 700L },
                rewrittenEdges = match { edges ->
                    edges.size == 2 &&
                        edges.any { it.sourceId == "person" && it.targetId == "target" } &&
                        edges.any { it.sourceId == "target" && it.targetId == "tool" } &&
                        edges.none { it.sourceId == it.targetId } &&
                        edges.all { it.id == KgId.edge(EdgeType.from(it.type), it.sourceId, it.targetId) }
                },
            )
        }
    }

    @Test
    fun `mergeNodes rejects self merge and missing nodes`() = runTest {
        assertFailsWith<IllegalArgumentException> {
            repo.mergeNodes("same", "same")
        }

        coEvery { dao.getNode("missing") } returns null
        assertFailsWith<NoSuchElementException> {
            repo.mergeNodes("missing", "target")
        }
    }
}
