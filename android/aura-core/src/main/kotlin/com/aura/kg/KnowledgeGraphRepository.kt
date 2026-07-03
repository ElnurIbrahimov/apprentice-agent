package com.aura.kg

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.ArrayDeque
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class KnowledgeGraphRepository @Inject constructor(
    private val dao: KnowledgeGraphDao,
) {
    private val mutex = Mutex()

    suspend fun saveGraph(
        nodes: List<KgNode>,
        edges: List<KgEdge>,
        sourceTurnId: String,
    ) = mutex.withLock {
        val now = System.currentTimeMillis()
        for (node in nodes) {
            val id = node.id.ifBlank { KgId.node(node.type, node.label) }
            dao.insertNode(
                node.copy(
                    id = id,
                    sourceTurnId = sourceTurnId,
                    updatedAt = now,
                ).toEntity()
            )
        }
        for (edge in edges) {
            val id = edge.id.ifBlank { KgId.edge(edge.type, edge.sourceId, edge.targetId) }
            dao.insertEdge(
                edge.copy(
                    id = id,
                    sourceTurnId = sourceTurnId,
                    lastReinforced = now,
                ).toEntity()
            )
        }
    }

    suspend fun search(query: String): List<KgNode> =
        dao.searchNodes(escapeLikeWildcards(query.trim())).map { KgNode.fromEntity(it) }

    private fun escapeLikeWildcards(s: String): String = s
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")

    suspend fun recent(limit: Int = 50): List<KgNode> =
        dao.recentNodes(limit).map { KgNode.fromEntity(it) }

    /**
     * KG nodes whose updatedAt is at or after [sinceMs], newest
     * first. Used by the morning brief to surface "X facts learned
     * yesterday" without scanning the full graph.
     */
    suspend fun recentSince(sinceMs: Long, limit: Int = 20): List<KgNode> =
        dao.recentNodesSince(sinceMs, limit).map { KgNode.fromEntity(it) }

    suspend fun getNode(id: String): KgNode? {
        val node = dao.getNode(id) ?: return null
        dao.incrementAccessCount(id)
        return KgNode.fromEntity(node.copy(accessCount = node.accessCount + 1))
    }

    suspend fun getNodeByLabel(label: String): KgNode? =
        dao.getNodeByLabel(label)?.let { KgNode.fromEntity(it) }

    suspend fun getNeighbors(id: String): Neighbors {
        val edges = dao.edgesForNode(id)
        val outgoing = edges.filter { it.sourceId == id }.map { KgEdge.fromEntity(it) }
        val incoming = edges.filter { it.targetId == id }.map { KgEdge.fromEntity(it) }
        return Neighbors(incoming, outgoing)
    }

    /**
     * Breadth-first search for shortest path between two nodes.
     * Returns list of node IDs from [fromId] to [toId], or empty if no path.
     */
    suspend fun findPath(fromId: String, toId: String): List<String> {
        if (fromId == toId) return listOf(fromId)
        val visited = mutableSetOf(fromId)
        val queue = ArrayDeque<List<String>>().apply { add(listOf(fromId)) }
        while (queue.isNotEmpty()) {
            val path = queue.removeFirst()
            val last = path.last()
            val edges = dao.edgesFrom(last)
            for (edge in edges) {
                val next = edge.targetId
                if (next in visited) continue
                val newPath = path + next
                if (next == toId) return newPath
                visited.add(next)
                queue.add(newPath)
            }
        }
        return emptyList()
    }

    suspend fun deleteNode(id: String) {
        dao.deleteEdgesForNode(id)
        dao.deleteNode(id)
    }

    suspend fun stats(): Stats = Stats(
        nodeCount = dao.nodeCount(),
        edgeCount = dao.edgeCount(),
    )

    data class Neighbors(
        val incoming: List<KgEdge>,
        val outgoing: List<KgEdge>,
    )

    data class Stats(val nodeCount: Int, val edgeCount: Int)
}
