package com.aura.kg

import com.aura.provenance.ConversationProvenance
import com.aura.memory.escapeLikeWildcards
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
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
        provenance: ConversationProvenance,
    ) = mutex.withLock {
        val now = System.currentTimeMillis()
        val stableTurnId = if (provenance.isPresent) {
            "${provenance.conversationId}:${provenance.turnTimestamp}"
        } else {
            ""
        }
        for (node in nodes) {
            val id = node.id.ifBlank { KgId.node(node.type, node.label) }
            dao.insertNode(
                node.copy(
                    id = id,
                    sourceTurnId = stableTurnId,
                    sourceConversationId = provenance.conversationId,
                    sourceTurnTimestamp = provenance.turnTimestamp,
                    updatedAt = now,
                ).toEntity()
            )
        }
        for (edge in edges) {
            val id = edge.id.ifBlank { KgId.edge(edge.type, edge.sourceId, edge.targetId) }
            dao.insertEdge(
                edge.copy(
                    id = id,
                    sourceTurnId = stableTurnId,
                    sourceConversationId = provenance.conversationId,
                    sourceTurnTimestamp = provenance.turnTimestamp,
                    lastReinforced = now,
                ).toEntity()
            )
        }
    }

    suspend fun search(query: String): List<KgNode> =
        dao.searchNodes(escapeLikeWildcards(query.trim())).map { KgNode.fromEntity(it) }

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

    /**
     * Edit user-correctable fields without changing the stable node id or
     * extraction provenance. Relations continue to point at the same node.
     */
    suspend fun updateNode(
        id: String,
        label: String,
        type: NodeType,
        properties: JsonObject,
        now: Long = System.currentTimeMillis(),
    ): KgNode = mutex.withLock {
        require(label.isNotBlank()) { "Node label cannot be blank" }
        val original = dao.getNode(id)
            ?: throw NoSuchElementException("Knowledge graph node not found: $id")
        val updated = original.copy(
            label = label.trim(),
            type = type.name.lowercase(),
            properties = Json.encodeToString(JsonObject.serializer(), properties),
            updatedAt = now,
        )
        dao.updateNode(updated)
        KgNode.fromEntity(updated)
    }

    /**
     * Merge [sourceId] into [targetId]. Target identity/provenance wins;
     * source-only properties and relations are retained. Relations that would
     * become target→target are discarded. The DAO publishes this atomically.
     */
    suspend fun mergeNodes(
        sourceId: String,
        targetId: String,
        now: Long = System.currentTimeMillis(),
    ): KgNode = mutex.withLock {
        require(sourceId != targetId) { "A node cannot be merged into itself" }
        val sourceEntity = dao.getNode(sourceId)
            ?: throw NoSuchElementException("Knowledge graph node not found: $sourceId")
        val targetEntity = dao.getNode(targetId)
            ?: throw NoSuchElementException("Knowledge graph node not found: $targetId")
        val source = KgNode.fromEntity(sourceEntity)
        val target = KgNode.fromEntity(targetEntity)
        val mergedProperties = JsonObject(source.properties + target.properties)
        val mergedTarget = targetEntity.copy(
            properties = Json.encodeToString(JsonObject.serializer(), mergedProperties),
            confidence = maxOf(sourceEntity.confidence, targetEntity.confidence),
            updatedAt = now,
            accessCount = sourceEntity.accessCount + targetEntity.accessCount,
            lastAccessed = maxOf(sourceEntity.lastAccessed, targetEntity.lastAccessed),
        )

        val rewritten = dao.neighbors(sourceId)
            .mapNotNull { edge ->
                val newSource = if (edge.sourceId == sourceId) targetId else edge.sourceId
                val newTarget = if (edge.targetId == sourceId) targetId else edge.targetId
                if (newSource == newTarget) return@mapNotNull null
                val type = EdgeType.from(edge.type)
                edge.copy(
                    id = KgId.edge(type, newSource, newTarget),
                    sourceId = newSource,
                    targetId = newTarget,
                    lastReinforced = maxOf(edge.lastReinforced, now),
                )
            }
            .distinctBy { Triple(it.sourceId, it.targetId, it.type) }

        dao.mergeNodeRecords(sourceId, mergedTarget, rewritten)
        target.copy(
            properties = mergedProperties,
            confidence = mergedTarget.confidence,
            updatedAt = now,
            accessCount = mergedTarget.accessCount,
            lastAccessed = mergedTarget.lastAccessed,
        )
    }

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
