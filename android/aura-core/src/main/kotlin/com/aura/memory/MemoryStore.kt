package com.aura.memory

import kotlinx.coroutines.flow.Flow
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class MemoryStore @Inject constructor(
    private val dao: MemoryDao,
    private val embedder: Embedder,
    private val vectorIndex: VectorIndex,
    private val writeGate: WriteGate,
) {
    suspend fun maybeStore(content: String, source: String = "user"): String? {
        val decision = writeGate.evaluate(content, source)
        if (!decision.shouldStore) return null
        val id = UUID.randomUUID().toString()
        val embedding = embedder.embed(content)
        val now = System.currentTimeMillis()
        dao.insert(
            MemoryEntity(
                id = id,
                content = content,
                source = source,
                category = decision.category,
                importance = decision.importance,
                embedding = Embedder.toBytes(embedding),
                createdAt = now,
                accessedAt = now,
                decayScore = 1.0f,
            )
        )
        return id
    }

    suspend fun store(content: String, source: String, category: String, importance: Float, tags: List<String> = emptyList()): String {
        val id = UUID.randomUUID().toString()
        val embedding = embedder.embed(content)
        val now = System.currentTimeMillis()
        dao.insert(
            MemoryEntity(
                id = id,
                content = content,
                source = source,
                category = category,
                importance = importance,
                embedding = Embedder.toBytes(embedding),
                createdAt = now,
                accessedAt = now,
                decayScore = 1.0f,
                tags = tags.joinToString(","),
            )
        )
        return id
    }

    suspend fun query(text: String, limit: Int = 5): List<MemoryEntity> {
        // BM25-like: simple text match by LIKE, then rerank with vector similarity
        val textHits = dao.searchByText("%$text%", limit * 3)
        if (textHits.isEmpty()) return emptyList()
        val qVec = embedder.embed(text)
        val candidates = textHits.mapNotNull { mem ->
            mem.embedding?.let { mem.id to Embedder.fromBytes(it) }
        }
        val vectorHits = vectorIndex.search(qVec, candidates, topK = limit * 2)
        val byId = textHits.associateBy { it.id }
        val hitIds = (vectorHits.map { it.memoryId } + textHits.map { it.id }).distinct()
        return hitIds.mapNotNull { byId[it] }.take(limit)
    }

    suspend fun recent(limit: Int = 20): List<MemoryEntity> = dao.recent(limit)
    suspend fun byCategory(category: String, limit: Int = 20): List<MemoryEntity> = dao.byCategory(category, limit)
    suspend fun top(limit: Int = 20): List<MemoryEntity> = dao.top(limit)
    suspend fun get(id: String): MemoryEntity? = dao.getById(id)
    suspend fun forget(id: String) = dao.delete(id)

    suspend fun touch(id: String) {
        dao.touch(id)
    }

    fun observeCount(): Flow<Int> = dao.count()
    suspend fun count(): Int = dao.countOnce()

    /**
     * Run decay pass: halves decayScore for all memories older than the cutoff.
     * Called periodically (WorkManager in v2) or on app open.
     */
    suspend fun runDecayPass() {
        val now = System.currentTimeMillis()
        // For each memory, recompute its score
        val all = dao.recent(1000)
        for (mem in all) {
            val newScore = FadeMem.compute(mem.createdAt, mem.accessedAt, now)
            if (kotlin.math.abs(newScore - mem.decayScore) > 0.05f) {
                dao.update(mem.copy(decayScore = newScore))
            }
        }
    }
}
