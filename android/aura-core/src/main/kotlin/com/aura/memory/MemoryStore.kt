package com.aura.memory

import kotlinx.coroutines.flow.Flow
import kotlin.math.sqrt
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
        // RRF fusion: text match + vector similarity + recency + access + decay + importance.
        // See [Retrieval.rankCandidates] for the RRF scoring details.
        // On hit, call [touch] to bump accessedAt + accessCount + decayScore. This
        // is what makes FadeMem meaningful — a frequently-recalled fact decays
        // slower. Without it, every memory decays at the same rate regardless of
        // how useful it actually is to the model.
        val escapedText = escapeLikeWildcards(text)
        val textHits = dao.searchByText("%$escapedText%", limit * 3)
        if (textHits.isEmpty()) return emptyList()
        val qVec = embedder.embed(text)

        // Build [ScoredMemory] candidates with text and vector similarity scores.
        val queryTokens = text.lowercase().split(Regex("\\s+")).filter { it.isNotBlank() }.toSet()
        val scoredCandidates = textHits.map { mem ->
            val embedding = mem.embedding?.let { Embedder.fromBytes(it) }

            // Text score: proportion of query tokens present in content (BM25-like).
            val contentLower = mem.content.lowercase()
            val matchedTokens = queryTokens.count { it in contentLower }
            val textScore = if (queryTokens.isNotEmpty()) matchedTokens.toFloat() / queryTokens.size else 0f

            // Vector score: cosine similarity if embedding available, else 0.
            val vectorScore = if (embedding != null) cosineSimilarity(qVec, embedding) else 0f

            ScoredMemory(memory = mem, textScore = textScore, vectorScore = vectorScore)
        }

        val results = Retrieval.rankCandidates(
            query = text,
            queryEmbedding = qVec,
            candidates = scoredCandidates,
            topK = limit,
            now = System.currentTimeMillis(),
        )

        // Touch is fire-and-forget; we don't want a failed decay update to break recall.
        for (mem in results) {
            runCatching { touch(mem.id) }
        }
        return results
    }

    /**
     * List memories filtered by category. Unlike [query] this is a direct
     * Room filter (no embedding or text matching) and does NOT touch the
     * returned memories — category browsing is metadata, not a recall.
     */
    suspend fun listByCategory(category: String, limit: Int = 50): List<MemoryEntity> =
        dao.byCategory(category, limit)

    suspend fun recent(limit: Int = 20): List<MemoryEntity> = dao.recent(limit)

    /**
     * Memories created in the last [sinceMs] ms, newest first. Used
     * by the morning brief to surface "what you learned yesterday."
     * Bounded by [limit] so a freshly-imported backup with 500
     * new rows doesn't blow up the LLM prompt.
     */
    suspend fun recentSince(sinceMs: Long, limit: Int = 20): List<MemoryEntity> =
        dao.recentSince(sinceMs, limit)

    /**
     * Memories whose decayScore is at or below [threshold]. The
     * morning brief uses this for the "X memories are fading" line.
     * Most-faded first.
     */
    suspend fun decayedBelow(threshold: Float, limit: Int = 20): List<MemoryEntity> =
        dao.decayedBelow(threshold, limit)
    suspend fun byCategory(category: String, limit: Int = 20): List<MemoryEntity> = dao.byCategory(category, limit)
    suspend fun top(limit: Int = 20): List<MemoryEntity> = dao.top(limit)
    suspend fun get(id: String): MemoryEntity? = dao.getById(id)
    suspend fun forget(id: String) = dao.delete(id)

    /**
     * Re-embed every memory that currently has a null embedding.
     * Returns the number of rows that were re-embedded.
     *
     * Use case: after a backup restore (which intentionally drops
     * embeddings — see [com.aura.backup.BackupManager.snapshot] for
     * why), every imported row has embedding=null. The next recall
     * would trigger a lazy re-embed per row, but that's slow when
     * there are hundreds. The Memory screen exposes this as a
     * "Rebuild embeddings" action so the user can do the sweep in
     * one pass.
     *
     * Memories with a non-null embedding are left alone. Embeddings
     * are model-specific so re-embedding with the current model is
     * safe and the result is what future recalls will use.
     *
     * Failures on individual rows are swallowed and logged; the
     * rebuild returns the count of successful re-embeds so the UI
     * can show "Rebuilt 142 of 145".
     */
    suspend fun rebuildEmbeddings(): Int {
        val pending = dao.allForExport().filter { it.embedding == null }
        if (pending.isEmpty()) return 0
        var rebuilt = 0
        for (mem in pending) {
            val ok = runCatching {
                val vec = embedder.embed(mem.content)
                dao.update(mem.copy(embedding = Embedder.toBytes(vec)))
            }.isSuccess
            if (ok) rebuilt += 1
        }
        return rebuilt
    }

    /**
     * Update an existing memory's content + category. Used by the
     * Memory edit UI when the user fixes a fact the model got wrong.
     * The embedding is set to null — the next recall will trigger a
     * lazy re-embed, or the user can hit the Memory tab's
     * "Rebuild embeddings" action to re-embed every invalidated row
     * in one pass.
     *
     * If [id] does not exist this is a no-op (the user probably
     * deleted the row from another path between opening the edit
     * dialog and tapping Save). The refresh happens automatically
     * via [observeCount] in the calling VM.
     */
    suspend fun update(id: String, content: String, category: String, importance: Float = 0.5f, tags: String = "") {
        val existing = dao.getById(id) ?: return
        dao.update(
            existing.copy(
                content = content,
                category = category,
                importance = importance,
                tags = tags,
                // Invalidate the embedding so the next recall re-embeds.
                embedding = null,
                // Bump accessedAt so a freshly-edited memory ranks higher
                // in the next recall.
                accessedAt = System.currentTimeMillis(),
            ),
        )
    }

    suspend fun touch(id: String) {
        dao.touch(id)
    }

    fun observeCount(): Flow<Int> = dao.count()
    suspend fun count(): Int = dao.countOnce()

    /**
     * Run decay pass: recompute the decay score for every memory. The work
     * is bounded by the table size; on a large memory DB this is still O(n)
     * but n is small in practice (hundreds to a few thousand).
     */
    suspend fun runDecayPass() {
        val now = System.currentTimeMillis()
        val all = dao.recent(10_000)  // hard cap; raise if needed
        for (mem in all) {
            val newScore = FadeMem.compute(mem.createdAt, mem.accessedAt, now)
            if (kotlin.math.abs(newScore - mem.decayScore) > 0.05f) {
                dao.update(mem.copy(decayScore = newScore))
            }
        }
    }
}

/** Fast cosine similarity between two same-dimension float arrays. */
private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    if (a.size != b.size) return 0f
    var dot = 0f
    var aNorm = 0f
    var bNorm = 0f
    for (i in a.indices) {
        dot += a[i] * b[i]
        aNorm += a[i] * a[i]
        bNorm += b[i] * b[i]
    }
    val aN = sqrt(aNorm)
    val bN = sqrt(bNorm)
    if (aN == 0f || bN == 0f) return 0f
    return dot / (aN * bN)
}

/**
 * Escape SQL LIKE wildcards (% and _) so user queries containing these
 * characters are matched literally rather than acting as pattern metacharacters.
 * Must be kept in sync with the ESCAPE '\' clause in MemoryDao.searchByText.
 */
internal fun escapeLikeWildcards(s: String): String = s
    .replace("\\", "\\\\")
    .replace("%", "\\%")
    .replace("_", "\\_")
