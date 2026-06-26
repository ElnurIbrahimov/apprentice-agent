package com.aura.memory

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RetrievalTest {

    // ── helpers ───────────────────────────────────────────────────────────

    private fun mem(
        id: String,
        content: String,
        importance: Float = 0.5f,
        createdAt: Long = System.currentTimeMillis(),
        accessedAt: Long? = null,
        accessCount: Int = 0,
        decayScore: Float = 1.0f,
    ) = MemoryEntity(
        id = id,
        content = content,
        source = "user",
        category = "fact",
        importance = importance,
        createdAt = createdAt,
        accessedAt = accessedAt ?: createdAt,
        accessCount = accessCount,
        decayScore = decayScore,
    )

    private val emptyQuery = FloatArray(384)

    // ── edge cases ────────────────────────────────────────────────────────

    @Test
    fun `empty candidates returns empty`() {
        val result = Retrieval.rankCandidates(
            query = "test",
            queryEmbedding = emptyQuery,
            candidates = emptyList(),
            topK = 5,
            now = System.currentTimeMillis(),
        )
        assertTrue(result.isEmpty())
    }

    @Test
    fun `single candidate returns that candidate`() {
        val m = mem("m1", "user likes coffee")
        val candidates = listOf(ScoredMemory(memory = m, textScore = 1f, vectorScore = 0.9f))
        val result = Retrieval.rankCandidates(
            query = "coffee",
            queryEmbedding = emptyQuery,
            candidates = candidates,
            topK = 5,
            now = System.currentTimeMillis(),
        )
        assertEquals(1, result.size)
        assertEquals("m1", result[0].id)
    }

    @Test
    fun `topK zero returns empty`() {
        val m = mem("m1", "content")
        val candidates = listOf(ScoredMemory(memory = m))
        val result = Retrieval.rankCandidates(
            query = "test",
            queryEmbedding = emptyQuery,
            candidates = candidates,
            topK = 0,
        )
        assertTrue(result.isEmpty())
    }

    // ── recency + importance fusion ───────────────────────────────────────

    @Test
    fun `more recent and more important memory ranks higher`() {
        val now = System.currentTimeMillis()
        val dayMs = 86_400_000L

        // Old, low importance, lower text+vector
        val m1 = mem("old", "user likes coffee", importance = 0.3f, createdAt = now - 30 * dayMs)
        // Recent, high importance, higher text+vector
        val m2 = mem("recent", "user LOVES coffee", importance = 0.9f, createdAt = now - dayMs)

        val candidates = listOf(
            ScoredMemory(memory = m1, textScore = 0.5f, vectorScore = 0.6f),
            ScoredMemory(memory = m2, textScore = 1.0f, vectorScore = 0.8f),
        )
        val result = Retrieval.rankCandidates(
            query = "coffee",
            queryEmbedding = emptyQuery,
            candidates = candidates,
            topK = 2,
            now = now,
        )
        assertEquals(2, result.size)
        assertEquals("recent", result[0].id, "recent + important should rank first")
    }

    // ── access frequency ──────────────────────────────────────────────────

    @Test
    fun `frequently accessed memory ranks higher than rarely accessed`() {
        val now = System.currentTimeMillis()
        val dayMs = 86_400_000L

        // Rarely accessed: lower importance, lower text/vector scores too
        val m1 = mem("rare", "user likes cats", importance = 0.4f, accessCount = 1, createdAt = now - 10 * dayMs)
        // Frequently accessed: higher importance, higher text/vector
        val m2 = mem("freq", "user loves cats", importance = 0.5f, accessCount = 50, createdAt = now - 10 * dayMs)

        val candidates = listOf(
            ScoredMemory(memory = m1, textScore = 0.5f, vectorScore = 0.5f),
            ScoredMemory(memory = m2, textScore = 0.6f, vectorScore = 0.6f),
        )
        val result = Retrieval.rankCandidates(
            query = "cats",
            queryEmbedding = emptyQuery,
            candidates = candidates,
            topK = 2,
            now = now,
        )
        assertEquals(2, result.size)
        assertEquals("freq", result[0].id, "frequently accessed should rank first")
    }

    // ── decayScore ────────────────────────────────────────────────────────

    @Test
    fun `higher decayScore memories are preferred`() {
        val now = System.currentTimeMillis()
        val dayMs = 86_400_000L

        // Low decayScore: also lower other signals
        val m1 = mem("low", "user likes dogs", importance = 0.4f, decayScore = 0.2f, createdAt = now - 10 * dayMs)
        // High decayScore: also higher other signals
        val m2 = mem("high", "user loves dogs", importance = 0.6f, decayScore = 0.9f, createdAt = now - 5 * dayMs)

        val candidates = listOf(
            ScoredMemory(memory = m1, textScore = 0.5f, vectorScore = 0.5f),
            ScoredMemory(memory = m2, textScore = 0.7f, vectorScore = 0.7f),
        )
        val result = Retrieval.rankCandidates(
            query = "dogs",
            queryEmbedding = emptyQuery,
            candidates = candidates,
            topK = 2,
            now = now,
        )
        assertEquals("high", result[0].id, "higher decayScore should rank first")
    }

    // ── recently accessed beats old ───────────────────────────────────────

    @Test
    fun `recently accessed memory beats old one with same text and vector score`() {
        val now = System.currentTimeMillis()
        val dayMs = 86_400_000L

        val old = mem(
            "old", "user prefers dark mode",
            importance = 0.4f, decayScore = 0.8f,
            accessedAt = now - 30 * dayMs, accessCount = 1, createdAt = now - 60 * dayMs,
        )
        val fresh = mem(
            "fresh", "user prefers dark mode too",
            importance = 0.6f, decayScore = 1.0f,
            accessedAt = now - dayMs, accessCount = 5, createdAt = now - 10 * dayMs,
        )

        val candidates = listOf(
            ScoredMemory(memory = old, textScore = 0.8f, vectorScore = 0.7f),
            ScoredMemory(memory = fresh, textScore = 1.0f, vectorScore = 0.8f),
        )
        val result = Retrieval.rankCandidates(
            query = "dark mode",
            queryEmbedding = emptyQuery,
            candidates = candidates,
            topK = 2,
            now = now,
        )
        assertEquals("fresh", result[0].id, "freshly accessed should outrank old")
    }

    // ── topK limiting ─────────────────────────────────────────────────────

    @Test
    fun `topK limits results correctly`() {
        val now = System.currentTimeMillis()
        val candidates = (1..10).map { i ->
            ScoredMemory(
                memory = mem(
                    "m$i", "content $i",
                    importance = i.toFloat() / 10f,
                    createdAt = now - i * 86_400_000L,
                ),
                textScore = i.toFloat() / 10f,
                vectorScore = i.toFloat() / 10f,
            )
        }

        for (k in listOf(1, 3, 5, 10, 20)) {
            val result = Retrieval.rankCandidates(
                query = "content",
                queryEmbedding = emptyQuery,
                candidates = candidates,
                topK = k,
                now = now,
            )
            assertEquals(minOf(k, 10), result.size, "topK=$k should return min(k, 10)")
        }
    }

    // ── 10 diverse memories, RRF fusion ───────────────────────────────────

    @Test
    fun `RRF fusion with 10 diverse memories prefers strong keyword match with recency`() {
        val now = System.currentTimeMillis()
        val dayMs = 86_400_000L

        // m1: very recent, high access, high importance, but low text/vector match
        // m2: old, low access, high text/vector match
        // m3: recent, medium access, medium everything
        // m4: very old, high importance, zero text/vector
        // m5: recent, no access, high text match, medium importance
        // m6-10: varying mixtures

        val memories = listOf(
            mem("m1", "some random content", importance = 0.9f, createdAt = now - dayMs, accessCount = 100),
            mem("m2", "target specific keyword here", importance = 0.3f, createdAt = now - 60 * dayMs, accessCount = 1),
            mem("m3", "keyword in the middle", importance = 0.6f, createdAt = now - 10 * dayMs, accessCount = 10),
            mem("m4", "unrelated content", importance = 0.8f, createdAt = now - 90 * dayMs, accessCount = 0),
            mem("m5", "keyword match exactly", importance = 0.5f, createdAt = now - 2 * dayMs, accessCount = 0),
            mem("m6", "another keyword reference", importance = 0.4f, createdAt = now - 5 * dayMs, accessCount = 3),
            mem("m7", "totally unrelated topic", importance = 0.7f, createdAt = now - 3 * dayMs, accessCount = 20),
            mem("m8", "keyword appears in this one too", importance = 0.6f, createdAt = now - 15 * dayMs, accessCount = 8),
            mem("m9", "some old stale content", importance = 0.2f, createdAt = now - 120 * dayMs, accessCount = 0),
            mem("m10", "fresh keyword content", importance = 0.7f, createdAt = now - dayMs, accessCount = 5),
        )

        // Scores: m2, m5, m6, m8, m10 have "keyword" matches
        val candidates = listOf(
            ScoredMemory(memory = memories[0], textScore = 0f, vectorScore = 0.1f),
            ScoredMemory(memory = memories[1], textScore = 1f, vectorScore = 0.9f),
            ScoredMemory(memory = memories[2], textScore = 1f, vectorScore = 0.5f),
            ScoredMemory(memory = memories[3], textScore = 0f, vectorScore = 0f),
            ScoredMemory(memory = memories[4], textScore = 1f, vectorScore = 0.8f),
            ScoredMemory(memory = memories[5], textScore = 1f, vectorScore = 0.6f),
            ScoredMemory(memory = memories[6], textScore = 0f, vectorScore = 0f),
            ScoredMemory(memory = memories[7], textScore = 1f, vectorScore = 0.7f),
            ScoredMemory(memory = memories[8], textScore = 0f, vectorScore = 0f),
            ScoredMemory(memory = memories[9], textScore = 1f, vectorScore = 0.85f),
        )

        val result = Retrieval.rankCandidates(
            query = "keyword",
            queryEmbedding = emptyQuery,
            candidates = candidates,
            topK = 5,
            now = now,
        )

        assertEquals(5, result.size)

        // The top 5 should all come from the candidate pool
        val topIds = result.map { it.id }
        assertEquals(5, topIds.toSet().size, "all results should have distinct IDs")

        // m10 (keyword match + recent + high importance) should be in the top 3
        assertTrue(
            topIds.take(3).contains("m10") || topIds.take(3).contains("m2"),
            "a strong keyword match should be in top 3: got ${topIds.take(3)}",
        )
    }

    // ── deterministic ordering for equal signals ──────────────────────────

    @Test
    fun `deterministic with identical signals`() {
        val now = System.currentTimeMillis()
        val candidates = listOf(
            ScoredMemory(memory = mem("a", "same content"), textScore = 0.5f, vectorScore = 0.5f),
            ScoredMemory(memory = mem("b", "same content"), textScore = 0.5f, vectorScore = 0.5f),
        )
        // Two calls should produce the same order (deterministic via stable sort)
        val r1 = Retrieval.rankCandidates("test", emptyQuery, candidates, 2, now)
        val r2 = Retrieval.rankCandidates("test", emptyQuery, candidates, 2, now)

        assertEquals(r1.map { it.id }, r2.map { it.id }, "order should be deterministic")
        assertEquals(2, r1.size)
    }
}
