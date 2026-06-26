package com.aura.memory

import kotlin.math.exp
import kotlin.math.ln

/**
 * A scored memory candidate with separate text-match and vector-similarity scores.
 *
 * @property memory  the underlying memory entity
 * @property textScore  BM25-like text overlap score (0..1)
 * @property vectorScore  cosine similarity between query and memory embedding (0..1)
 */
data class ScoredMemory(
    val memory: MemoryEntity,
    val textScore: Float = 0f,
    val vectorScore: Float = 0f,
)

/**
 * RRF (Reciprocal Rank Fusion) retrieval combiner.
 *
 * Ranks memory candidates by fusing six signals:
 * - [textScore]: term-overlap score (BM25-like)
 * - [vectorScore]: cosine similarity from vector search
 * - recencyScore: exponential decay based on [MemoryEntity.createdAt]
 * - accessScore: blend of access recency and access frequency
 * - decayScore: [MemoryEntity.decayScore] (FadeMem)
 * - importance: [MemoryEntity.importance] (semantic importance)
 *
 * Each signal produces a rank ordering; the final score is the sum of
 * 1/(k + rankᵢ) across all signals (RRF). k defaults to 60, which
 * moderates the influence of any single signal's top ranks.
 *
 * The output preserves the [MemoryEntity] objects sorted by fused score
 * descending, truncated to [topK].
 */
object Retrieval {

    /** RRF constant — higher values smooth rank disparities. */
    private const val RRF_K = 60f

    /** Half-life for recency and access-recency scoring (days). */
    private const val SIGNAL_HALF_LIFE_DAYS = 7.0

    // ── Internal rankable container ──────────────────────────────────────

    private data class Rankable(
        val index: Int,
        val memory: MemoryEntity,
        val textScore: Float,
        val vectorScore: Float,
        val recencyScore: Float,
        val accessScore: Float,
        val decayScore: Float,
        val importance: Float,
    )

    // ── Public API ───────────────────────────────────────────────────────

    /**
     * Rank memory candidates using RRF fusion and return the top-[topK].
     *
     * @param query  the original query text (used for text-score normalization)
     * @param queryEmbedding  the embedding vector of the query
     * @param candidates  scored candidates (memory + textScore + vectorScore)
     * @param topK  number of results to return
     * @param now  current timestamp in millis (injectable for deterministic tests)
     */
    fun rankCandidates(
        query: String,
        queryEmbedding: FloatArray,
        candidates: List<ScoredMemory>,
        topK: Int,
        now: Long = System.currentTimeMillis(),
    ): List<MemoryEntity> {
        if (candidates.isEmpty() || topK <= 0) return emptyList()

        // 1) Build rankables with all computed signals ──────────────────────
        val rankables = candidates.mapIndexed { idx, sm ->
            val mem = sm.memory

            // Recency: exponential decay from createdAt
            val ageDays = (now - mem.createdAt).coerceAtLeast(0L) / 86_400_000.0
            val recencyScore = exp(-ageDays * ln(2.0) / SIGNAL_HALF_LIFE_DAYS).toFloat()

            // Access score: blend of access recency (half-life decay) and
            // access frequency (logistic-style saturation).
            val daysSinceAccess = (now - mem.accessedAt).coerceAtLeast(0L) / 86_400_000.0
            val accessRecency = exp(-daysSinceAccess * ln(2.0) / SIGNAL_HALF_LIFE_DAYS).toFloat()
            val accessFreq = mem.accessCount.toFloat() / (mem.accessCount + 5).toFloat()
            val accessScore = accessRecency * 0.5f + accessFreq * 0.5f

            Rankable(
                index = idx,
                memory = mem,
                textScore = sm.textScore,
                vectorScore = sm.vectorScore,
                recencyScore = recencyScore,
                accessScore = accessScore,
                decayScore = mem.decayScore,
                importance = mem.importance,
            )
        }

        // 2) Rank by each signal (1 = best) ─────────────────────────────────
        fun rankBy(selector: (Rankable) -> Float): Map<Int, Int> =
            rankables
                .sortedByDescending { selector(it) }
                .mapIndexed { rank, r -> r.index to rank + 1 }
                .toMap()

        val textRanks = rankBy { it.textScore }
        val vectorRanks = rankBy { it.vectorScore }
        val recencyRanks = rankBy { it.recencyScore }
        val accessRanks = rankBy { it.accessScore }
        val decayRanks = rankBy { it.decayScore }
        val importanceRanks = rankBy { it.importance }

        // 3) Compute RRF score for each candidate ───────────────────────────
        val rrfScores = rankables.associate { r ->
            val score = listOf(
                1.0 / (RRF_K + textRanks.getValue(r.index)),
                1.0 / (RRF_K + vectorRanks.getValue(r.index)),
                1.0 / (RRF_K + recencyRanks.getValue(r.index)),
                1.0 / (RRF_K + accessRanks.getValue(r.index)),
                1.0 / (RRF_K + decayRanks.getValue(r.index)),
                1.0 / (RRF_K + importanceRanks.getValue(r.index)),
            ).sum()
            r.index to score
        }

        // 4) Sort by RRF score descending, take topK ─────────────────────────
        return rankables
            .sortedByDescending { rrfScores.getValue(it.index) }
            .take(topK)
            .map { it.memory }
    }
}
