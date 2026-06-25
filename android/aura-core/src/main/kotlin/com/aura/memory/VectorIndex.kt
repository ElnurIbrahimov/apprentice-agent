package com.aura.memory

import kotlin.math.sqrt

class VectorIndex(private val dim: Int = 384) {
    data class Hit(val memoryId: String, val score: Float)

    fun search(query: FloatArray, candidates: List<Pair<String, FloatArray>>, topK: Int = 10): List<Hit> {
        if (candidates.isEmpty()) return emptyList()
        val qn = norm(query)
        if (qn == 0f) return emptyList()
        return candidates.mapNotNull { (id, vec) ->
            val score = cosine(query, qn, vec)
            if (score > 0.05f) Hit(id, score) else null
        }.sortedByDescending { it.score }.take(topK)
    }

    private fun norm(v: FloatArray): Float {
        var s = 0f
        for (x in v) s += x * x
        return sqrt(s)
    }

    private fun cosine(a: FloatArray, aNorm: Float, b: FloatArray): Float {
        if (b.size != a.size) return 0f
        var dot = 0f
        var bNorm = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            bNorm += b[i] * b[i]
        }
        val bN = sqrt(bNorm)
        if (bN == 0f) return 0f
        return dot / (aNorm * bN)
    }
}
