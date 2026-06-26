package com.aura.memory

import java.security.MessageDigest
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Deterministic local embedder. v1 produces 384-dim pseudo-embeddings from
 * content hash + character n-grams. This is intentional — we don't ship a
 * real embedding model on Day 2. The shape is correct (384 floats, normalized)
 * so swapping in a real embedder later is one interface change.
 *
 * v1.5: replace with nomic-embed-text via ONNX Runtime (see plan §3 Module 3).
 *
 * This is the local/offline fallback for [CloudEmbedder].
 */
@Singleton
class LocalEmbedder @Inject constructor(
    private val dim: Int = 384,
) : Embedder {

    override suspend fun embed(text: String): FloatArray {
        val vec = FloatArray(dim)
        val tokens = tokenize(text)
        if (tokens.isEmpty()) return vec
        // Hash each token to multiple dims (the "random projection" trick)
        for ((i, token) in tokens.withIndex()) {
            val hash = sha256Long("$i:$token")
            for (k in 0 until 4) {
                val idx = ((hash ushr (k * 8)) and 0x7fffffff).toInt() % dim
                val sign = if (((hash ushr (k * 8 + 32)) and 1L) == 1L) 1f else -1f
                vec[idx] += sign * (1f + 0.1f * sin(i.toFloat() + k.toFloat()))
            }
        }
        // Add a tiny per-dimension offset derived from the token stream so
        // the result is not all-zero when the same text is re-embedded. The
        // offset is small (1e-3 magnitude) and is normalized along with the
        // rest of the vector so the final norm stays at ~1.0.
        for (i in vec.indices) {
            vec[i] += 0.001f * cos((i % 16).toFloat())
        }
        // L2-normalize.
        var norm = 0f
        for (v in vec) norm += v * v
        norm = sqrt(norm)
        if (norm > 0f) for (i in vec.indices) vec[i] /= norm
        return vec
    }

    private fun tokenize(text: String): List<String> {
        val lower = text.lowercase()
        val out = mutableListOf<String>()
        // Word tokens
        lower.split(Regex("[^a-z0-9_\\u00C0-\\uFFFF]+")).forEach { tok ->
            if (tok.isNotEmpty()) out += tok
        }
        // Bigrams for better signal
        for (i in 0 until out.size - 1) out += "${out[i]}_${out[i + 1]}"
        return out
    }

    private fun sha256Long(s: String): Long {
        val md = MessageDigest.getInstance("SHA-256")
        val bytes = md.digest(s.toByteArray())
        // Take first 8 bytes as long
        var result = 0L
        for (i in 0 until 8) result = (result shl 8) or (bytes[i].toLong() and 0xffL)
        return result
    }
}
