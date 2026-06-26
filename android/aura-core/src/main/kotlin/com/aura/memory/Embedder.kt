package com.aura.memory

/**
 * Embedding interface. Produces a normalized FloatArray representation of text.
 *
 * v1: [LocalEmbedder] uses deterministic pseudo-embedding (SHA-256 + n-grams).
 * v2 (this task): [CloudEmbedder] wraps [LocalEmbedder] as fallback and calls
 * Ollama Cloud's embeddings API when a key is configured.
 */
interface Embedder {

    /** Embed [text] into a unit-normalized vector. */
    suspend fun embed(text: String): FloatArray

    companion object {
        fun toBytes(vec: FloatArray): ByteArray {
            val out = ByteArray(vec.size * 4)
            for (i in vec.indices) {
                val bits = java.lang.Float.floatToRawIntBits(vec[i])
                out[i * 4] = (bits ushr 24).toByte()
                out[i * 4 + 1] = (bits ushr 16).toByte()
                out[i * 4 + 2] = (bits ushr 8).toByte()
                out[i * 4 + 3] = bits.toByte()
            }
            return out
        }

        fun fromBytes(bytes: ByteArray): FloatArray {
            val n = bytes.size / 4
            val out = FloatArray(n)
            for (i in 0 until n) {
                val b0 = bytes[i * 4].toInt() and 0xff
                val b1 = bytes[i * 4 + 1].toInt() and 0xff
                val b2 = bytes[i * 4 + 2].toInt() and 0xff
                val b3 = bytes[i * 4 + 3].toInt() and 0xff
                val bits = (b0 shl 24) or (b1 shl 16) or (b2 shl 8) or b3
                out[i] = java.lang.Float.intBitsToFloat(bits)
            }
            return out
        }
    }
}
