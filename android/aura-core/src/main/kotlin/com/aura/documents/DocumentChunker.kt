package com.aura.documents

/** Deterministic, paragraph-friendly text chunking for local document recall. */
object DocumentChunker {
    const val DEFAULT_MAX_CHARS = 1_800
    const val DEFAULT_OVERLAP_CHARS = 180

    fun chunk(
        text: String,
        maxChars: Int = DEFAULT_MAX_CHARS,
        overlapChars: Int = DEFAULT_OVERLAP_CHARS,
    ): List<String> {
        require(maxChars >= 80) { "maxChars must be at least 80" }
        require(overlapChars in 0 until maxChars) { "overlapChars must be smaller than maxChars" }

        val clean = text
            .replace("\r\n", "\n")
            .replace('\r', '\n')
            .replace(Regex("[ \\t]+\n"), "\n")
            .replace(Regex("\n{3,}"), "\n\n")
            .trim()
        if (clean.isEmpty()) return emptyList()
        if (clean.length <= maxChars) return listOf(clean)

        val chunks = mutableListOf<String>()
        var start = 0
        while (start < clean.length) {
            var end = minOf(start + maxChars, clean.length)
            if (end < clean.length) {
                val preferred = clean.lastIndexOf("\n\n", end).takeIf { it >= start + maxChars / 2 }
                    ?: clean.lastIndexOf(' ', end).takeIf { it >= start + maxChars / 2 }
                if (preferred != null) end = preferred
            }
            val value = clean.substring(start, end).trim()
            if (value.isNotEmpty()) chunks += value
            if (end >= clean.length) break

            var next = (end - overlapChars).coerceAtLeast(start + 1)
            val boundary = clean.indexOf(' ', next).takeIf { it in next until end }
            if (boundary != null) next = boundary + 1
            start = next
        }
        return chunks
    }
}