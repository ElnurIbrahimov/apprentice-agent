package com.aura.memory

/**
 * Decides whether a piece of content is worth storing as a memory.
 * Mirrors aura/memory/write_gate.py. v1: simple heuristics. v1.5: learned.
 */
class WriteGate {
    data class Decision(
        val shouldStore: Boolean,
        val category: String = "fact",
        val importance: Float = 0.5f,
        val reason: String = "",
    )

    fun evaluate(content: String, source: String): Decision {
        val lower = content.lowercase().trim()
        if (lower.isEmpty()) return Decision(false, reason = "empty")
        if (lower.length < 4) return Decision(false, reason = "too_short")
        if (source == "system") return Decision(false, reason = "system_msg")

        // Category classification by keyword
        val category = when {
            listOf("i prefer", "i like", "i hate", "i love", "i use", "my favorite", "i always", "i never")
                .any { lower.contains(it) } -> "preference"
            listOf("remember", "don't forget", "keep in mind", "note that")
                .any { lower.contains(it) } -> "fact"
            lower.startsWith("remind me") || lower.startsWith("todo") || lower.contains("schedule") -> "task"
            listOf("call mom", "call dad", "my friend", "my colleague")
                .any { lower.contains(it) } -> "person"
            lower.startsWith("today") || lower.startsWith("yesterday") || lower.contains("happened") -> "episode"
            listOf("idea", "what if", "maybe we could", "we should")
                .any { lower.contains(it) } -> "idea"
            else -> "fact"
        }

        val importance = when (category) {
            "preference" -> 0.8f
            "person" -> 0.7f
            "task" -> 0.6f
            "episode" -> 0.4f
            else -> 0.5f
        }

        return Decision(
            shouldStore = true,
            category = category,
            importance = importance,
            reason = "classified",
        )
    }
}
