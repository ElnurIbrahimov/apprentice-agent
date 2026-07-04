package com.aura.ui.util

/**
 * User-facing label for a `prefix:model` provider id, e.g.
 * `"ollama:deepseek-v4-pro:cloud"` → `"DeepSeek V4 Pro · ollama"`.
 *
 * Shared between the Chat screen header (current model) and the
 * History screen (the model that produced a saved conversation).
 * Centralized here so G4 (default model) and G6 (embedding model)
 * can use the same names without a third copy of this function.
 */
fun modelDisplayName(id: String): String {
    val parts = id.split(":", limit = 2)
    val provider = parts.getOrNull(0) ?: "?"
    val model = parts.getOrNull(1) ?: id
    return when (id) {
        "ollama:deepseek-v4-pro:cloud" -> "DeepSeek V4 Pro · $provider"
        "ollama:kimi-k2.7-code:cloud" -> "Kimi K2.7 · $provider"
        "anthropic:claude-sonnet-4-5" -> "Claude Sonnet 4.5 · $provider"
        else -> "$model · $provider"
    }
}
