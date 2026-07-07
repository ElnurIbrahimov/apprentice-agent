package com.aura.ui.util

/**
 * User-facing label for a `prefix:model` provider id, e.g.
 * `\"ollama:deepseek-v4-pro:cloud\"` → `\"DeepSeek V4 Pro · Ollama\"`.
 *
 * Shared between the Chat screen header (current model) and the
 * History screen (the model that produced a saved conversation).
 * Centralized here so the chat header, history rows, and model picker
 * all show the same name.
 *
 * Derives everything from the model ID — no hardcoded model lists
 * that go stale when providers add new models.
 */
fun modelDisplayName(id: String): String {
    val parts = id.split(":", limit = 2)
    val provider = parts.getOrNull(0) ?: "?"
    val model = parts.getOrNull(1) ?: id

    // Strip common suffixes
    val clean = model.replace(Regex(":cloud$|:latest$|:free$"), "")

    // Convert kebab/snake to Title Case
    val displayName = clean
        .replace("-", " ")
        .replace("_", " ")
        .split(" ")
        .joinToString(" ") { word ->
            if (word.any { it.isDigit() } || word.length <= 2) word.uppercase()
            else word.replaceFirstChar { it.uppercase() }
        }

    val providerLabel = when (provider) {
        "ollama" -> "Ollama"
        "anthropic" -> "Anthropic"
        "openai" -> "OpenAI"
        "deepseek" -> "DeepSeek"
        "gemini" -> "Gemini"
        "groq" -> "Groq"
        "openrouter" -> "OpenRouter"
        "nvidia" -> "NVIDIA"
        "moa" -> "MoA"
        else -> provider.replaceFirstChar { it.uppercase() }
    }

    return "$displayName · $providerLabel"
}