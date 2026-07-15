package com.aura.providers

/**
 * Canonical user-facing label for a provider prefix.
 *
 * Single source of truth for "what does the user see when a model
 * id starts with `xai:`?" — the Chat header, the model picker,
 * the onboarding step, and the history row all consume this
 * function. Adding a new provider means adding its label here
 * (or letting the `else` branch capitalize the prefix).
 *
 * Hardcoded labels stay here ONLY when capitalization rules
 * would produce a wrong-looking name (e.g. `nvidia` → "NVIDIA",
 * not "Nvidia"). The `else` branch handles every new prefix
 * gracefully without touching this file.
 */
fun providerLabel(prefix: String): String = when (prefix) {
    "ollama" -> "Ollama"
    "anthropic" -> "Anthropic"
    "openai" -> "OpenAI"
    "deepseek" -> "DeepSeek"
    "gemini" -> "Gemini"
    "groq" -> "Groq"
    "openrouter" -> "OpenRouter"
    "nvidia" -> "NVIDIA"
    "moa" -> "MoA"
    "xai" -> "xAI Grok"
    "together" -> "Together AI"
    "cerebras" -> "Cerebras"
    "llama" -> "Meta Llama"
    "chatgpt" -> "ChatGPT"
    "agnes" -> "Agnes AI"
    "custom" -> "Custom Endpoint"
    "mistral" -> "Mistral AI"
    "brave" -> "Brave Search"
    "tavily" -> "Tavily Search"
    "firecrawl" -> "Firecrawl"
    "exa" -> "Exa Search"
    "jina" -> "Jina Reader"
    "elevenlabs" -> "ElevenLabs"
    "stability" -> "Stability AI"
    "kling" -> "Kling AI"
    "worldlabs" -> "World Labs"
    else -> prefix.replaceFirstChar { it.uppercase() }
}
