package com.aura.memory

import com.aura.providers.ChatOptions
import com.aura.providers.ProviderMessage
import com.aura.providers.ProviderRegistry
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * LLM-augmented write gate. Wraps the heuristic [WriteGate] with a
 * cloud LLM call that makes the final "is this worth remembering?"
 * decision and produces a better category + importance score.
 *
 * Flow:
 * 1. Heuristic gate runs first (fast, no network). If it says
 *    "don't store" (empty, too short, system msg), we skip the LLM
 *    call entirely — saves a cloud round-trip on garbage.
 * 2. If the heuristic says "store", we ask the LLM: "Is this worth
 *    remembering? Reply with a JSON object."
 * 3. If the LLM says NO, we skip. If YES, we store with the LLM's
 *    category + importance.
 *
 * The LLM call is best-effort: if it fails (network error, bad JSON,
 * timeout), we fall back to the heuristic decision so the memory
 * is still stored with the heuristic classification. This means
 * the LLM gate only *improves* the decision — it never blocks
 * storage on failure.
 *
 * @param heuristic The fast pre-filter gate (keyword-based)
 * @param registry Provider registry for the LLM call
 * @param modelId The model to use for the gate (should be a fast/cheap model)
 */
class LlmWriteGate(
    private val heuristic: WriteGate,
    private val registry: ProviderRegistry,
    private val modelId: String,
) {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    suspend fun evaluate(content: String, source: String): WriteGate.Decision {
        // 1) Heuristic pre-filter — fast, no network
        val heuristicDecision = heuristic.evaluate(content, source)
        if (!heuristicDecision.shouldStore) return heuristicDecision

        // 2) LLM gate — best-effort, falls back to heuristic on failure
        val llmDecision = runCatching {
            llmEvaluate(content)
        }.getOrNull()

        // 3) Merge: LLM decision wins if it parsed, heuristic is the fallback
        return llmDecision ?: heuristicDecision
    }

    private suspend fun llmEvaluate(content: String): WriteGate.Decision? {
        val systemPrompt = """
            You are a memory gate for a personal AI assistant. Decide if the user's
            message is worth remembering long-term. Personal facts, preferences,
            and important information should be stored. Casual chatter, questions,
            and transient requests should not.

            Reply with a JSON object:
            {"store": true/false, "category": "fact|preference|person|episode|idea|task", "importance": 0.0-1.0}

            Be conservative — only store things the user would want recalled in
            future conversations. A question like "what's the weather?" is NOT
            worth storing. "I prefer dark mode" IS worth storing.
        """.trimIndent()

        val messages = listOf(
            ProviderMessage(role = ProviderMessage.Role.system, content = systemPrompt),
            ProviderMessage(role = ProviderMessage.Role.user, content = content),
        )

        val text = StringBuilder()
        try {
            registry.chat(modelId, messages, ChatOptions(temperature = 0.1, maxTokens = 100), emptyList())
                .collect { chunk ->
                    chunk.text?.let { text.append(it) }
                }
        } catch (_: Exception) {
            return null
        }

        val raw = text.toString().trim()
        if (raw.isEmpty()) return null

        // Try to extract JSON from the response (models sometimes wrap in markdown)
        val jsonStr = extractJson(raw) ?: return null

        return try {
            val parsed = json.parseToJsonElement(jsonStr).jsonObject
            val shouldStore = parsed["store"]?.jsonPrimitive?.contentOrNull?.lowercase() == "true"
            if (!shouldStore) {
                WriteGate.Decision(shouldStore = false, reason = "llm_rejected")
            } else {
                val category = parsed["category"]?.jsonPrimitive?.contentOrNull ?: "fact"
                val importance = parsed["importance"]?.jsonPrimitive?.doubleOrNull?.toFloat() ?: 0.5f
                WriteGate.Decision(
                    shouldStore = true,
                    category = category,
                    importance = importance.coerceIn(0f, 1f),
                    reason = "llm_classified",
                )
            }
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Extract a JSON object from a response that may be wrapped in
     * markdown code fences or have extra text around it.
     */
    private fun extractJson(text: String): String? {
        // Try direct parse first
        if (text.startsWith("{")) return text
        // Try to find JSON in code fences: ```json ... ```
        val fenceMatch = Regex("```(?:json)?\\s*(\\{.*?})\\s*```", RegexOption.DOT_MATCHES_ALL)
            .find(text)
        if (fenceMatch != null) return fenceMatch.groupValues[1]
        // Try to find a bare JSON object
        val bareMatch = Regex("\\{(.*?)}", RegexOption.DOT_MATCHES_ALL).find(text)
        if (bareMatch != null) return bareMatch.value
        return null
    }
}