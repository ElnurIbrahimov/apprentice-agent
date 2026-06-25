package com.aura.providers

import javax.inject.Inject
import javax.inject.Singleton

/**
 * Task-aware model routing. Mirrors aura/core/router.py + aura/routing/.
 * - Simple chat: cheapest configured model
 * - Tool-heavy: model with best tool-call accuracy
 * - Vision: multimodal model
 * - Long context: largest context window
 * - Code: code-tuned model
 */
@Singleton
class ProviderRouter @Inject constructor(
    private val registry: ProviderRegistry,
) {
    enum class Task { CHAT, TOOL_HEAVY, VISION, LONG_CONTEXT, CODE, CHEAP }

    data class RoutingRule(
        val task: Task,
        val preferred: List<String>, // model IDs in priority order
        val fallback: String, // last-resort cloud model
    )

    private val rules: List<RoutingRule> = listOf(
        RoutingRule(Task.CHEAP, listOf("local:gemma3-1b-q4", "ollama:nemotron-3-super:cloud"), "ollama:deepseek-v3.2:cloud"),
        RoutingRule(Task.CHAT, listOf("ollama:deepseek-v3.2:cloud", "anthropic:claude-sonnet-4-5", "openai:gpt-5.2"), "ollama:deepseek-v3.2:cloud"),
        RoutingRule(Task.TOOL_HEAVY, listOf("anthropic:claude-sonnet-4-5", "ollama:kimi-k2.6:cloud", "openai:gpt-5.2"), "ollama:deepseek-v3.2:cloud"),
        RoutingRule(Task.VISION, listOf("ollama:kimi-k2.6:cloud", "anthropic:claude-sonnet-4-5", "openai:gpt-5.2"), "ollama:kimi-k2.6:cloud"),
        RoutingRule(Task.LONG_CONTEXT, listOf("ollama:qwen3.5:cloud", "anthropic:claude-sonnet-4-5", "ollama:minimax-m2.7:cloud"), "ollama:qwen3.5:cloud"),
        RoutingRule(Task.CODE, listOf("ollama:qwen3-coder:480b-cloud", "ollama:minimax-m2.7:cloud", "anthropic:claude-sonnet-4-5"), "ollama:qwen3-coder:480b-cloud"),
    )

    fun route(task: Task): String {
        val configured = registry.configured().map { it.prefix }.toSet()
        val rule = rules.firstOrNull { it.task == task } ?: rules.first { it.task == Task.CHAT }
        // First preferred model whose provider is configured
        return rule.preferred.firstOrNull { id ->
            val providerPrefix = id.substringBefore(":")
            providerPrefix in configured
        } ?: rule.fallback
    }
}
