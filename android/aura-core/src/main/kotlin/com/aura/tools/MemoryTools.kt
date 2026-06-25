package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.memory.MemoryEntity
import com.aura.memory.MemoryStore
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import javax.inject.Inject
import javax.inject.Singleton

/**
 * 'remember' tool — let the model explicitly store a fact in long-term memory.
 * Risk: WRITE_LOCAL (changes the memory DB).
 */
@Singleton
class RememberTool @Inject constructor(
    private val memoryStore: MemoryStore,
) {
    fun definition() = ToolDefinition(
        name = "remember",
        description = "Store a fact in long-term memory. Use for user preferences, important context, things to recall later. Examples: 'user prefers dark mode', 'user is allergic to peanuts', 'project codename is Aurora'.",
        parameters = ToolParameters(
            properties = mapOf(
                "fact" to ToolProperty(type = "string", description = "The fact to remember. Be concise and specific."),
                "category" to ToolProperty(type = "string", description = "Optional: fact/preference/person/project/idea/task. Defaults to 'fact'."),
            ),
            required = listOf("fact"),
        ),
    )

    val tool = Tool(
        name = "remember",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val fact = call.arguments["fact"] as? String ?: return@Tool ToolResult.Error("missing 'fact'", "bad_args")
            val category = call.arguments["category"] as? String ?: "fact"
            try {
                val id = memoryStore.store(fact, source = "user", category = category, importance = 0.7f)
                ToolResult.Ok("Remembered: $fact (id $id)")
            } catch (e: Exception) {
                ToolResult.Error("remember failed: ${e.message}", "exception")
            }
        },
    )
}

/**
 * 'recall' tool — let the model explicitly search memory.
 * Risk: READ_ONLY.
 */
@Singleton
class RecallTool @Inject constructor(
    private val memoryStore: MemoryStore,
) {
    fun definition() = ToolDefinition(
        name = "recall",
        description = "Search long-term memory for facts matching a query. Returns up to 5 most relevant memories.",
        parameters = ToolParameters(
            properties = mapOf(
                "query" to ToolProperty(type = "string", description = "What to search for"),
                "limit" to ToolProperty(type = "integer", description = "Max results (default 5, max 20)"),
            ),
            required = listOf("query"),
        ),
    )

    val tool = Tool(
        name = "recall",
        description = definition().description,
        risk = ToolRisk.READ_ONLY,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val query = call.arguments["query"] as? String ?: return@Tool ToolResult.Error("missing 'query'", "bad_args")
            val limit = (call.arguments["limit"] as? Int ?: 5).coerceIn(1, 20)
            try {
                val hits = memoryStore.query(query, limit)
                if (hits.isEmpty()) {
                    ToolResult.Ok("No memories found for: $query")
                } else {
                    val text = hits.mapIndexed { i, m ->
                        "${i + 1}. [${m.category}] ${m.content}"
                    }.joinToString("\n")
                    ToolResult.Ok(text)
                }
            } catch (e: Exception) {
                ToolResult.Error("recall failed: ${e.message}", "exception")
            }
        },
    )
}
