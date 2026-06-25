package com.aura.agent

import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Risk level a tool carries. Drives permission gating and confirmation UX.
 * - READ_ONLY: never destructive. Examples: web_search, calendar_read.
 * - WRITE_LOCAL: changes local state. Examples: calendar_write, app_launcher.
 * - WRITE_REMOTE: makes a network call that mutates. Examples: email send.
 * - PRIVACY: touches personal data. Examples: contacts_get, location_now.
 * - DESTRUCTIVE: irreversible. Examples: file_pick + delete, factory_reset (don't add this).
 */
enum class ToolRisk { READ_ONLY, WRITE_LOCAL, WRITE_REMOTE, PRIVACY, DESTRUCTIVE }

data class Tool(
    val name: String,
    val description: String,
    val risk: ToolRisk,
    val requiredPermissions: List<String> = emptyList(),
    val parameters: ToolParameters = ToolParameters(),
    val execute: suspend (ToolCall, ToolContext) -> ToolResult,
)

data class ToolCall(val id: String, val name: String, val arguments: Map<String, Any?>)

sealed class ToolResult {
    data class Ok(val output: String) : ToolResult()
    data class Error(val message: String, val code: String = "tool_error") : ToolResult()
    data class NeedsPermission(val permission: String, val rationale: String) : ToolResult()
    data class NeedsApproval(val rationale: String) : ToolResult()
}

data class ToolContext(
    val conversationId: String,
    val userId: String = "default",
    val permissions: Set<String> = emptySet(),
    val timeout: Long = 30_000L,
)

/**
 * Holds all tools. The agentic loop looks them up by name and dispatches.
 * Mirrors aura/core/tool_executor.py + aura/toolsets.py.
 */
@Singleton
class ToolRegistry @Inject constructor() {
    private val tools: MutableMap<String, Tool> = mutableMapOf()

    fun register(tool: Tool) { tools[tool.name] = tool }
    fun unregister(name: String) { tools.remove(name) }
    fun get(name: String): Tool? = tools[name]
    fun all(): List<Tool> = tools.values.toList()
    fun names(): List<String> = tools.keys.toList()
    fun byRisk(min: ToolRisk): List<Tool> = tools.values.filter { it.risk.ordinal >= min.ordinal }

    fun definitions(): List<ToolDefinition> = tools.values.map { t ->
        ToolDefinition(name = t.name, description = t.description, parameters = t.parameters)
    }
}
