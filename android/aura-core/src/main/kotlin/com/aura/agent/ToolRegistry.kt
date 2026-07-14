package com.aura.agent

import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Risk level a tool carries. Drives permission gating and confirmation UX.
 * - READ_ONLY: no state mutation and no paid remote execution.
 * - REMOTE_COST: invokes a metered remote API but does not mutate state.
 * - WRITE_LOCAL: changes local state. Examples: reminders, app settings.
 * - WRITE_REMOTE: mutates state outside the device.
 * - PRIVACY: reads or transmits personal data.
 * - DESTRUCTIVE: irreversible.
 */
enum class ToolRisk { READ_ONLY, REMOTE_COST, WRITE_LOCAL, WRITE_REMOTE, PRIVACY, DESTRUCTIVE }

data class Tool(
    val name: String,
    val description: String,
    val risk: ToolRisk,
    val requiredPermissions: List<String> = emptyList(),
    val parameters: ToolParameters = ToolParameters(),
    val execute: suspend (ToolCall, ToolContext) -> ToolResult,
    /**
     * Top-level group the tool belongs to, used by the Tools
     * browser screen. Categories are stable strings — see
     * [com.aura.tools.ToolCategories]. Empty string = "other".
     */
    val category: String = "",
)

data class ToolCall(val id: String, val name: String, val arguments: Map<String, Any?>)

sealed class ToolResult {
    data class Ok(val output: String) : ToolResult()
    data class Error(
        val message: String,
        val code: String = "tool_error",
        val typedError: com.aura.core.error.AuraError? = null,
    ) : ToolResult()
    data class NeedsPermission(val permission: String, val rationale: String) : ToolResult()
    data class NeedsApproval(val rationale: String) : ToolResult()
}

data class ToolContext(
    val conversationId: String,
    val userId: String = "default",
    /** Latest user-authored message for tools that require explicit consent. */
    val userMessage: String = "",
    val permissions: Set<String> = emptySet(),
    val timeout: Long = 30_000L,
    /**
     * Tool names explicitly approved by a first-party UI action. This is
     * narrower than a boolean: approval for one metered tool cannot authorize
     * another tool in the same resumed automation.
     */
    val approvedRemoteCostTools: Set<String> = emptySet(),
    /**
     * Session-level write flag. When false, the tool executor refuses to
     * run tools whose risk >= WRITE_LOCAL — this is the privacy boundary
     * used by the incognito toggle. READ_ONLY tools (e.g. recall, web_search)
     * still run so the user can keep using the assistant without writing
     * anything to local state.
     */
    val memoryEnabled: Boolean = true,
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
        ToolDefinition(
            name = t.name,
            description = t.description,
            parameters = t.parameters,
            category = t.category,
        )
    }
}
