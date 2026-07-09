package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.hands.HandRepository
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.Lazy
import javax.inject.Inject
import javax.inject.Singleton

/**
 * run_hand: look up a saved hand by name and execute it step-by-step.
 * Mirrors aura/tools/run_hand.py.
 * Risk: WRITE_LOCAL (runs tools that may write local state).
 *
 * Uses Lazy<ToolExecutor> to break the Dagger cycle: ToolRegistry provider
 * in ToolsModule depends on RunHandTool, and RunHandTool needs ToolExecutor,
 * which itself depends on ToolRegistry. Lazy defers resolution until execute()
 * time, so the graph can construct.
 */
@Singleton
class RunHandTool @Inject constructor(
    private val repository: HandRepository,
    private val executor: Lazy<ToolExecutor>,
) {
    fun definition() = ToolDefinition(
        name = "run_hand",
        description = "Execute a saved automation macro (hand) by its name. The hand runs its sequence of tool steps and returns the combined output.",
        parameters = ToolParameters(
            properties = mapOf(
                "name" to ToolProperty(type = "string", description = "The name of the hand to execute"),
            ),
            required = listOf("name"),
        ),
    )

    val tool = Tool(
        name = "run_hand",
        description = definition().description,
        risk = ToolRisk.WRITE_LOCAL,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val name = call.arguments["name"] as? String
                ?: return@Tool ToolResult.Error("missing 'name' argument", "bad_args")
            val hand = repository.getByName(name)
                ?: return@Tool ToolResult.Error("Hand not found: $name", "not_found")
            repository.run(hand, executor.get(), ctx)
        },
    category = "automation")
}
