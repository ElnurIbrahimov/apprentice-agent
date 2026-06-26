package com.aura.hands

import com.aura.agent.ToolContext
import com.aura.agent.ToolExecutor
import com.aura.agent.ToolResult
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.contentOrNull
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Handles CRUD operations for hands and executes them step-by-step.
 */
@Singleton
class HandRepository @Inject constructor(
    private val dao: HandDao,
) {
    private val json = Json { ignoreUnknownKeys = true }

    suspend fun getAll(): List<Hand> = dao.getAll()

    suspend fun getByName(name: String): Hand? = dao.getByName(name)

    suspend fun insert(hand: Hand) = dao.insert(hand)

    suspend fun update(hand: Hand) = dao.update(hand)

    suspend fun deleteByName(name: String) = dao.deleteByName(name)

    /**
     * Execute a hand step-by-step via the ToolExecutor.
     * Returns a concatenated summary of each step's result.
     */
    suspend fun run(
        hand: Hand,
        executor: ToolExecutor,
        ctx: ToolContext,
    ): ToolResult {
        if (!hand.enabled) {
            return ToolResult.Error("Hand '${hand.name}' is disabled", "hand_disabled")
        }
        val steps = parseSteps(hand.steps)
        if (steps.isEmpty()) {
            return ToolResult.Ok("No steps defined for hand '${hand.name}'")
        }
        val outputs = mutableListOf<String>()
        for ((i, step) in steps.withIndex()) {
            val toolName = step["tool"] ?: return ToolResult.Error(
                "Step ${i + 1}: missing 'tool' in $step", "bad_step"
            )
            val args = step["args"] ?: "{}"
            val result = executor.execute(toolName, args, ctx)
            when (result) {
                is ToolResult.Ok -> outputs.add("Step ${i + 1} ($toolName): ${result.output}")
                is ToolResult.Error -> return ToolResult.Error(
                    "Step ${i + 1} ($toolName) failed: ${result.message}", result.code
                )
                is ToolResult.NeedsPermission -> return result
                is ToolResult.NeedsApproval -> return result
            }
        }
        return ToolResult.Ok(
            "Hand '${hand.name}' completed.\n${outputs.joinToString("\n")}"
        )
    }

    private fun parseSteps(stepsJson: String): List<Map<String, String>> {
        val element = try {
            json.parseToJsonElement(stepsJson)
        } catch (_: Exception) {
            return emptyList()
        }
        val arr = element.jsonArray
        return arr.mapNotNull { elem ->
            val obj = elem.jsonObject
            val tool = obj["tool"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
            val args = when (val a = obj["args"]) {
                null -> "{}"
                else -> (a as? kotlinx.serialization.json.JsonPrimitive)?.contentOrNull ?: a.toString()
            }
            mapOf("tool" to tool, "args" to args)
        }
    }
}
