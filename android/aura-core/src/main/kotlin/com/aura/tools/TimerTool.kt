package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolCall
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Timer / stopwatch tool. The agent can start a timer, check how much
 * time has elapsed, and stop it. Timers are in-memory only (not persisted)
 * — they reset on app restart.
 *
 * Risk: READ_ONLY (no phone permissions, just in-memory state).
 */
@Singleton
class TimerTool @Inject constructor() {

    private val timers = ConcurrentHashMap<String, Long>()

    fun definition() = ToolDefinition(
        name = "timer",
        description = "Start, check, or stop a timer. Actions: 'start' (returns a timer ID), 'check' (returns elapsed seconds), 'stop' (returns final elapsed seconds and removes the timer).",
        parameters = ToolParameters(
            properties = mapOf(
                "action" to ToolProperty(
                    type = "string",
                    description = "One of: start, check, stop",
                ),
                "timer_id" to ToolProperty(
                    type = "string",
                    description = "Timer ID (required for check and stop, ignored for start)",
                ),
            ),
            required = listOf("action"),
        ),
    )

    val tool = Tool(
        name = "timer",
        description = definition().description,
        risk = ToolRisk.READ_ONLY,
        parameters = definition().parameters,
        execute = { call, _ ->
            val action = call.arguments["action"] as? String
                ?: return@Tool ToolResult.Error("missing 'action' argument", "bad_args")
            when (action.lowercase()) {
                "start" -> {
                    val id = UUID.randomUUID().toString()
                    timers[id] = System.currentTimeMillis()
                    ToolResult.Ok("Timer started. ID: $id")
                }
                "check" -> {
                    val id = call.arguments["timer_id"] as? String
                        ?: return@Tool ToolResult.Error("missing 'timer_id' for check action", "bad_args")
                    val start = timers[id]
                        ?: return@Tool ToolResult.Error("No timer with ID $id", "not_found")
                    val elapsedSec = (System.currentTimeMillis() - start) / 1000.0
                    ToolResult.Ok("Elapsed: %.1f seconds".format(elapsedSec))
                }
                "stop" -> {
                    val id = call.arguments["timer_id"] as? String
                        ?: return@Tool ToolResult.Error("missing 'timer_id' for stop action", "bad_args")
                    val start = timers.remove(id)
                        ?: return@Tool ToolResult.Error("No timer with ID $id", "not_found")
                    val elapsedSec = (System.currentTimeMillis() - start) / 1000.0
                    ToolResult.Ok("Timer stopped. Total elapsed: %.1f seconds".format(elapsedSec))
                }
                else -> ToolResult.Error("Unknown action '$action'. Use: start, check, stop", "bad_args")
            }
        },
    category = "productivity")
}