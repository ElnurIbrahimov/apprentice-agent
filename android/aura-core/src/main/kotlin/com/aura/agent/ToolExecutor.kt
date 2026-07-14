package com.aura.agent

import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import com.aura.usage.UsageTracker
import android.content.Context
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.runInterruptible
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.longOrNull
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Dispatches tool calls. Wraps ToolRegistry with permission checks and JSON parsing.
 * Mirrors aura/core/tool_executor.py.
 */
@Singleton
class ToolExecutor @Inject constructor(
    private val registry: ToolRegistry,
    @ApplicationContext private val context: Context,
    private val usageTracker: UsageTracker = UsageTracker(),
) {
    private val remoteCostApprovalGate = RemoteCostApprovalGate()
    suspend fun execute(name: String, argumentsJson: String, ctx: ToolContext): ToolResult {
        val tool = registry.get(name) ?: return ToolResult.Error("Unknown tool: $name", "unknown_tool")

        // Privacy boundary: in an incognito session, refuse any tool that
        // mutates local state. READ_ONLY tools (recall, web_search, kg_query)
        // are still allowed so the user can keep asking questions.
        if (!ctx.memoryEnabled && tool.risk.ordinal >= ToolRisk.WRITE_LOCAL.ordinal) {
            return ToolResult.Error(
                message = "Tool '$name' is disabled in incognito mode (would write to local state).",
                code = "incognito_blocked",
            )
        }

        // Permission gate. We resolve against the live PackageManager state every
        // call so the model sees the freshest answer (user may have just granted
        // the permission via Settings).
        for (perm in tool.requiredPermissions) {
            if (!isGranted(perm)) {
                return ToolResult.NeedsPermission(perm, "Tool $name requires $perm")
            }
        }

        val args = try { parseArgs(argumentsJson, tool.parameters) } catch (e: Exception) {
            return ToolResult.Error("Bad arguments: ${e.message}", "bad_args")
        }

        if (tool.risk == ToolRisk.REMOTE_COST) {
            remoteCostApprovalGate.authorize(name, args, ctx)?.let { rationale ->
                return ToolResult.NeedsApproval(rationale)
            }
        }

        val call = ToolCall(id = "", name = name, arguments = args)
        val result = try {
            withTimeout(ctx.timeout) {
                // Tools may bridge suspend APIs and legacy blocking Android/HTTP
                // calls. The interruptible IO boundary keeps both kinds off Main
                // and lets cancellation preempt a blocking call promptly.
                runInterruptible(Dispatchers.IO) {
                    runBlocking { tool.execute(call, ctx) }
                }
            }
        } catch (e: TimeoutCancellationException) {
            ToolResult.Error(
                message = "Tool '$name' timed out after ${ctx.timeout / 1000}s",
                code = "tool_timeout",
            )
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            ToolResult.Error(e.message ?: "tool failed", "exception")
        }
        if (result is ToolResult.Ok) usageTracker.recordToolResult(result.output.length)
        return result
    }

    private fun isGranted(permission: String): Boolean = try {
        ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
    } catch (e: Exception) {
        // Defensive: some permissions (e.g. BIND_NOTIFICATION_LISTENER_SERVICE) are
        // not regular Android permissions and checkSelfPermission may throw.
        // We treat that as "not granted" so the model gets a NeedsPermission
        // result and can explain the situation to the user.
        false
    }

    /**
     * Parse JSON args against a ToolParameters schema. Coerces types loosely
     * (string→int, etc.) so models that get the type wrong still work.
     */
    private fun parseArgs(json: String, schema: ToolParameters): Map<String, Any?> {
        val obj = if (json.isBlank()) JsonObject(emptyMap()) else Json.parseToJsonElement(json).jsonObject
        val out = mutableMapOf<String, Any?>()
        for ((k, prop) in schema.properties) {
            val v = obj[k] ?: continue
            out[k] = coerce(v, prop)
        }
        return out
    }

    private fun coerce(v: kotlinx.serialization.json.JsonElement, prop: ToolProperty): Any? = when {
        v is JsonPrimitive && prop.type == "string" -> v.contentOrNull
        v is JsonPrimitive && prop.type == "integer" -> v.intOrNull
        v is JsonPrimitive && prop.type == "number" -> v.doubleOrNull
        v is JsonPrimitive && prop.type == "boolean" -> v.booleanOrNull
        v is JsonPrimitive && prop.type == "any" -> v.contentOrNull
        v is JsonArray && prop.type == "array" -> v.map { coerce(it, ToolProperty(type = "any")) }
        v is JsonObject && prop.type == "object" -> v.mapValues { coerce(it.value, ToolProperty(type = "any")) }
        v is JsonPrimitive -> v.contentOrNull
        else -> v.toString()
    }
}

/**
 * One-shot approval gate for metered API calls. The model cannot approve its
 * own call by repeating it in the same turn: the exact parsed arguments must
 * be requested again after a later, explicitly affirmative user message.
 */
internal class RemoteCostApprovalGate {
    private data class Key(val conversationId: String, val toolName: String)
    private data class Pending(
        val arguments: Map<String, Any?>,
        val requestingMessage: String,
    )

    private val pending = mutableMapOf<Key, Pending>()

    @Synchronized
    fun authorize(
        toolName: String,
        arguments: Map<String, Any?>,
        context: ToolContext,
    ): String? {
        val key = Key(context.conversationId, toolName)
        val existing = pending[key]
        if (existing == null || existing.arguments != arguments) {
            pending[key] = Pending(arguments.toMap(), context.userMessage)
            return rationale(toolName)
        }

        val isLaterTurn = context.userMessage.isNotBlank() &&
            context.userMessage != existing.requestingMessage
        if (!isLaterTurn || !isExplicitApproval(context.userMessage)) {
            return rationale(toolName)
        }

        pending.remove(key)
        return null
    }

    private fun rationale(toolName: String): String =
        "Running '$toolName' may consume paid API credits. Reply with an explicit confirmation to continue."

    private fun isExplicitApproval(message: String): Boolean {
        val normalized = message
            .lowercase(Locale.US)
            .replace(Regex("[^a-z0-9\\s]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
        return normalized in CONFIRMATIONS
    }

    private companion object {
        val CONFIRMATIONS = setOf(
            "yes", "yes please", "yes confirm", "confirm", "confirmed",
            "go ahead", "do it", "continue", "approve", "approved",
        )
    }
}
