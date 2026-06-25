package com.aura.agent

import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import android.content.Context
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import dagger.hilt.android.qualifiers.ApplicationContext
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
) {
    suspend fun execute(name: String, argumentsJson: String, ctx: ToolContext): ToolResult {
        val tool = registry.get(name) ?: return ToolResult.Error("Unknown tool: $name", "unknown_tool")

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

        val call = ToolCall(id = "", name = name, arguments = args)
        return try {
            tool.execute(call, ctx)
        } catch (e: kotlinx.coroutines.CancellationException) {
            throw e
        } catch (e: Exception) {
            ToolResult.Error(e.message ?: "tool failed", "exception")
        }
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
