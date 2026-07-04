package com.aura.hands

/**
 * Parsed step of a [Hand]. Keeping this in aura-core so that
 * both the repository (execution) and the app UI can build / edit
 * steps without duplicating JSON knowledge.
 */
data class HandStep(
    val tool: String,
    val args: Map<String, String>,
) {
    /**
     * Serialize to the JSON shape the repository expects:
     * {"tool":"name","args":{"k":"v"}}
     */
    fun toJsonObject(): kotlinx.serialization.json.JsonObject {
        val argsJson = args.mapValues { (_, v) ->
            kotlinx.serialization.json.JsonPrimitive(v)
        }
        return kotlinx.serialization.json.JsonObject(mapOf(
            "tool" to kotlinx.serialization.json.JsonPrimitive(tool),
            "args" to kotlinx.serialization.json.JsonObject(argsJson),
        ))
    }
}
