package com.aura.hands

/**
 * Extension to build a JSON argument string from a plain string map.
 * Only string values are supported by this simplified mapper; values
 * that look numeric/boolean are still emitted as JSON strings to
 * keep the hand editor simple.
 */
fun Map<String, String>.toJsonString(): String {
    if (isEmpty()) return "{}"
    val entries = entries.joinToString(",") { (k, v) ->
        val escaped = v.replace("\\", "\\\\").replace("\"", "\\\"")
        "\"$k\":\"$escaped\""
    }
    return "{$entries}"
}
