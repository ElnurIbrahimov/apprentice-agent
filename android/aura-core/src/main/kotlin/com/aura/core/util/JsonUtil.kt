package com.aura.core.util

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Minimal JSON escaping helper for Kotlin source code that can't rely on
 * kotlinx.serialization (e.g. when building dynamic payloads with unknown
 * keys). Prefer `Json.encodeToString` for typed values; this is a safety
 * fallback for string interpolation paths.
 */
fun escapeJsonString(s: String): String = s
    .replace("\\", "\\\\")
    .replace("\"", "\\\"")
    .replace("\n", "\\n")
    .replace("\r", "\\r")
    .replace("\t", "\\t")

/**
 * Build a Firecrawl scrape request body safely. Centralizes the JSON shape
 * so callers don't hand-roll JSON strings.
 */
fun buildFirecrawlBody(url: String, formats: List<String> = listOf("markdown")): String {
    return buildJsonObject {
        put("url", url)
        put("formats", JsonArray(formats.map { JsonPrimitive(it) }))
    }.toString()
}
