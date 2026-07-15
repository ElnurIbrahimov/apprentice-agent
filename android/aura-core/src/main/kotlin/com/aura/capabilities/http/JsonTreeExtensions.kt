package com.aura.capabilities.http

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * JSON tree helpers used by every capability provider. All helpers are
 * extension functions on `JsonElement?` so they chain through Map.get
 * which returns `JsonElement?` (never null asserted, never unsafe).
 */
fun JsonElement?.asJsonObjectOrNull(): JsonObject? = this as? JsonObject
fun JsonElement?.asJsonArrayOrNull(): JsonArray? = this as? JsonArray
fun JsonElement?.asJsonPrimitiveOrNull(): JsonPrimitive? = this as? JsonPrimitive
fun JsonElement?.stringOrNull(): String? = (this as? JsonPrimitive)?.content
fun JsonElement?.intOrNull(): Int? = (this as? JsonPrimitive)?.content?.toIntOrNull()
fun JsonElement?.doubleOrNull(): Double? = (this as? JsonPrimitive)?.content?.toDoubleOrNull()
