package com.aura.capabilities

import com.aura.capabilities.http.asJsonObjectOrNull
import com.aura.capabilities.http.stringOrNull
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Verifies the safe-JSON-tree helpers used by every capability provider.
 * These helpers turn Kotlin's "every JsonElement is nullable, every Map.get
 * returns null" into a chainable, non-`!!` API.
 */
class JsonTreeExtensionsTest {

    @Test
    fun `stringOrNull returns content for string primitives`() {
        val obj = Json.parseToJsonElement("""{"a":"hello"}""")
        val a = (obj.asJsonObjectOrNull() ?: error("not object"))["a"]
        assertEquals("hello", a.stringOrNull())
    }

    @Test
    fun `stringOrNull returns null for missing key`() {
        val obj = Json.parseToJsonElement("""{}""")
        val missing = (obj.asJsonObjectOrNull() ?: error("not object"))["missing"]
        assertNull(missing.stringOrNull())
    }

    @Test
    fun `stringOrNull returns content of any primitive including numbers`() {
        // JsonPrimitive.content returns the raw string form for all primitive
        // types, so stringOrNull is a safe "give me a string" accessor.
        val obj = Json.parseToJsonElement("""{"a":42,"b":true,"c":null}""")
        val a = (obj.asJsonObjectOrNull() ?: error("not object"))["a"]
        val b = (obj.asJsonObjectOrNull() ?: error("not object"))["b"]
        assertEquals("42", a.stringOrNull())
        assertEquals("true", b.stringOrNull())
    }

    @Test
    fun `stringOrNull chains through null receiver without crash`() {
        val nullable: kotlinx.serialization.json.JsonElement? = null
        assertNull(nullable.stringOrNull())
    }

    @Test
    fun `asJsonObjectOrNull navigates nested objects`() {
        val obj = Json.parseToJsonElement("""{"data":{"id":"x"}}""")
        val data = (obj.asJsonObjectOrNull() ?: error("not object"))["data"].asJsonObjectOrNull() ?: error("not data")
        assertEquals("x", data["id"].stringOrNull())
    }
}
