package com.aura.skills

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests the [Skill] envelope's JSON serialization round-trip and validation
 * guards (the `init` requirements).
 */
class SkillsSerializationTest {

    @Test
    fun `encode and decode round trip`() {
        val original = listOf(
            Skill(name = "review", description = "PR review", body = "# Review\n- check tests"),
            Skill(name = "summarize", description = "", body = "Summarize the conversation."),
        )
        val encoded = original.encodeToJsonString()
        val decoded = encoded.decodeAsSkillList()
        assertEquals(original.size, decoded.size)
        for ((a, b) in original.zip(decoded)) {
            assertEquals(a.id, b.id)
            assertEquals(a.name, b.name)
            assertEquals(a.body, b.body)
        }
    }

    @Test
    fun `decode on blank returns empty`() {
        assertEquals(emptyList<Skill>(), "".decodeAsSkillList())
        assertEquals(emptyList<Skill>(), "   ".decodeAsSkillList())
    }

    @Test
    fun `decode on malformed JSON returns empty (does not crash)`() {
        assertEquals(emptyList<Skill>(), "{not json".decodeAsSkillList())
        assertEquals(emptyList<Skill>(), "{\"unexpected\":true}".decodeAsSkillList())
    }

    @Test
    fun `skill init rejects blank name`() {
        try {
            Skill(name = "  ", description = "x", body = "y")
            assertTrue("Expected IllegalArgumentException for blank name", false)
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("name"))
        }
    }

    @Test
    fun `skill init rejects too-long description`() {
        try {
            Skill(name = "ok", description = "x".repeat(241), body = "y")
            assertTrue("Expected IllegalArgumentException for long description", false)
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("description"))
        }
    }

    @Test
    fun `preview returns first non-heading line`() {
        val skill = Skill(name = "x", description = "", body = "# Title\n\nReal preview line.\nMore text.")
        assertEquals("Real preview line.", skill.preview())
    }

    @Test
    fun `preview truncates long lines`() {
        val long = "a".repeat(300)
        val skill = Skill(name = "x", description = "", body = long)
        assertTrue(skill.preview().length == 140)
    }

    @Test
    fun `preview returns placeholder for empty body`() {
        val skill = Skill(name = "x", description = "", body = "")
        assertEquals("(empty)", skill.preview())
    }
}
