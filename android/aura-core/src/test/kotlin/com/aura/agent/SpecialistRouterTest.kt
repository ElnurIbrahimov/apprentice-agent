package com.aura.agent

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for [SpecialistRouter] and [Specialist] data class.
 */
class SpecialistRouterTest {

    // ------------------------------------------------------------------
    // Specialist data class basics
    // ------------------------------------------------------------------

    @Test
    fun `specialist has expected fields`() {
        val s = Specialist(
            name = "test",
            icon = "\uD83D\uDC40",
            systemPrompt = "Test prompt",
            toolsAllowed = setOf("tool_a"),
            suggestedModel = "model-x",
        )
        assertEquals("test", s.name)
        assertEquals("\uD83D\uDC40", s.icon)
        assertEquals("Test prompt", s.systemPrompt)
        assertEquals(setOf("tool_a"), s.toolsAllowed)
        assertEquals("model-x", s.suggestedModel)
    }

    @Test
    fun `specialist - suggested model is nullable`() {
        val s = Specialist(name = "a", icon = "\uD83D\uDC40", systemPrompt = "p")
        assertNull(s.suggestedModel)
        assertTrue(s.toolsAllowed.isEmpty())
    }

    @Test
    fun `all predefined specialists are available`() {
        assertEquals(7, Specialist.ALL.size)
        assertTrue(Specialist.ALL.any { it.name == "general" })
        assertTrue(Specialist.ALL.any { it.name == "coder" })
        assertTrue(Specialist.ALL.any { it.name == "researcher" })
        assertTrue(Specialist.ALL.any { it.name == "writer" })
        assertTrue(Specialist.ALL.any { it.name == "creative" })
        assertTrue(Specialist.ALL.any { it.name == "executive" })
        assertTrue(Specialist.ALL.any { it.name == "phone_native" })
    }

    @Test
    fun `byName lookup works`() {
        assertEquals("coder", Specialist.byName("coder")?.name)
        assertEquals("creative", Specialist.byName("creative")?.name)
        assertNull(Specialist.byName("nonexistent"))
    }

    @Test
    fun `general has no tool restriction`() {
        assertTrue(Specialist.General.toolsAllowed.isEmpty())
    }

    @Test
    fun `coder has expected tools`() {
        assertTrue(Specialist.Coder.toolsAllowed.contains("brave_search"))
        assertTrue(Specialist.Coder.toolsAllowed.contains("tavily_search"))
        assertTrue(Specialist.Coder.toolsAllowed.contains("fetch_url"))
    }

    @Test
    fun `researcher has expected tools`() {
        assertTrue(Specialist.Researcher.toolsAllowed.contains("deep_research"))
        assertTrue(Specialist.Researcher.toolsAllowed.contains("brave_search"))
        assertTrue(Specialist.Researcher.toolsAllowed.contains("tavily_search"))
    }

    @Test
    fun `creative has expected tools`() {
        assertTrue(Specialist.Creative.toolsAllowed.contains("image_gen"))
        assertEquals(setOf("image_gen"), Specialist.Creative.toolsAllowed)
    }

    @Test
    fun `executive has expected tools`() {
        assertTrue(Specialist.Executive.toolsAllowed.contains("calendar_read"))
        assertTrue(Specialist.Executive.toolsAllowed.contains("calendar_write"))
        assertTrue(Specialist.Executive.toolsAllowed.contains("contacts_search"))
        assertTrue(Specialist.Executive.toolsAllowed.contains("remember"))
        assertTrue(Specialist.Executive.toolsAllowed.contains("recall"))
    }

    @Test
    fun `phone_native has expected tools`() {
        assertTrue(Specialist.PhoneNative.toolsAllowed.contains("photo_library"))
        assertTrue(Specialist.PhoneNative.toolsAllowed.contains("location_now"))
        assertTrue(Specialist.PhoneNative.toolsAllowed.contains("set_reminder"))
    }

    // ------------------------------------------------------------------
    // Router – General fallback
    // ------------------------------------------------------------------

    @Test
    fun `general fallback for empty query`() {
        assertNull(SpecialistRouter.pickSpecialist(""))
        assertNull(SpecialistRouter.pickSpecialist("   "))
    }

    @Test
    fun `general fallback for casual phrases`() {
        assertNull(SpecialistRouter.pickSpecialist("hello"))
        assertNull(SpecialistRouter.pickSpecialist("how are you"))
        assertNull(SpecialistRouter.pickSpecialist("thanks"))
        assertNull(SpecialistRouter.pickSpecialist("good morning"))
    }

    // ------------------------------------------------------------------
    // Router – Coder
    // ------------------------------------------------------------------

    @Test
    fun `coder routing for code keywords`() {
        assertEquals("coder", SpecialistRouter.pickSpecialist("fix this code for me")?.name)
        assertEquals("coder", SpecialistRouter.pickSpecialist("debug my kotlin app")?.name)
        assertEquals("coder", SpecialistRouter.pickSpecialist("how do I write a python function")?.name)
        assertEquals("coder", SpecialistRouter.pickSpecialist("gradle build failed")?.name)
        assertEquals("coder", SpecialistRouter.pickSpecialist("review this pull request")?.name)
    }

    @Test
    fun `coder routing for language names`() {
        assertEquals("coder", SpecialistRouter.pickSpecialist("kotlin coroutines")?.name)
        assertEquals("coder", SpecialistRouter.pickSpecialist("python list comprehension")?.name)
        assertEquals("coder", SpecialistRouter.pickSpecialist("java stream API")?.name)
        assertEquals("coder", SpecialistRouter.pickSpecialist("javascript promises")?.name)
    }

    // ------------------------------------------------------------------
    // Router – Researcher
    // ------------------------------------------------------------------

    @Test
    fun `researcher routing for research keywords`() {
        assertEquals("researcher", SpecialistRouter.pickSpecialist("research quantum computing")?.name)
        assertEquals("researcher", SpecialistRouter.pickSpecialist("deep research on climate change")?.name)
        assertEquals("researcher", SpecialistRouter.pickSpecialist("web search for AI news")?.name)
        assertEquals("researcher", SpecialistRouter.pickSpecialist("find papers on cryptography")?.name)
        assertEquals("researcher", SpecialistRouter.pickSpecialist("look up the population of France")?.name)
    }

    @Test
    fun `researcher routing for who-what-questions`() {
        assertEquals("researcher", SpecialistRouter.pickSpecialist("who is the president")?.name)
        assertEquals("researcher", SpecialistRouter.pickSpecialist("what is photosynthesis")?.name)
        assertEquals("researcher", SpecialistRouter.pickSpecialist("tell me about black holes")?.name)
        assertEquals("researcher", SpecialistRouter.pickSpecialist("explain how neural networks work")?.name)
    }

    @Test
    fun `researcher routing with citations`() {
        assertEquals("researcher", SpecialistRouter.pickSpecialist("cite sources for that claim")?.name)
        assertEquals("researcher", SpecialistRouter.pickSpecialist("find documentation for OkHttp")?.name)
    }

    // ------------------------------------------------------------------
    // Router – Creative
    // ------------------------------------------------------------------

    @Test
    fun `creative routing for image keywords`() {
        assertEquals("creative", SpecialistRouter.pickSpecialist("generate an image of a sunset")?.name)
        assertEquals("creative", SpecialistRouter.pickSpecialist("draw a cat")?.name)
        assertEquals("creative", SpecialistRouter.pickSpecialist("create art in cyberpunk style")?.name)
        assertEquals("creative", SpecialistRouter.pickSpecialist("design a logo for my app")?.name)
        assertEquals("creative", SpecialistRouter.pickSpecialist("make a poster")?.name)
    }

    @Test
    fun `creative routing for vision keywords`() {
        assertEquals("creative", SpecialistRouter.pickSpecialist("what's in this photo")?.name)
        assertEquals("creative", SpecialistRouter.pickSpecialist("describe this picture")?.name)
    }

    // ------------------------------------------------------------------
    // Router – Executive
    // ------------------------------------------------------------------

    @Test
    fun `executive routing for calendar`() {
        assertEquals("executive", SpecialistRouter.pickSpecialist("schedule a meeting")?.name)
        assertEquals("executive", SpecialistRouter.pickSpecialist("create a calendar event")?.name)
        assertEquals("executive", SpecialistRouter.pickSpecialist("add appointment")?.name)
    }

    @Test
    fun `executive routing for contacts`() {
        assertEquals("executive", SpecialistRouter.pickSpecialist("find John's contact")?.name)
        assertEquals("executive", SpecialistRouter.pickSpecialist("search contacts")?.name)
    }

    @Test
    fun `executive routing for memory`() {
        assertEquals("executive", SpecialistRouter.pickSpecialist("remember my passport number")?.name)
        assertEquals("executive", SpecialistRouter.pickSpecialist("recall what I asked yesterday")?.name)
    }

    @Test
    fun `executive routing for tasks`() {
        assertEquals("executive", SpecialistRouter.pickSpecialist("add task to my todo list")?.name)
        assertEquals("executive", SpecialistRouter.pickSpecialist("what are my deadlines")?.name)
    }

    // ------------------------------------------------------------------
    // Router – PhoneNative
    // ------------------------------------------------------------------

    @Test
    fun `phone_native routing for camera and gallery`() {
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("take a photo")?.name)
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("open the camera")?.name)
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("show my gallery")?.name)
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("open gallery")?.name)
    }

    @Test
    fun `phone_native routing for location`() {
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("where am i")?.name)
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("my current location")?.name)
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("what's my GPS location")?.name)
    }

    @Test
    fun `phone_native routing for reminders`() {
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("set a reminder")?.name)
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("remind me to buy milk")?.name)
    }

    @Test
    fun `phone_native routing for device actions`() {
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("launch spotify")?.name)
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("open chrome")?.name)
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("check battery level")?.name)
        assertEquals("phone_native", SpecialistRouter.pickSpecialist("turn on wifi")?.name)
    }

    // ------------------------------------------------------------------
    // Router – Edge cases
    // ------------------------------------------------------------------

    @Test
    fun `code related query overrides phone_native`() {
        // "take a photo of my code" should route to Coder, not PhoneNative
        assertEquals("coder", SpecialistRouter.pickSpecialist("take a photo of my code")?.name)
    }

    @Test
    fun `research query not overridden by general keywords`() {
        assertEquals("researcher", SpecialistRouter.pickSpecialist("tell me about the history of photography")?.name)
    }

    // ------------------------------------------------------------------
    // Router – Writer
    // ------------------------------------------------------------------

    @Test
    fun `writer has expected tools`() {
        assertTrue(Specialist.Writer.toolsAllowed.contains("creative_read_project"))
        assertTrue(Specialist.Writer.toolsAllowed.contains("creative_add_world_item"))
    }

    @Test
    fun `writer routing for writing keywords`() {
        assertEquals("writer", SpecialistRouter.pickSpecialist("write me a novel opening")?.name)
        assertEquals("writer", SpecialistRouter.pickSpecialist("draft a story chapter")?.name)
        assertEquals("writer", SpecialistRouter.pickSpecialist("outline a screenplay act")?.name)
        assertEquals("writer", SpecialistRouter.pickSpecialist("help me with a magic system")?.name)
        assertEquals("writer", SpecialistRouter.pickSpecialist("build a world bible together")?.name)
    }

    @Test
    fun `case insensitive routing`() {
        assertEquals("coder", SpecialistRouter.pickSpecialist("KOTLIN")?.name)
        assertEquals("coder", SpecialistRouter.pickSpecialist("Python")?.name)
        assertEquals("researcher", SpecialistRouter.pickSpecialist("Research")?.name)
        assertEquals("creative", SpecialistRouter.pickSpecialist("IMAGE")?.name)
    }
}
