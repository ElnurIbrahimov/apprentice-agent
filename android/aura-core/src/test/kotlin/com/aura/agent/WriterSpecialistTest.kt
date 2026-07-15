package com.aura.agent

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class WriterSpecialistTest {
    @Test
    fun `writer specialist exposes creative project tools without a model override`() {
        val writer = Specialist.Writer
        assertEquals("writer", writer.name)
        assertTrue("creative_read_project" in writer.toolsAllowed)
        assertTrue("creative_add_world_item" in writer.toolsAllowed)
        assertEquals(null, writer.suggestedModel)
        assertTrue(writer in Specialist.ALL)
    }

    @Test
    fun `router selects writer for fiction and worldbuilding`() {
        assertEquals(Specialist.Writer, SpecialistRouter.pickSpecialist("Help me write the next chapter of my novel"))
        assertEquals(Specialist.Writer, SpecialistRouter.pickSpecialist("Worldbuild a magic system with strict costs"))
        assertEquals(Specialist.Writer, SpecialistRouter.pickSpecialist("Run a what-if scenario for my protagonist"))
    }

    @Test
    fun `writer does not steal coding or visual art requests`() {
        assertEquals(Specialist.Coder, SpecialistRouter.pickSpecialist("Write Kotlin code for a parser"))
        assertEquals(Specialist.Creative, SpecialistRouter.pickSpecialist("Draw a fantasy poster"))
    }
}