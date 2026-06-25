package com.aura.memory

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class WriteGateTest {

    @Test
    fun `person category for family references`() {
        val gate = WriteGate()
        val d = gate.evaluate("Call mom tomorrow at 5", "user")
        assertEquals("person", d.category)
    }

    @Test
    fun `task category for reminders`() {
        val gate = WriteGate()
        val d = gate.evaluate("remind me to take the trash out at 8pm", "user")
        assertEquals("task", d.category)
    }

    @Test
    fun `idea category for brainstorming`() {
        val gate = WriteGate()
        val d = gate.evaluate("what if we added dark mode to the app", "user")
        assertEquals("idea", d.category)
    }
}
