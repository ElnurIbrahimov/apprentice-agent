package com.aura.ui.components

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Lock the [DefaultQuickChips] list. We don't want the empty
 * state to silently grow a 6th chip or rename "Code" without
 * the change being intentional.
 */
class DefaultQuickChipsTest {
    @Test
    fun `default chips cover the five core use cases`() {
        val labels = DefaultQuickChips.map { it.label }
        assertEquals(5, labels.size, "expected 5 starter chips")
        assertTrue("Research" in labels)
        assertTrue("Code" in labels)
        assertTrue("Brainstorm" in labels)
        assertTrue("Rewrite" in labels)
        assertTrue("Memory" in labels)
    }

    @Test
    fun `every chip has non-empty prompt and label`() {
        DefaultQuickChips.forEach { chip ->
            assertTrue(chip.prompt.isNotBlank(), "prompt blank for ${chip.label}")
            assertTrue(chip.label.isNotBlank(), "label blank")
        }
    }

    @Test
    fun `labels are unique`() {
        val labels = DefaultQuickChips.map { it.label }
        assertEquals(labels.size, labels.toSet().size, "duplicate chip labels")
    }
}
