package com.aura.ui.components

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Lock the [DefaultQuickActions] list. We don't want the empty
 * state to silently drop a quick action or to swap their order
 * without test coverage — these are the first thing a new user
 * sees when they open the app cold.
 */
class DefaultQuickActionsTest {

    @Test
    fun `five quick actions, all unique titles`() {
        val titles = DefaultQuickActions.map { it.title }
        assertEquals(5, titles.size)
        assertEquals(titles.size, titles.toSet().size, "duplicate title: $titles")
    }

    @Test
    fun `each quick action has a non-empty prompt`() {
        for (action in DefaultQuickActions) {
            assertTrue(action.prompt.isNotBlank(), "empty prompt for ${action.title}")
            // A real prompt should have at least 4 words so the
            // user understands what they'll get.
            assertTrue(
                action.prompt.split(" ").size >= 4,
                "prompt too short for ${action.title}: ${action.prompt}",
            )
        }
    }

    @Test
    fun `each quick action has non-empty title and subtitle`() {
        for (action in DefaultQuickActions) {
            assertTrue(action.title.isNotBlank(), "empty title")
            assertTrue(action.subtitle.isNotBlank(), "empty subtitle for ${action.title}")
        }
    }

    @Test
    fun `research action mentions research so users connect icon to capability`() {
        val research = DefaultQuickActions.firstOrNull { it.title == "Research a topic" }
        assertTrue(research != null)
        assertTrue(
            research!!.prompt.contains("research", ignoreCase = true) ||
                research.prompt.contains("summary", ignoreCase = true),
            "Research quick action should mention research or summary in prompt",
        )
    }
}
