package com.aura.ui.nav

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AuraBottomNavigationRouteTest {

    @Test
    fun `query parameters do not break top level selection`() {
        assertEquals("chat", normalizedBaseRoute("chat?draft=hello"))
        assertEquals("memory", normalizedBaseRoute("memory"))
        assertEquals(null, normalizedBaseRoute(null))
    }

    @Test
    fun `navigation exposes exactly four stable top level routes`() {
        assertEquals(listOf("home", "chat", "memory", "settings"), topLevelRoutes.map { it.route })
        assertTrue(topLevelRoutes.map { it.route }.toSet().size == topLevelRoutes.size)
    }
}
