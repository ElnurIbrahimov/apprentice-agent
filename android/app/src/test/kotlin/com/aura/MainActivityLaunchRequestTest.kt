package com.aura

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class MainActivityLaunchRequestTest {
    @Test
    fun `brief payload implies chat and increments request sequence`() {
        val request = resolveAuraLaunchRequest(
            openChat = false,
            openMemory = false,
            morningBriefSummary = "Today: ship the draft",
            previousSequence = 4,
        )

        requireNotNull(request)
        assertEquals(5, request.sequence)
        assertTrue(request.openChat)
        assertFalse(request.openMemory)
        assertEquals("Today: ship the draft", request.morningBriefSummary)
    }

    @Test
    fun `memory request does not accidentally open chat`() {
        val request = resolveAuraLaunchRequest(
            openChat = false,
            openMemory = true,
            morningBriefSummary = null,
            previousSequence = 8,
        )

        requireNotNull(request)
        assertEquals(9, request.sequence)
        assertFalse(request.openChat)
        assertTrue(request.openMemory)
        assertNull(request.morningBriefSummary)
    }

    @Test
    fun `empty intent produces no navigation request`() {
        assertNull(
            resolveAuraLaunchRequest(
                openChat = false,
                openMemory = false,
                morningBriefSummary = "  ",
                previousSequence = 2,
            ),
        )
    }
}