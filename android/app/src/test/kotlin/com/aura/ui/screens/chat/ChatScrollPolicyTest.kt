package com.aura.ui.screens.chat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ChatScrollPolicyTest {
    @Test
    fun `user drag away from live edge detaches`() {
        assertTrue(shouldDetachFromLiveEdge(isUserDragging = true, physicallyAtLiveEdge = false))
        assertFalse(shouldDetachFromLiveEdge(isUserDragging = false, physicallyAtLiveEdge = false))
        assertFalse(shouldDetachFromLiveEdge(isUserDragging = true, physicallyAtLiveEdge = true))
    }

    @Test
    fun `jump only appears after detaching with content`() {
        assertTrue(shouldShowJumpToLatest(turnCount = 3, followLiveEdge = false))
        assertFalse(shouldShowJumpToLatest(turnCount = 3, followLiveEdge = true))
        assertFalse(shouldShowJumpToLatest(turnCount = 0, followLiveEdge = false))
    }

    @Test
    fun `tokens auto follow only while attached`() {
        assertTrue(shouldAutoFollow(turnCount = 1, followLiveEdge = true))
        assertFalse(shouldAutoFollow(turnCount = 1, followLiveEdge = false))
        assertFalse(shouldAutoFollow(turnCount = 0, followLiveEdge = true))
    }

    @Test
    fun `source provenance resolves the exact turn`() {
        assertEquals(1, findSourceTurnIndex(listOf(100L, 200L, 300L), 200L))
    }

    @Test
    fun `missing source provenance does not jump to a different turn`() {
        assertEquals(-1, findSourceTurnIndex(listOf(100L, 300L), 200L))
    }
}
