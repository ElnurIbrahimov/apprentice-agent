package com.aura.ui.viewmodel

import com.aura.agent.Conversation
import com.aura.agent.Turn
import java.io.ByteArrayInputStream
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNull

class ChatMediaPolicyTest {

    @Test
    fun `completed vision tool turn reuses one id for call and result`() {
        val updated = Conversation(turns = listOf(Turn(user = "describe")))
            .attachCompletedToolTurn(
                id = "vision-1",
                name = "vision",
                arguments = "{}",
                result = "a mountain",
            )

        val toolTurn = updated.turns.last().toolTurns.single()
        assertEquals("vision-1", toolTurn.id)
        assertEquals("a mountain", toolTurn.result)
    }

    @Test
    fun `bounded audio read rejects one byte over limit`() {
        val input = ByteArrayInputStream(ByteArray(11) { it.toByte() })

        assertNull(readStreamWithinLimit(input, maxBytes = 10))
    }

    @Test
    fun `bounded audio read preserves data within limit`() {
        val expected = byteArrayOf(1, 2, 3, 4)

        assertContentEquals(expected, readStreamWithinLimit(ByteArrayInputStream(expected), 4))
    }

    @Test
    fun `image sample size keeps largest dimension near target`() {
        assertEquals(4, calculateImageSampleSize(width = 4032, height = 3024, target = 1024))
        assertEquals(1, calculateImageSampleSize(width = 800, height = 600, target = 1024))
    }
}
