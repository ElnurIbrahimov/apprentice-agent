package com.aura.ui.screens

import com.aura.hands.Hand
import kotlin.test.Test
import kotlin.test.assertEquals

class HandsScreenLogicTest {
    @Test
    fun `runtime inputs include declared defaults and unresolved step placeholders`() {
        val hand = Hand(
            id = "h1",
            name = "Weather",
            variables = """{"city":"Baku"}""",
            steps = """[{"tool":"weather","args":{"city":"{{ city }}","units":"{{units}}"}}]""",
        )

        assertEquals(
            linkedMapOf("city" to "Baku", "units" to ""),
            runtimeVariableInputs(hand),
        )
    }
}
