package com.aura.ui.components

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs

class ScreenStateTest {

    @Test
    fun `mapContent transforms only content`() {
        val mapped = ScreenState.Content(2).mapContent { it * 3 }
        assertEquals(6, assertIs<ScreenState.Content<Int>>(mapped).value)
    }

    @Test
    fun `mapContent preserves terminal non-content states`() {
        assertIs<ScreenState.Loading>(ScreenState.Loading.mapContent<Int, Int> { it })
        assertIs<ScreenState.Empty>(
            ScreenState.Empty("Nothing here", "Create the first item").mapContent<Int, Int> { it },
        )
        assertIs<ScreenState.Error>(
            ScreenState.Error("Could not load", "Check the connection").mapContent<Int, Int> { it },
        )
    }
}
