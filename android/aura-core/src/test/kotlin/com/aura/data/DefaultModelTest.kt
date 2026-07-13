package com.aura.data

import org.junit.Test
import kotlin.test.assertEquals

class DefaultModelTest {

    @Test
    fun `model normalization preserves catalog id exactly`() {
        assertEquals(
            "test:primary-model",
            normalizeModelId("test:primary-model"),
        )
    }

    @Test
    fun `model normalization trims transport whitespace`() {
        assertEquals(
            "test:primary-model",
            normalizeModelId("  test:primary-model  "),
        )
    }
}
