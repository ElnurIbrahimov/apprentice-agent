package com.aura.ui.theme

import org.junit.Test
import kotlin.test.assertEquals

class AuraDimensionsTest {

    @Test
    fun `compact shell dimensions match UX contract`() {
        assertEquals(16f, AuraDimensions.compactGutter.value)
        assertEquals(56f, AuraDimensions.topAppBarHeight.value)
        assertEquals(64f, AuraDimensions.bottomNavigationHeight.value)
        assertEquals(48f, AuraDimensions.minimumTouchTarget.value)
        assertEquals(600f, AuraDimensions.contentMaxWidth.value)
    }

    @Test
    fun `motion stays inside restrained timing budget`() {
        assertEquals(150, AuraDimensions.motionFastMs)
        assertEquals(200, AuraDimensions.motionStandardMs)
        assertEquals(220, AuraDimensions.motionSlowMs)
    }
}
