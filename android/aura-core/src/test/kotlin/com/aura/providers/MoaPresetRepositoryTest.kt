package com.aura.providers

import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import kotlin.test.assertTrue

@RunWith(RobolectricTestRunner::class)
class MoaPresetRepositoryTest {

    @Test
    fun `bundled catalog contains no concrete model presets`() {
        val context = RuntimeEnvironment.getApplication().applicationContext
        val presets = MoaPresetRepository(context).loadPresets()

        assertTrue(presets.isEmpty())
    }
}
