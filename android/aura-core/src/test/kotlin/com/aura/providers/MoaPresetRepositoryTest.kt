package com.aura.providers

import android.content.Context
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@RunWith(RobolectricTestRunner::class)
class MoaPresetRepositoryTest {

    @Test
    fun `loads presets from asset`() {
        val context = RuntimeEnvironment.getApplication().applicationContext
        val repo = MoaPresetRepository(context)
        val presets = repo.loadPresets()

        assertEquals(1, presets.size)
        assertTrue("default" in presets)
        val preset = presets.getValue("default")
        assertEquals(2, preset.referenceModels.size)
        assertEquals("deepseek", preset.aggregator.providerPrefix)
        assertEquals("deepseek-v4-pro:cloud", preset.aggregator.modelName)
        assertTrue(preset.enabled)
    }
}
