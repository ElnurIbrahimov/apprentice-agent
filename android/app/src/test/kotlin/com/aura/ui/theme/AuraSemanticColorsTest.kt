package com.aura.ui.theme

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import org.junit.Test
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class AuraSemanticColorsTest {

    @Test
    fun `dark and light palettes are semantically distinct`() {
        assertNotEquals(DarkAuraSemanticColors.background, LightAuraSemanticColors.background)
        assertNotEquals(DarkAuraSemanticColors.surface0, LightAuraSemanticColors.surface0)
        assertNotEquals(DarkAuraSemanticColors.textPrimary, LightAuraSemanticColors.textPrimary)
        assertNotEquals(DarkAuraSemanticColors.userBubble, LightAuraSemanticColors.userBubble)
    }

    @Test
    fun `normal semantic text meets WCAG AA contrast`() {
        listOf(DarkAuraSemanticColors, LightAuraSemanticColors).forEach { palette ->
            assertTrue(contrast(palette.textPrimary, palette.background) >= 4.5f)
            assertTrue(contrast(palette.textSecondary, palette.surface0) >= 4.5f)
            assertTrue(contrast(palette.textTertiary, palette.background) >= 4.5f)
            assertTrue(contrast(palette.onActionPrimary, palette.actionPrimary) >= 4.5f)
        }
    }

    @Test
    fun `status colors remain distinguishable from their surfaces`() {
        listOf(DarkAuraSemanticColors, LightAuraSemanticColors).forEach { palette ->
            assertTrue(contrast(palette.success, palette.background) >= 3f)
            assertTrue(contrast(palette.warning, palette.background) >= 3f)
            assertTrue(contrast(palette.error, palette.background) >= 3f)
            assertTrue(contrast(palette.info, palette.background) >= 3f)
        }
    }

    private fun contrast(foreground: Color, background: Color): Float {
        val lighter = maxOf(foreground.luminance(), background.luminance())
        val darker = minOf(foreground.luminance(), background.luminance())
        return (lighter + 0.05f) / (darker + 0.05f)
    }
}
