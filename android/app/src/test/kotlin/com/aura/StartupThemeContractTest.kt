package com.aura

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import com.aura.ui.theme.resolvesDarkTheme

class StartupThemeContractTest {

    @Test
    fun `explicit app theme overrides system theme for platform chrome`() {
        assertTrue(resolvesDarkTheme("dark", systemDark = false))
        assertFalse(resolvesDarkTheme("light", systemDark = true))
        assertTrue(resolvesDarkTheme("system", systemDark = true))
        assertFalse(resolvesDarkTheme("system", systemDark = false))
    }

    private fun themeFile(sourceSet: String): String {
        val candidates = listOf(
            File("src/main/res/$sourceSet/themes.xml"),
            File("app/src/main/res/$sourceSet/themes.xml"),
        )
        return candidates.first(File::exists).readText()
    }

    private fun item(xml: String, name: String): String =
        Regex("<item\\s+name=\"${Regex.escape(name)}\"[^>]*>([^<]+)</item>")
            .find(xml)?.groupValues?.get(1)?.trim().orEmpty()

    /**
     * Lock the launch theme contract for both light and dark system
     * modes. The app used to flash a light gray screen for 200-400ms
     * during cold start because the parent theme was
     * `android:Theme.Material.Light.NoActionBar` and the
     * `windowBackground` was a solid color. Now both buckets use the
     * `aura_splash` drawable (a vertical gradient matching the
     * launcher icon background) so the cold-start frame reads as part
     * of the brand, not a stock Android surface.
     *
     * The status bar still gets the per-bucket color so the system
     * chrome (clock, signal icons) reads correctly against the splash.
     */
    @Test
    fun `launch window follows light and dark resources`() {
        val light = themeFile("values")
        val dark = themeFile("values-night")

        // Both buckets must point at the brand splash drawable.
        assertEquals("@drawable/aura_splash", item(light, "android:windowBackground"))
        assertEquals("@drawable/aura_splash", item(dark, "android:windowBackground"))

        // Status bar color + icon tint are per-bucket.
        assertEquals("#F7F8FA", item(light, "android:statusBarColor"))
        assertEquals("true", item(light, "android:windowLightStatusBar"))
        assertEquals("#030303", item(dark, "android:statusBarColor"))
        assertEquals("false", item(dark, "android:windowLightStatusBar"))
    }

    @Test
    fun `app lock has no fixed startup spacer`() {
        val candidates = listOf(
            File("src/main/kotlin/com/aura/MainActivity.kt"),
            File("app/src/main/kotlin/com/aura/MainActivity.kt"),
        )
        val source = candidates.first(File::exists).readText()
        assertFalse(source.contains("Spacer(Modifier.height(120.dp))"))
    }
}
