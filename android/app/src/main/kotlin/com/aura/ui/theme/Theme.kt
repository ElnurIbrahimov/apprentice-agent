package com.aura.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

/**
 * AuraTheme — wraps the Material 3 color scheme with the Aura design
 * tokens defined in [AuraTokens]. Dark theme is the default because
 * Aura runs dark-first, matching the Aura Web UI.
 *
 * Dynamic color is OFF by default. The Aura brand uses a fixed
 * violet/blue palette (matching the web `--accent-purple` #8B5CF6
 * and `--accent-blue` #3B82F6) and dynamic color pulls in
 * wallpaper-derived hues that clash with the brand identity. The
 * user can opt-in via Settings.
 */
@Composable
fun AuraTheme(
    themeMode: String = "system",
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit,
) {
    val context = LocalContext.current
    val systemDark = isSystemInDarkTheme()
    val darkTheme = when (themeMode) {
        "light" -> false
        "dark" -> true
        else -> systemDark
    }
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && themeMode == "system" -> {
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }
    val semanticColors = if (darkTheme) DarkAuraSemanticColors else LightAuraSemanticColors
    CompositionLocalProvider(LocalAuraSemanticColors provides semanticColors) {
        MaterialTheme(
            colorScheme = colorScheme,
            typography = AuraTypography,
            shapes = AuraShapes,
            content = content,
        )
    }
}

// ── Dark scheme — Aura brand (vivid violet on near-black) ─────────────────
private val DarkColors = darkColorScheme(
    primary = AuraTokens.Dark.accentPurple,
    onPrimary = Color.White,
    primaryContainer = AuraTokens.Dark.surface3,
    onPrimaryContainer = AuraTokens.Dark.accentPurple,
    secondary = AuraTokens.Dark.accentBlue,
    onSecondary = Color.White,
    secondaryContainer = AuraTokens.Dark.surface2,
    onSecondaryContainer = AuraTokens.Dark.accentBlue,
    tertiary = AuraTokens.Dark.accentGreen,
    onTertiary = Color.White,
    tertiaryContainer = AuraTokens.Dark.surface2,
    onTertiaryContainer = AuraTokens.Dark.accentGreen,
    background = AuraTokens.Dark.bgBase,
    onBackground = AuraTokens.Dark.textPrimary,
    surface = AuraTokens.Dark.surface1,
    onSurface = AuraTokens.Dark.textPrimary,
    surfaceVariant = AuraTokens.Dark.surface2,
    onSurfaceVariant = AuraTokens.Dark.textSecondary,
    surfaceTint = AuraTokens.Dark.accentPurple,
    inverseSurface = Color.White,
    inverseOnSurface = AuraTokens.Dark.surface0,
    inversePrimary = AuraTokens.Dark.accentPurple,
    outline = AuraTokens.Dark.borderDefault,
    outlineVariant = AuraTokens.Dark.borderSubtle,
    error = Color(0xFFEF4444),
    onError = Color.White,
    errorContainer = AuraTokens.Dark.aiError,
    onErrorContainer = Color(0xFFFCA5A5),
    scrim = Color(0x99000000),
)

// ── Light scheme (low priority — Aura runs dark-first) ───────────────────
private val LightColors = lightColorScheme(
    primary = AuraTokens.Light.accentPurple,
    onPrimary = Color.White,
    primaryContainer = AuraTokens.Light.surface1,
    onPrimaryContainer = AuraTokens.Light.accentPurple,
    secondary = AuraTokens.Light.accentBlue,
    onSecondary = Color.White,
    secondaryContainer = AuraTokens.Light.surface1,
    onSecondaryContainer = AuraTokens.Light.accentBlue,
    tertiary = AuraTokens.Light.accentGreen,
    onTertiary = Color.White,
    background = AuraTokens.Light.bgBase,
    onBackground = AuraTokens.Light.textPrimary,
    surface = AuraTokens.Light.surface0,
    onSurface = AuraTokens.Light.textPrimary,
    surfaceVariant = AuraTokens.Light.surface1,
    onSurfaceVariant = AuraTokens.Light.textSecondary,
    outline = AuraTokens.Light.borderDefault,
    outlineVariant = AuraTokens.Light.borderSubtle,
    error = Color(0xFFDC2626),
    onError = Color.White,
    scrim = Color(0x66000000),
)
