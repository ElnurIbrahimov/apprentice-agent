package com.aura.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

// ── Dark theme (primary — dark-first design) ──────────────────────────────────
// Deep charcoal background, teal accent, warm secondary. Feels premium
// and modern, not generic Material purple.

private val DarkPrimary = Color(0xFF2DD4BF)        // teal-400
private val DarkOnPrimary = Color(0xFF003731)
private val DarkPrimaryContainer = Color(0xFF00504A)
private val DarkOnPrimaryContainer = Color(0xFF6FF7E6)
private val DarkSecondary = Color(0xFFB0BEC5)       // blue-grey-200
private val DarkOnSecondary = Color(0xFF253238)
private val DarkSecondaryContainer = Color(0xFF37474F)
private val DarkOnSecondaryContainer = Color(0xFFCFD8DC)
private val DarkTertiary = Color(0xFFF9A825)        // amber-700
private val DarkOnTertiary = Color(0xFF3E2A00)
private val DarkTertiaryContainer = Color(0xFF573E00)
private val DarkOnTertiaryContainer = Color(0xFFFFD9A0)
private val DarkBackground = Color(0xFF0F0F10)       // near-black, not pure black
private val DarkSurface = Color(0xFF1A1A1C)         // slight elevation
private val DarkSurfaceVariant = Color(0xFF252528)
private val DarkOnSurfaceVariant = Color(0xFFCACACA)
private val DarkOnSurface = Color(0xFFEAEAEA)
private val DarkOutline = Color(0xFF3A3A3D)
private val DarkOutlineVariant = Color(0xFF2A2A2D)
private val DarkError = Color(0xFFEF5350)
private val DarkOnError = Color(0xFFFFFFFF)

// ── Light theme ───────────────────────────────────────────────────────────────

private val LightPrimary = Color(0xFF00897B)       // teal-600
private val LightOnPrimary = Color(0xFFFFFFFF)
private val LightPrimaryContainer = Color(0xFFB2DFDB)
private val LightOnPrimaryContainer = Color(0xFF004D45)
private val LightSecondary = Color(0xFF546E7A)       // blue-grey-600
private val LightOnSecondary = Color(0xFFFFFFFF)
private val LightSecondaryContainer = Color(0xFFCFD8DC)
private val LightOnSecondaryContainer = Color(0xFF1A2327)
private val LightTertiary = Color(0xFFE65100)        // orange-900
private val LightOnTertiary = Color(0xFFFFFFFF)
private val LightTertiaryContainer = Color(0xFFFFD180)
private val LightOnTertiaryContainer = Color(0xFF2E1A00)
private val LightBackground = Color(0xFFFAFAFA)
private val LightSurface = Color(0xFFFFFFFF)
private val LightSurfaceVariant = Color(0xFFECECEC)
private val LightOnSurfaceVariant = Color(0xFF444444)
private val LightOnSurface = Color(0xFF1A1A1A)
private val LightOutline = Color(0xFFCCCCCC)
private val LightOutlineVariant = Color(0xFFE0E0E0)
private val LightError = Color(0xFFD32F2F)
private val LightOnError = Color(0xFFFFFFFF)

private val DarkColors = darkColorScheme(
    primary = DarkPrimary,
    onPrimary = DarkOnPrimary,
    primaryContainer = DarkPrimaryContainer,
    onPrimaryContainer = DarkOnPrimaryContainer,
    secondary = DarkSecondary,
    onSecondary = DarkOnSecondary,
    secondaryContainer = DarkSecondaryContainer,
    onSecondaryContainer = DarkOnSecondaryContainer,
    tertiary = DarkTertiary,
    onTertiary = DarkOnTertiary,
    tertiaryContainer = DarkTertiaryContainer,
    onTertiaryContainer = DarkOnTertiaryContainer,
    background = DarkBackground,
    surface = DarkSurface,
    surfaceVariant = DarkSurfaceVariant,
    onSurfaceVariant = DarkOnSurfaceVariant,
    onSurface = DarkOnSurface,
    outline = DarkOutline,
    outlineVariant = DarkOutlineVariant,
    error = DarkError,
    onError = DarkOnError,
)

private val LightColors = lightColorScheme(
    primary = LightPrimary,
    onPrimary = LightOnPrimary,
    primaryContainer = LightPrimaryContainer,
    onPrimaryContainer = LightOnPrimaryContainer,
    secondary = LightSecondary,
    onSecondary = LightOnSecondary,
    secondaryContainer = LightSecondaryContainer,
    onSecondaryContainer = LightOnSecondaryContainer,
    tertiary = LightTertiary,
    onTertiary = LightOnTertiary,
    tertiaryContainer = LightTertiaryContainer,
    onTertiaryContainer = LightOnTertiaryContainer,
    background = LightBackground,
    surface = LightSurface,
    surfaceVariant = LightSurfaceVariant,
    onSurfaceVariant = LightOnSurfaceVariant,
    onSurface = LightOnSurface,
    outline = LightOutline,
    outlineVariant = LightOutlineVariant,
    error = LightError,
    onError = LightOnError,
)

@Composable
fun AuraTheme(
    themeMode: String = "system",
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val context = LocalContext.current
    val systemDark = isSystemInDarkTheme()
    val darkTheme = when (themeMode) {
        "light" -> false
        "dark" -> true
        else -> systemDark
    }
    // Dynamic color uses the device wallpaper — nice on Android 12+
    // but it can clash with our teal palette. Default off so the
    // Aura brand identity is consistent.
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && themeMode == "system" -> {
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }
    MaterialTheme(
        colorScheme = colorScheme,
        typography = AuraTypography,
        shapes = AuraShapes,
        content = content,
    )
}