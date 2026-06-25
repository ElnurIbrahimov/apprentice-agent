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

private val AuraPrimary = Color(0xFF7C3AED)
private val AuraOnPrimary = Color(0xFFFFFFFF)
private val AuraPrimaryContainer = Color(0xFF1E1B4B)
private val AuraOnPrimaryContainer = Color(0xFFDDD6FE)
private val AuraSecondary = Color(0xFF06B6D4)
private val AuraTertiary = Color(0xFFF59E0B)
private val AuraBackground = Color(0xFFFAFAFA)
private val AuraSurface = Color(0xFFFFFFFF)
private val AuraOnSurface = Color(0xFF1F2937)

private val AuraDarkPrimary = Color(0xFFA78BFA)
private val AuraDarkOnPrimary = Color(0xFF1E1B4B)
private val AuraDarkPrimaryContainer = Color(0xFF3730A3)
private val AuraDarkOnPrimaryContainer = Color(0xFFEDE9FE)
private val AuraDarkSecondary = Color(0xFF22D3EE)
private val AuraDarkTertiary = Color(0xFFFBBF24)
private val AuraDarkBackground = Color(0xFF0A0A0A)
private val AuraDarkSurface = Color(0xFF111111)
private val AuraDarkOnSurface = Color(0xFFF3F4F6)

private val LightColors = lightColorScheme(
    primary = AuraPrimary,
    onPrimary = AuraOnPrimary,
    primaryContainer = AuraPrimaryContainer,
    onPrimaryContainer = AuraOnPrimaryContainer,
    secondary = AuraSecondary,
    tertiary = AuraTertiary,
    background = AuraBackground,
    surface = AuraSurface,
    onSurface = AuraOnSurface,
)

private val DarkColors = darkColorScheme(
    primary = AuraDarkPrimary,
    onPrimary = AuraDarkOnPrimary,
    primaryContainer = AuraDarkPrimaryContainer,
    onPrimaryContainer = AuraDarkOnPrimaryContainer,
    secondary = AuraDarkSecondary,
    tertiary = AuraDarkTertiary,
    background = AuraDarkBackground,
    surface = AuraDarkSurface,
    onSurface = AuraDarkOnSurface,
)

@Composable
fun AuraTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val context = LocalContext.current
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
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
