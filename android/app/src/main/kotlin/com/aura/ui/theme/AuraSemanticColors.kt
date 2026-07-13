package com.aura.ui.theme

import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.ReadOnlyComposable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color

@Immutable
data class AuraSemanticColors(
    val background: Color,
    val surface0: Color,
    val surface1: Color,
    val surface2: Color,
    val surface3: Color,
    val borderSubtle: Color,
    val borderDefault: Color,
    val borderStrong: Color,
    val borderFocus: Color,
    val textPrimary: Color,
    val textSecondary: Color,
    val textTertiary: Color,
    val actionPrimary: Color,
    val onActionPrimary: Color,
    val actionDisabled: Color,
    val onActionDisabled: Color,
    val success: Color,
    val warning: Color,
    val error: Color,
    val info: Color,
    val riskReadOnly: Color,
    val riskWriteLocal: Color,
    val riskWriteRemote: Color,
    val userBubble: Color,
    val onUserBubble: Color,
    val assistantAccent: Color,
    val aiThinking: Color,
    val aiToolCall: Color,
    val aiToolResult: Color,
    val selection: Color,
    val streaming: Color,
    val scrim: Color,
)

val DarkAuraSemanticColors = AuraSemanticColors(
    background = Color(0xFF030303),
    surface0 = Color(0xFF09090B),
    surface1 = Color(0xFF121214),
    surface2 = Color(0xFF1A1A1D),
    surface3 = Color(0xFF232326),
    borderSubtle = Color(0x1FFFFFFF),
    borderDefault = Color(0x33FFFFFF),
    borderStrong = Color(0x52FFFFFF),
    borderFocus = Color(0xFF9A7AF8),
    textPrimary = Color(0xFFF4F4F5),
    textSecondary = Color(0xFFB5B5BD),
    textTertiary = Color(0xFF92929B),
    actionPrimary = Color(0xFF7C3AED),
    onActionPrimary = Color.White,
    actionDisabled = Color(0xFF29292E),
    onActionDisabled = Color(0xFF85858E),
    success = Color(0xFF34D399),
    warning = Color(0xFFFBBF24),
    error = Color(0xFFF87171),
    info = Color(0xFF60A5FA),
    riskReadOnly = Color(0xFF60A5FA),
    riskWriteLocal = Color(0xFFFBBF24),
    riskWriteRemote = Color(0xFFF87171),
    userBubble = Color(0xFFF4F4F5),
    onUserBubble = Color(0xFF09090B),
    assistantAccent = Color(0xFF9A7AF8),
    aiThinking = Color(0x409A7AF8),
    aiToolCall = Color(0x403B82F6),
    aiToolResult = Color(0x4034D399),
    selection = Color(0x669A7AF8),
    streaming = Color(0xFF34D399),
    scrim = Color(0xB3000000),
)

val LightAuraSemanticColors = AuraSemanticColors(
    background = Color(0xFFF8F8FA),
    surface0 = Color.White,
    surface1 = Color(0xFFF1F1F4),
    surface2 = Color(0xFFE7E7EC),
    surface3 = Color(0xFFDCDCE3),
    borderSubtle = Color(0x1F18181B),
    borderDefault = Color(0x3318181B),
    borderStrong = Color(0x5218181B),
    borderFocus = Color(0xFF6D28D9),
    textPrimary = Color(0xFF18181B),
    textSecondary = Color(0xFF44444B),
    textTertiary = Color(0xFF5E5E66),
    actionPrimary = Color(0xFF6D28D9),
    onActionPrimary = Color.White,
    actionDisabled = Color(0xFFE0E0E5),
    onActionDisabled = Color(0xFF686871),
    success = Color(0xFF047857),
    warning = Color(0xFF92400E),
    error = Color(0xFFB91C1C),
    info = Color(0xFF1D4ED8),
    riskReadOnly = Color(0xFF1D4ED8),
    riskWriteLocal = Color(0xFF92400E),
    riskWriteRemote = Color(0xFFB91C1C),
    userBubble = Color(0xFF6D28D9),
    onUserBubble = Color.White,
    assistantAccent = Color(0xFF6D28D9),
    aiThinking = Color(0x266D28D9),
    aiToolCall = Color(0x261D4ED8),
    aiToolResult = Color(0x26047857),
    selection = Color(0x336D28D9),
    streaming = Color(0xFF047857),
    scrim = Color(0x66000000),
)

val LocalAuraSemanticColors = staticCompositionLocalOf { DarkAuraSemanticColors }

object AuraThemeTokens {
    val colors: AuraSemanticColors
        @Composable
        @ReadOnlyComposable
        get() = LocalAuraSemanticColors.current
}
