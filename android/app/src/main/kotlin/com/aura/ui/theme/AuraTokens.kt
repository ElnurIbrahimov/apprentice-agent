package com.aura.ui.theme

import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color

/**
 * Aura design tokens — 1:1 with `D:\Aura\web\src\index.css` CSS custom
 * properties. Every color, border, glow, and surface elevation used by
 * Aura Web should be reachable from this object so the Android UI
 * matches the web UI token-for-token.
 *
 * Why an object + not a `ColorScheme` extension:
 * - We need surfaces 0–4 (5 elevation steps), not Material's 3.
 * - We need a `moodAccent` for proactive events.
 * - We need explicit glow rgba values for boxShadow usage.
 * - We need typography sizes that don't fit MaterialTheme's limited slots.
 */
object AuraTokens {

    // ── Dark theme (primary — Aura runs dark-first) ──────────────────────
    object Dark {
        // Surfaces
        val bgBase = Color(0xFF030303)
        val surface0 = Color(0xFF09090B)
        val surface1 = Color(0xFF121214)
        val surface2 = Color(0xFF1A1A1D)
        val surface3 = Color(0xFF1E1E21)
        val surface4 = Color(0xFF232326)

        // Panel for translucent panels
        val bgPanel = Color(0x66000000)            // rgba(20,20,22,0.4)
        val bgPanelHover = Color(0x0DFFFFFF)       // rgba(255,255,255,0.05)

        // Borders (4% / 8% / 16% white)
        val borderSubtle = Color(0x0AFFFFFF)
        val borderDefault = Color(0x14FFFFFF)
        val borderStrong = Color(0x29FFFFFF)
        val borderFocus = Color(0x33FFFFFF)

        // Text
        val textPrimary = Color(0xFFEDEDED)
        val textSecondary = Color(0xFFA1A1AA)
        val textTertiary = Color(0xFF6B6B6B)
        val textQuaternary = Color(0xFF444444)

        // Accent (vivid violet — same as web)
        val accentPurple = Color(0xFF8B5CF6)
        val accentBlue = Color(0xFF3B82F6)
        val accentGreen = Color(0xFF10A37F)

        // Mood accent (default = purple; can be overridden by mood)
        val moodAccent = accentPurple

        // Glows
        val glowPurple = Color(0x668B5CF6)         // rgba(139,92,246,0.4)
        val glowPurpleStrong = Color(0x998B5CF6)
        val glowBlue = Color(0x663B82F6)
        val glowGreen = Color(0x6610A37F)
        val glowYellow = Color(0x66EAB308)
        val glowRed = Color(0x66EF4444)
        val glowOrange = Color(0x66F97316)

        // AI semantic
        val aiThinking = Color(0x268B5CF6)
        val aiToolCall = Color(0x263B82F6)
        val aiToolResult = Color(0x2610A37F)
        val aiError = Color(0x26EF4444)

        // User bubble
        val userBubbleBg = Color(0xFFFFFFFF)
        val userBubbleText = Color(0xFF000000)
        val userBubbleShadow = Color(0x40000000)    // 25% black

        // Tooltip
        val tooltipBg = Color(0xFF232326)
        val tooltipShadow = Color(0x80000000)      // 50% black

        // Avatar
        val avatarGradientStart = Color(0x1FFFFFFF) // rgba(255,255,255,0.12)
        val avatarGradientEnd = Color(0x05FFFFFF)   // rgba(255,255,255,0.02)
        val avatarBorder = Color(0x14FFFFFF)        // 7% white
        val proactiveAvatarGradientStart = Color(0x33A855F7)
        val proactivePing = Color(0xBFF472B6)       // pink-400/75

        // Send button ready
        val sendReady = Color(0xFF7C3AED)
        val sendReadyShadow = Color(0x667C3AED)    // 40% purple

        // Mode chip backgrounds
        val modeSearch = Color(0x33A78BFA)         // rgba(167,139,250,0.2)
        val modeResearch = Color(0x3360A5FA)        // blue
        val modeDeepResearch = Color(0x33FB923C)    // orange
        val modeAgent = Color(0x3334D399)           // green
        val modeSwarm = Color(0x33F472B6)           // pink
        val modeCompare = Color(0x33FBBF24)         // yellow
        val modeDelegate = Color(0x33A78BFA)

        // Scrollbar
        val scrollbarThumb = Color(0xFF333333)
        val scrollbarThumbHover = Color(0xFF666666)
    }

    // ── Light theme (lower priority; matches web light) ──────────────────
    object Light {
        val bgBase = Color(0xFFFAFAFA)
        val surface0 = Color(0xFFFFFFFF)
        val surface1 = Color(0xFFF5F5F5)
        val surface2 = Color(0xFFEFEFEF)
        val surface3 = Color(0xFFE8E8E8)
        val surface4 = Color(0xFFDCDCDC)
        val bgPanel = Color(0x99FFFFFF)
        val bgPanelHover = Color(0x0A000000)
        val borderSubtle = Color(0x0A000000)
        val borderDefault = Color(0x14000000)
        val borderStrong = Color(0x29000000)
        val borderFocus = Color(0x33000000)
        val textPrimary = Color(0xFF1A1A1A)
        val textSecondary = Color(0xFF444444)
        val textTertiary = Color(0xFF6B6B6B)
        val textQuaternary = Color(0xFFA0A0A0)
        val accentPurple = Color(0xFF7C3AED)
        val accentBlue = Color(0xFF2563EB)
        val accentGreen = Color(0xFF059669)
        val moodAccent = accentPurple
        val userBubbleBg = Color(0xFF7C3AED)
        val userBubbleText = Color(0xFFFFFFFF)
        val userBubbleShadow = Color(0x337C3AED)
    }

    // ── Shared gradients (dark theme primary) ────────────────────────────
    val accentGradient: Brush = Brush.linearGradient(
        colors = listOf(Color(0xFF8B5CF6), Color(0xFF3B82F6)),
    )
    val userBubbleGradient: Brush = Brush.linearGradient(
        colors = listOf(Color(0xFFFFFFFF), Color(0xFFE5E5E5)),
    )
    val avatarGradient: Brush = Brush.linearGradient(
        colors = listOf(Color(0x1FFFFFFF), Color(0x05FFFFFF)),
    )
    val proactiveAvatarGradient: Brush = Brush.linearGradient(
        colors = listOf(Color(0x33A855F7), Color(0x05FFFFFF)),
    )
}
