package com.aura.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.aura.R

/**
 * Inter Display — a premium, distinctive typeface for the Aura brand.
 * Inter is the modern standard for tech products (used by Vercel,
 * Linear, GitHub, Figma). The "Display" variant has tighter spacing
 * and better legibility at small sizes than default Inter.
 */
val InterDisplay = FontFamily(
    Font(R.font.inter_regular, FontWeight.Normal),
    Font(R.font.inter_medium, FontWeight.Medium),
    Font(R.font.inter_semibold, FontWeight.SemiBold),
    Font(R.font.inter_bold, FontWeight.Bold),
)

val AuraTypography = Typography(
    displayLarge = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.Bold, fontSize = 36.sp, lineHeight = 44.sp, letterSpacing = (-0.5).sp),
    displayMedium = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.Bold, fontSize = 28.sp, lineHeight = 36.sp, letterSpacing = (-0.25).sp),
    headlineLarge = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.SemiBold, fontSize = 26.sp, lineHeight = 34.sp, letterSpacing = (-0.2).sp),
    headlineMedium = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.SemiBold, fontSize = 22.sp, lineHeight = 30.sp, letterSpacing = (-0.1).sp),
    titleLarge = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.SemiBold, fontSize = 19.sp, lineHeight = 26.sp, letterSpacing = (-0.1).sp),
    titleMedium = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.Medium, fontSize = 15.sp, lineHeight = 22.sp, letterSpacing = 0.sp),
    titleSmall = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.Medium, fontSize = 13.sp, lineHeight = 18.sp, letterSpacing = 0.1.sp),
    bodyLarge = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.Normal, fontSize = 15.sp, lineHeight = 23.sp, letterSpacing = 0.sp),
    bodyMedium = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.Normal, fontSize = 14.sp, lineHeight = 21.sp, letterSpacing = 0.1.sp),
    bodySmall = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.Normal, fontSize = 12.sp, lineHeight = 17.sp, letterSpacing = 0.3.sp),
    labelLarge = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.Medium, fontSize = 13.sp, lineHeight = 18.sp, letterSpacing = 0.3.sp),
    labelMedium = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.Medium, fontSize = 11.sp, lineHeight = 16.sp, letterSpacing = 0.4.sp),
    labelSmall = TextStyle(fontFamily = InterDisplay, fontWeight = FontWeight.Medium, fontSize = 10.sp, lineHeight = 14.sp, letterSpacing = 0.5.sp),
)