package com.aura.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.Dp
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraThemeTokens

/** Geometry-first loading placeholder. Callers pass the same height/fraction as final content. */
@Composable
fun AuraSkeleton(
    height: Dp,
    modifier: Modifier = Modifier,
    widthFraction: Float = 1f,
    testTag: kotlin.String? = null,
) {
    val colors = AuraThemeTokens.colors
    val tagged = if (testTag == null) modifier else modifier.testTag(testTag)
    Box(
        modifier = tagged
            .fillMaxWidth(widthFraction.coerceIn(0.1f, 1f))
            .height(height)
            .background(colors.surface1, RoundedCornerShape(AuraDimensions.cardRadius)),
    )
}
